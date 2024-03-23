


# Training a Simple LSTM {#Training-a-Simple-LSTM}

In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:
1. Create custom Lux models.
  
1. Become familiar with the Lux recurrent neural network API.
  
1. Training using Optimisers.jl and Zygote.jl.
  

## Package Imports {#Package-Imports}

```julia
using Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Random, Statistics
```


## Dataset {#Dataset}

We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.

```julia
function get_dataloaders(; dataset_size=1000, sequence_length=50)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(
                                 d[1][:, (sequence_length + 1):end], :, sequence_length, 1)
                             for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end
```


```
get_dataloaders (generic function with 1 method)
```


## Creating a Classifier {#Creating-a-Classifier}

We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model since it will contain a lstm block and a classifier head.

We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define `Lux.initialparameters` and `Lux.initialstates`.

To understand more about container layers, please look at [Container Layer](/manual/interface#Container-Layer).

```julia
struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end
```


We won't define the model from scratch but rather use the [`Lux.LSTMCell`](/api/Lux/layers#Lux.LSTMCell) and [`Lux.Dense`](/api/Lux/layers#Lux.Dense).

```julia
function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims), Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##225".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.

Now we need to define the behavior of the Classifier when it is invoked.

```julia
function (s::SpiralClassifier)(
        x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(Lux._eachslice(x, Val(2)))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    # After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    # Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end
```


## Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}

Now let's define the binarycrossentropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.

```julia
function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), y_pred, st
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


Finally lets create an optimiser given the model parameters.

```julia
function create_optimiser(ps)
    opt = Optimisers.Adam(0.01f0)
    return Optimisers.setup(opt, ps)
end
```


```
create_optimiser (generic function with 1 method)
```


## Training the Model {#Training-the-Model}

```julia
function main()
    # Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    # Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    dev = gpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)

    for epoch in 1:25
        # Train the model
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st), back = pullback(compute_loss, x, y, model, ps, st)
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        # Validate the model
        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss Accuracy $acc")
        end
    end

    return (ps, st) |> cpu_device()
end

ps_trained, st_trained = main()
```


```
┌ Warning: `replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`.
└ @ LuxCore ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxCore/t4mG0/src/LuxCore.jl:13
Epoch [1]: Loss 0.5617838
Epoch [1]: Loss 0.50694406
Epoch [1]: Loss 0.46752283
Epoch [1]: Loss 0.44967222
Epoch [1]: Loss 0.43701318
Epoch [1]: Loss 0.404436
Epoch [1]: Loss 0.38705772
Validation: Loss 0.3731279 Accuracy 1.0
Validation: Loss 0.36890095 Accuracy 1.0
Epoch [2]: Loss 0.3744645
Epoch [2]: Loss 0.34580004
Epoch [2]: Loss 0.32723597
Epoch [2]: Loss 0.3250464
Epoch [2]: Loss 0.2967051
Epoch [2]: Loss 0.28769827
Epoch [2]: Loss 0.26858088
Validation: Loss 0.26086766 Accuracy 1.0
Validation: Loss 0.25814652 Accuracy 1.0
Epoch [3]: Loss 0.257881
Epoch [3]: Loss 0.24425709
Epoch [3]: Loss 0.22889914
Epoch [3]: Loss 0.221697
Epoch [3]: Loss 0.21067034
Epoch [3]: Loss 0.19932003
Epoch [3]: Loss 0.18789057
Validation: Loss 0.18136394 Accuracy 1.0
Validation: Loss 0.17981496 Accuracy 1.0
Epoch [4]: Loss 0.17708877
Epoch [4]: Loss 0.17270245
Epoch [4]: Loss 0.16440842
Epoch [4]: Loss 0.15325171
Epoch [4]: Loss 0.14697716
Epoch [4]: Loss 0.14265165
Epoch [4]: Loss 0.12977241
Validation: Loss 0.12916318 Accuracy 1.0
Validation: Loss 0.12809679 Accuracy 1.0
Epoch [5]: Loss 0.12845057
Epoch [5]: Loss 0.12151758
Epoch [5]: Loss 0.11744191
Epoch [5]: Loss 0.11084199
Epoch [5]: Loss 0.105567336
Epoch [5]: Loss 0.10217978
Epoch [5]: Loss 0.099979
Validation: Loss 0.09419527 Accuracy 1.0
Validation: Loss 0.093192905 Accuracy 1.0
Epoch [6]: Loss 0.09491375
Epoch [6]: Loss 0.08690649
Epoch [6]: Loss 0.084235
Epoch [6]: Loss 0.08263468
Epoch [6]: Loss 0.07804661
Epoch [6]: Loss 0.07608506
Epoch [6]: Loss 0.070215076
Validation: Loss 0.06984544 Accuracy 1.0
Validation: Loss 0.06888022 Accuracy 1.0
Epoch [7]: Loss 0.06911827
Epoch [7]: Loss 0.065867305
Epoch [7]: Loss 0.063650094
Epoch [7]: Loss 0.062248416
Epoch [7]: Loss 0.056451026
Epoch [7]: Loss 0.054892324
Epoch [7]: Loss 0.05704314
Validation: Loss 0.052408855 Accuracy 1.0
Validation: Loss 0.051507127 Accuracy 1.0
Epoch [8]: Loss 0.04998064
Epoch [8]: Loss 0.050990228
Epoch [8]: Loss 0.047633596
Epoch [8]: Loss 0.044494696
Epoch [8]: Loss 0.043981597
Epoch [8]: Loss 0.043175988
Epoch [8]: Loss 0.0381772
Validation: Loss 0.039636202 Accuracy 1.0
Validation: Loss 0.038840882 Accuracy 1.0
Epoch [9]: Loss 0.038151808
Epoch [9]: Loss 0.037649233
Epoch [9]: Loss 0.037887987
Epoch [9]: Loss 0.033622134
Epoch [9]: Loss 0.032977432
Epoch [9]: Loss 0.03176272
Epoch [9]: Loss 0.02970782
Validation: Loss 0.03044242 Accuracy 1.0
Validation: Loss 0.029734025 Accuracy 1.0
Epoch [10]: Loss 0.030749768
Epoch [10]: Loss 0.02661626
Epoch [10]: Loss 0.028908512
Epoch [10]: Loss 0.025691176
Epoch [10]: Loss 0.02528986
Epoch [10]: Loss 0.026169363
Epoch [10]: Loss 0.023724344
Validation: Loss 0.024018344 Accuracy 1.0
Validation: Loss 0.023402583 Accuracy 1.0
Epoch [11]: Loss 0.021990037
Epoch [11]: Loss 0.023160774
Epoch [11]: Loss 0.02201864
Epoch [11]: Loss 0.021258485
Epoch [11]: Loss 0.020097775
Epoch [11]: Loss 0.020819396
Epoch [11]: Loss 0.02175207
Validation: Loss 0.019565038 Accuracy 1.0
Validation: Loss 0.01904281 Accuracy 1.0
Epoch [12]: Loss 0.018808749
Epoch [12]: Loss 0.01905261
Epoch [12]: Loss 0.016745511
Epoch [12]: Loss 0.018015318
Epoch [12]: Loss 0.016731951
Epoch [12]: Loss 0.017261356
Epoch [12]: Loss 0.017227922
Validation: Loss 0.016422339 Accuracy 1.0
Validation: Loss 0.015972843 Accuracy 1.0
Epoch [13]: Loss 0.015132045
Epoch [13]: Loss 0.016656332
Epoch [13]: Loss 0.014962329
Epoch [13]: Loss 0.014765747
Epoch [13]: Loss 0.01434944
Epoch [13]: Loss 0.014413631
Epoch [13]: Loss 0.014894309
Validation: Loss 0.01417277 Accuracy 1.0
Validation: Loss 0.013774357 Accuracy 1.0
Epoch [14]: Loss 0.013828257
Epoch [14]: Loss 0.014110817
Epoch [14]: Loss 0.013162319
Epoch [14]: Loss 0.012766699
Epoch [14]: Loss 0.012588069
Epoch [14]: Loss 0.012704199
Epoch [14]: Loss 0.010297785
Validation: Loss 0.012458872 Accuracy 1.0
Validation: Loss 0.012117846 Accuracy 1.0
Epoch [15]: Loss 0.012150814
Epoch [15]: Loss 0.0117537
Epoch [15]: Loss 0.011561595
Epoch [15]: Loss 0.010796378
Epoch [15]: Loss 0.012192611
Epoch [15]: Loss 0.011078331
Epoch [15]: Loss 0.011329126
Validation: Loss 0.011174276 Accuracy 1.0
Validation: Loss 0.010853187 Accuracy 1.0
Epoch [16]: Loss 0.011073467
Epoch [16]: Loss 0.010328382
Epoch [16]: Loss 0.0106158
Epoch [16]: Loss 0.010392973
Epoch [16]: Loss 0.010320656
Epoch [16]: Loss 0.009959584
Epoch [16]: Loss 0.009436324
Validation: Loss 0.010094302 Accuracy 1.0
Validation: Loss 0.009810004 Accuracy 1.0
Epoch [17]: Loss 0.009510262
Epoch [17]: Loss 0.009828018
Epoch [17]: Loss 0.009837867
Epoch [17]: Loss 0.0086804945
Epoch [17]: Loss 0.009613379
Epoch [17]: Loss 0.009065224
Epoch [17]: Loss 0.009959637
Validation: Loss 0.009227949 Accuracy 1.0
Validation: Loss 0.00895871 Accuracy 1.0
Epoch [18]: Loss 0.008649558
Epoch [18]: Loss 0.009064774
Epoch [18]: Loss 0.009089775
Epoch [18]: Loss 0.00814496
Epoch [18]: Loss 0.008537401
Epoch [18]: Loss 0.00837403
Epoch [18]: Loss 0.0085766595
Validation: Loss 0.008465059 Accuracy 1.0
Validation: Loss 0.008223541 Accuracy 1.0
Epoch [19]: Loss 0.008217361
Epoch [19]: Loss 0.008080833
Epoch [19]: Loss 0.00835396
Epoch [19]: Loss 0.008018145
Epoch [19]: Loss 0.0076180613
Epoch [19]: Loss 0.00756696
Epoch [19]: Loss 0.00748879
Validation: Loss 0.007834196 Accuracy 1.0
Validation: Loss 0.007601344 Accuracy 1.0
Epoch [20]: Loss 0.0073088147
Epoch [20]: Loss 0.0076688593
Epoch [20]: Loss 0.007164187
Epoch [20]: Loss 0.0075491616
Epoch [20]: Loss 0.007225069
Epoch [20]: Loss 0.0075183855
Epoch [20]: Loss 0.006258921
Validation: Loss 0.007260983 Accuracy 1.0
Validation: Loss 0.0070499647 Accuracy 1.0
Epoch [21]: Loss 0.0072765225
Epoch [21]: Loss 0.006447955
Epoch [21]: Loss 0.006856644
Epoch [21]: Loss 0.006434706
Epoch [21]: Loss 0.006936591
Epoch [21]: Loss 0.0073496792
Epoch [21]: Loss 0.0058302633
Validation: Loss 0.006777958 Accuracy 1.0
Validation: Loss 0.0065760934 Accuracy 1.0
Epoch [22]: Loss 0.0070049986
Epoch [22]: Loss 0.0062750485
Epoch [22]: Loss 0.0064058322
Epoch [22]: Loss 0.006334737
Epoch [22]: Loss 0.0060944567
Epoch [22]: Loss 0.0061284807
Epoch [22]: Loss 0.0067774192
Validation: Loss 0.0063386145 Accuracy 1.0
Validation: Loss 0.0061513623 Accuracy 1.0
Epoch [23]: Loss 0.0057366686
Epoch [23]: Loss 0.005977152
Epoch [23]: Loss 0.0060129054
Epoch [23]: Loss 0.0062564537
Epoch [23]: Loss 0.006071605
Epoch [23]: Loss 0.0059350315
Epoch [23]: Loss 0.0056401533
Validation: Loss 0.0059511038 Accuracy 1.0
Validation: Loss 0.005773206 Accuracy 1.0
Epoch [24]: Loss 0.005562652
Epoch [24]: Loss 0.005699288
Epoch [24]: Loss 0.0053489897
Epoch [24]: Loss 0.0056724143
Epoch [24]: Loss 0.005372773
Epoch [24]: Loss 0.0060118064
Epoch [24]: Loss 0.005812697
Validation: Loss 0.0056022103 Accuracy 1.0
Validation: Loss 0.0054336716 Accuracy 1.0
Epoch [25]: Loss 0.0051853876
Epoch [25]: Loss 0.005615293
Epoch [25]: Loss 0.0057526673
Epoch [25]: Loss 0.0050906083
Epoch [25]: Loss 0.00518959
Epoch [25]: Loss 0.004885157
Epoch [25]: Loss 0.005511411
Validation: Loss 0.0052832137 Accuracy 1.0
Validation: Loss 0.005124004 Accuracy 1.0

```


## Saving the Model {#Saving-the-Model}

We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don't save the model

```julia
@save "trained_model.jld2" {compress = true} ps_trained st_trained
```


Let's try loading the model

```julia
@load "trained_model.jld2" ps_trained st_trained
```


```
2-element Vector{Symbol}:
 :ps_trained
 :st_trained
```


## Appendix {#Appendix}

```julia
using InteractiveUtils
InteractiveUtils.versioninfo()
if @isdefined(LuxCUDA) && CUDA.functional(); println(); CUDA.versioninfo(); end
if @isdefined(LuxAMDGPU) && LuxAMDGPU.functional(); println(); AMDGPU.versioninfo(); end
```


```
Julia Version 1.10.2
Commit bd47eca2c8a (2024-03-01 10:14 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 48 × AMD EPYC 7402 24-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-2/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

CUDA runtime 12.3, artifact installation
CUDA driver 12.4
NVIDIA driver 550.54.14

CUDA libraries: 
- CUBLAS: 12.3.4
- CURAND: 10.3.4
- CUFFT: 11.0.12
- CUSOLVER: 11.5.4
- CUSPARSE: 12.2.0
- CUPTI: 21.0.0
- NVML: 12.0.0+550.54.14

Julia packages: 
- CUDA: 5.2.0
- CUDA_Driver_jll: 0.7.0+1
- CUDA_Runtime_jll: 0.11.1+0

Toolchain:
- Julia: 1.10.2
- LLVM: 15.0.7

Environment:
- JULIA_CUDA_HARD_MEMORY_LIMIT: 25%

1 device:
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.391 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```



---


_This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl)._
