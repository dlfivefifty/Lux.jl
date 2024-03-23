
# Why we wrote Lux? {#Why-we-wrote-Lux?}

Julia already has quite a few well established Neural Network Frameworks – [Flux](https://fluxml.ai/) & [KNet](https://denizyuret.github.io/Knet.jl/latest/). However, certain design elements – **Coupled Model and Parameters** & **Internal Mutations** – associated with these frameworks make them less compiler and user friendly. Making changes to address these problems in the respective frameworks would be too disruptive for users. Here comes in `Lux`: a neural network framework built completely using pure functions to make it both compiler and autodiff friendly.

## Design Principles {#Design-Principles}
- **Layers must be immutable** – cannot store any parameter/state but rather store the information to construct them
  
- **Layers are pure functions**
  
- **Layers return a Tuple containing the result and the updated state**
  
- **Given same inputs the outputs must be same** – yes this must hold true even for stochastic functions. Randomness must be controlled using `rng`s passed in the state.
  
- **Easily extensible**
  

## Why use Lux over Flux? {#Why-use-Lux-over-Flux?}
- **Neural Networks for SciML**: For SciML Applications (Neural ODEs, Deep Equilibrium Models) solvers typically expect a monolithic parameter vector. Flux enables this via its `destructure` mechanism, but `destructure` comes with various [edge cases and limitations](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.destructure). Lux forces users to make an explicit distinction between state variables and parameter variables to avoid these issues. Also, it comes battery-included for distributed training using [FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl) _(I know :P the naming)_
  
- **Sensible display of Custom Layers** – Ever wanted to see Pytorch like Network printouts or wondered how to extend the pretty printing of Flux's layers? Lux handles all of that by default.
  
- **Truly immutable models** - No _unexpected internal mutations_ since all layers are implemented as pure functions. All layers are also _deterministic_ given the parameters and state: if a layer is supposed to be stochastic (say `Dropout`), the state must contain a seed which is then updated after the function call.
  
- **Easy Parameter Manipulation** – By separating parameter data and layer structures, Lux makes implementing `WeightNorm`, `SpectralNorm`, etc. downright trivial. Without this separation, it is much harder to pass such parameters around without mutations which AD systems don't like.
  

## Why not use Lux? {#Why-not-use-Lux?}
- **Small Neural Networks on CPU** – Lux is developed for training large neural networks. For smaller architectures, we recommend using [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl).
  
- **Lux won't magically speed up your code (yet)** – Lux shares the same backend with Flux and so if your primary desire to shift is driven by performance, you will be disappointed.
  
- **XLA Support** – Lux doesn't compile to XLA which means no TPU support unfortunately.
  
