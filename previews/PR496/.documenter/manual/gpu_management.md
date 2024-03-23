
# GPU Management {#GPU-Management}

::: info Info

Starting from `v0.5`, Lux has transitioned to a new GPU management system. The old system using `cpu` and `gpu` functions is still in place but will be removed in `v0.6`. Using the  old functions might lead to performance regressions if used inside performance critical code.

:::

`Lux.jl` can handle multiple GPU backends. Currently, the following backends are supported:

```julia
using Lux, LuxCUDA, LuxAMDGPU  # Important to load trigger packages

supported_gpu_backends()
```


```
("CUDA", "AMDGPU", "Metal")
```


::: danger Metal Support

Support for Metal GPUs should be considered extremely experimental at this point.

:::

## Automatic Backend Management (Recommended Approach) {#Automatic-Backend-Management-(Recommended-Approach)}

Automatic Backend Management is done by two simple functions: `cpu_device` and `gpu_device`.
1. [`cpu_device`](/api/Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.cpu_device): This is a simple function and just returns a `LuxCPUDevice` object.
  

```julia
cdev = cpu_device()
```


```
(::LuxCPUDevice) (generic function with 5 methods)
```


```julia
x_cpu = randn(Float32, 3, 2)
```


```
3×2 Matrix{Float32}:
  0.433884   0.229779
 -0.459193  -1.95972
 -0.541064  -1.40102
```

1. [`gpu_device`](/api/Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.gpu_device): This function performs automatic GPU device selection and returns an object.
  1. If no GPU is available, it returns a `LuxCPUDevice` object.
    
  1. If a LocalPreferences file is present, then the backend specified in the file is used. To set a backend, use `Lux.gpu_backend!(<backend_name>)`.
    1. If the trigger package corresponding to the device is not loaded, then a warning is displayed.
      
    1. If no LocalPreferences file is present, then the first working GPU with loaded trigger package is used.
      
    
  

```julia
gdev = gpu_device()

x_gpu = x_cpu |> gdev
```


```
3×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
  0.433884   0.229779
 -0.459193  -1.95972
 -0.541064  -1.40102
```


```julia
(x_gpu |> cdev) ≈ x_cpu
```


```
true
```


## Manual Backend Management {#Manual-Backend-Management}

Automatic Device Selection can be circumvented by directly using `LuxCPUDevice` and `AbstractLuxGPUDevice` objects.

```julia
cdev = LuxCPUDevice()

x_cpu = randn(Float32, 3, 2)

if LuxCUDA.functional()
    gdev = LuxCUDADevice()
    x_gpu = x_cpu |> gdev
elseif LuxAMDGPU.functional()
    gdev = LuxAMDGPUDevice()
    x_gpu = x_cpu |> gdev
else
    @info "No GPU is available. Using CPU."
    x_gpu = x_cpu
end

(x_gpu |> cdev) ≈ x_cpu
```


```
true
```

