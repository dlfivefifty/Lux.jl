


# Boltz {#Boltz}

Accelerate ⚡ your ML research using pre-built Deep Learning Models with Lux.

## Index {#Index}
- [`Boltz.ClassTokens`](#Boltz.ClassTokens)
- [`Boltz.MultiHeadAttention`](#Boltz.MultiHeadAttention)
- [`Boltz.ViPosEmbedding`](#Boltz.ViPosEmbedding)
- [`Boltz._fast_chunk`](#Boltz._fast_chunk)
- [`Boltz._flatten_spatial`](#Boltz._flatten_spatial)
- [`Boltz._seconddimmean`](#Boltz._seconddimmean)
- [`Boltz._vgg_block`](#Boltz._vgg_block)
- [`Boltz._vgg_classifier_layers`](#Boltz._vgg_classifier_layers)
- [`Boltz._vgg_convolutional_layers`](#Boltz._vgg_convolutional_layers)
- [`Boltz.transformer_encoder`](#Boltz.transformer_encoder)
- [`Boltz.vgg`](#Boltz.vgg)


# Computer Vision Models {#Computer-Vision-Models}

## Classification Models: Native Lux Models {#Classification-Models:-Native-Lux-Models}

|         MODEL NAME |             FUNCTION |        NAME | PRETRAINED | TOP 1 ACCURACY (%) | TOP 5 ACCURACY (%) |
| ------------------:| --------------------:| -----------:|:----------:|:------------------:|:------------------:|
|                VGG |                `vgg` |    `:vgg11` |     ✅      |       67.35        |       87.91        |
|                VGG |                `vgg` |    `:vgg13` |     ✅      |       68.40        |       88.48        |
|                VGG |                `vgg` |    `:vgg16` |     ✅      |       70.24        |       89.80        |
|                VGG |                `vgg` |    `:vgg19` |     ✅      |       71.09        |       90.27        |
|                VGG |                `vgg` | `:vgg11_bn` |     ✅      |       69.09        |       88.94        |
|                VGG |                `vgg` | `:vgg13_bn` |     ✅      |       69.66        |       89.49        |
|                VGG |                `vgg` | `:vgg16_bn` |     ✅      |       72.11        |       91.02        |
|                VGG |                `vgg` | `:vgg19_bn` |     ✅      |       72.95        |       91.32        |
| Vision Transformer | `vision_transformer` |     `:tiny` |     🚫      |                    |                    |
| Vision Transformer | `vision_transformer` |    `:small` |     🚫      |                    |                    |
| Vision Transformer | `vision_transformer` |     `:base` |     🚫      |                    |                    |
| Vision Transformer | `vision_transformer` |    `:large` |     🚫      |                    |                    |
| Vision Transformer | `vision_transformer` |     `:huge` |     🚫      |                    |                    |
| Vision Transformer | `vision_transformer` |    `:giant` |     🚫      |                    |                    |
| Vision Transformer | `vision_transformer` | `:gigantic` |     🚫      |                    |                    |


## Building Blocks {#Building-Blocks}
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz.ClassTokens' href='#Boltz.ClassTokens'>#</a>&nbsp;<b><u>Boltz.ClassTokens</u></b> &mdash; <i>Type</i>.




```julia
ClassTokens(dim; init=Lux.zeros32)
```


Appends class tokens to an input with embedding dimension `dim` for use in many vision transformer namels.


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vit.jl#L59-L64)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz.MultiHeadAttention' href='#Boltz.MultiHeadAttention'>#</a>&nbsp;<b><u>Boltz.MultiHeadAttention</u></b> &mdash; <i>Type</i>.




```julia
MultiHeadAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
                   attention_dropout_rate::T=0.0f0,
                   projection_dropout_rate::T=0.0f0) where {T}
```


Multi-head self-attention layer


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vit.jl#L2-L8)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz.ViPosEmbedding' href='#Boltz.ViPosEmbedding'>#</a>&nbsp;<b><u>Boltz.ViPosEmbedding</u></b> &mdash; <i>Type</i>.




```julia
ViPosEmbedding(embedsize, npatches;
               init = (rng, dims...) -> randn(rng, Float32, dims...))
```


Positional embedding layer used by many vision transformer-like namels.


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vit.jl#L83-L88)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz.transformer_encoder' href='#Boltz.transformer_encoder'>#</a>&nbsp;<b><u>Boltz.transformer_encoder</u></b> &mdash; <i>Function</i>.




```julia
transformer_encoder(in_planes, depth, number_heads; mlp_ratio = 4.0f0, dropout = 0.0f0)
```


Transformer as used in the base ViT architecture. ([reference](https://arxiv.org/abs/2010.11929)).

**Arguments**
- `in_planes`: number of input channels
  
- `depth`: number of attention blocks
  
- `number_heads`: number of attention heads
  
- `mlp_ratio`: ratio of MLP layers to the number of input channels
  
- `dropout_rate`: dropout rate
  


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vit.jl#L106-L119)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz.vgg' href='#Boltz.vgg'>#</a>&nbsp;<b><u>Boltz.vgg</u></b> &mdash; <i>Function</i>.




```julia
vgg(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)
```


Create a VGG model ([reference](https://arxiv.org/abs/1409.1556v6)).

**Arguments**
- `imsize`: input image width and height as a tuple
  
- `config`: the configuration for the convolution layers
  
- `inchannels`: number of input channels
  
- `batchnorm`: set to `true` to use batch normalization after each convolution
  
- `nclasses`: number of output classes
  
- `fcsize`: intermediate fully connected layer size (see `Metalhead._vgg_classifier_layers`)
  
- `dropout`: dropout level between fully connected layers
  


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vgg.jl#L67-L82)

</div>
<br>

### Non-Public API {#Non-Public-API}
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz._seconddimmean' href='#Boltz._seconddimmean'>#</a>&nbsp;<b><u>Boltz._seconddimmean</u></b> &mdash; <i>Function</i>.




```julia
_seconddimmean(x)
```


Computes the mean of `x` along dimension `2`


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/utils.jl#L27-L31)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz._fast_chunk' href='#Boltz._fast_chunk'>#</a>&nbsp;<b><u>Boltz._fast_chunk</u></b> &mdash; <i>Function</i>.




```julia
_fast_chunk(x::AbstractArray, ::Val{n}, ::Val{dim})
```


Type-stable and faster version of `MLUtils.chunk`


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/utils.jl#L1-L5)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz._flatten_spatial' href='#Boltz._flatten_spatial'>#</a>&nbsp;<b><u>Boltz._flatten_spatial</u></b> &mdash; <i>Function</i>.




```julia
_flatten_spatial(x::AbstractArray{T, 4})
```


Flattens the first 2 dimensions of `x`, and permutes the remaining dimensions to (2, 1, 3)


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/utils.jl#L18-L22)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz._vgg_block' href='#Boltz._vgg_block'>#</a>&nbsp;<b><u>Boltz._vgg_block</u></b> &mdash; <i>Function</i>.




```julia
_vgg_block(input_filters, output_filters, depth, batchnorm)
```


A VGG block of convolution layers ([reference](https://arxiv.org/abs/1409.1556v6)).

**Arguments**
- `input_filters`: number of input feature maps
  
- `output_filters`: number of output feature maps
  
- `depth`: number of convolution/convolution + batch norm layers
  
- `batchnorm`: set to `true` to include batch normalization after each convolution
  


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vgg.jl#L1-L12)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz._vgg_classifier_layers' href='#Boltz._vgg_classifier_layers'>#</a>&nbsp;<b><u>Boltz._vgg_classifier_layers</u></b> &mdash; <i>Function</i>.




```julia
_vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
```


Create VGG classifier (fully connected) layers ([reference](https://arxiv.org/abs/1409.1556v6)).

**Arguments**
- `imsize`: tuple `(width, height, channels)` indicating the size after the convolution layers (see `Metalhead._vgg_convolutional_layers`)
  
- `nclasses`: number of output classes
  
- `fcsize`: input and output size of the intermediate fully connected layer
  
- `dropout`: the dropout level between each fully connected layer
  


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vgg.jl#L48-L61)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='Boltz._vgg_convolutional_layers' href='#Boltz._vgg_convolutional_layers'>#</a>&nbsp;<b><u>Boltz._vgg_convolutional_layers</u></b> &mdash; <i>Function</i>.




```julia
_vgg_convolutional_layers(config, batchnorm, inchannels)
```


Create VGG convolution layers ([reference](https://arxiv.org/abs/1409.1556v6)).

**Arguments**
- `config`: vector of tuples `(output_channels, num_convolutions)` for each block (see `Metalhead._vgg_block`)
  
- `batchnorm`: set to `true` to include batch normalization after each convolution
  
- `inchannels`: number of input channels
  


[source](https://github.com/LuxDL/Boltz.jl/blob/v0.3.3/src/vision/vgg.jl#L25-L36)

</div>
<br>

## Classification Models: Imported from Metalhead.jl {#Classification-Models:-Imported-from-Metalhead.jl}

::: tip Tip

You need to load `Flux` and `Metalhead` before using these models.

:::

| MODEL NAME |    FUNCTION |                  NAME | PRETRAINED | TOP 1 ACCURACY (%) | TOP 5 ACCURACY (%) |
| ----------:| -----------:| ---------------------:|:----------:|:------------------:|:------------------:|
|    AlexNet |   `alexnet` |            `:alexnet` |     ✅      |       54.48        |       77.72        |
|     ResNet |    `resnet` |           `:resnet18` |     🚫      |       68.08        |       88.44        |
|     ResNet |    `resnet` |           `:resnet34` |     🚫      |       72.13        |       90.91        |
|     ResNet |    `resnet` |           `:resnet50` |     🚫      |       74.55        |       92.36        |
|     ResNet |    `resnet` |          `:resnet101` |     🚫      |       74.81        |       92.36        |
|     ResNet |    `resnet` |          `:resnet152` |     🚫      |       77.63        |       93.84        |
|  ConvMixer | `convmixer` |              `:small` |     🚫      |                    |                    |
|  ConvMixer | `convmixer` |               `:base` |     🚫      |                    |                    |
|  ConvMixer | `convmixer` |              `:large` |     🚫      |                    |                    |
|   DenseNet |  `densenet` |        `:densenet121` |     🚫      |                    |                    |
|   DenseNet |  `densenet` |        `:densenet161` |     🚫      |                    |                    |
|   DenseNet |  `densenet` |        `:densenet169` |     🚫      |                    |                    |
|   DenseNet |  `densenet` |        `:densenet201` |     🚫      |                    |                    |
|  GoogleNet | `googlenet` |          `:googlenet` |     🚫      |                    |                    |
|  MobileNet | `mobilenet` |       `:mobilenet_v1` |     🚫      |                    |                    |
|  MobileNet | `mobilenet` |       `:mobilenet_v2` |     🚫      |                    |                    |
|  MobileNet | `mobilenet` | `:mobilenet_v3_small` |     🚫      |                    |                    |
|  MobileNet | `mobilenet` | `:mobilenet_v3_large` |     🚫      |                    |                    |
|    ResNeXT |   `resnext` |          `:resnext50` |     🚫      |                    |                    |
|    ResNeXT |   `resnext` |         `:resnext101` |     🚫      |                    |                    |
|    ResNeXT |   `resnext` |         `:resnext152` |     🚫      |                    |                    |


These models can be created using `<FUNCTION>(<NAME>; pretrained = <PRETRAINED>)`

### Preprocessing {#Preprocessing}

All the pretrained models require that the images be normalized with the parameters `mean = [0.485f0, 0.456f0, 0.406f0]` and `std = [0.229f0, 0.224f0, 0.225f0]`.
