import{_ as s,c as i,o as a,a4 as e}from"./chunks/framework.BSO0Jayu.js";const E=JSON.parse('{"title":"Switching between Deep Learning Frameworks","description":"","frontmatter":{},"headers":[],"relativePath":"api/Lux/switching_frameworks.md","filePath":"api/Lux/switching_frameworks.md","lastUpdated":null}'),t={name:"api/Lux/switching_frameworks.md"},l=e(`<h1 id="Switching-between-Deep-Learning-Frameworks" tabindex="-1">Switching between Deep Learning Frameworks <a class="header-anchor" href="#Switching-between-Deep-Learning-Frameworks" aria-label="Permalink to &quot;Switching between Deep Learning Frameworks {#Switching-between-Deep-Learning-Frameworks}&quot;">​</a></h1><h2 id="flux-to-lux-migrate-api" tabindex="-1">Flux Models to Lux Models <a class="header-anchor" href="#flux-to-lux-migrate-api" aria-label="Permalink to &quot;Flux Models to Lux Models {#flux-to-lux-migrate-api}&quot;">​</a></h2><p><code>Flux.jl</code> has been around in the Julia ecosystem for a long time and has a large userbase, hence we provide a way to convert <code>Flux</code> models to <code>Lux</code> models.</p><div class="tip custom-block"><p class="custom-block-title">Tip</p><p>Accessing these functions require manually loading <code>Flux</code>, i.e., <code>using Flux</code> must be present somewhere in the code for these to be used.</p></div><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Adapt.adapt-Tuple{FromFluxAdaptor, Any}" href="#Adapt.adapt-Tuple{FromFluxAdaptor, Any}">#</a> <b><u>Adapt.adapt</u></b> — <i>Method</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Adapt</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adapt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(from</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">FromFluxAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, L)</span></span></code></pre></div><p>Adapt a Flux model <code>l</code> to Lux model. See <a href="/api/Lux/switching_frameworks#Lux.FromFluxAdaptor"><code>FromFluxAdaptor</code></a> for more details.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/47b1f847e1a6d9a23f55aaa5ace539c14286c123/src/transform/flux.jl#L73-L77" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.FromFluxAdaptor" href="#Lux.FromFluxAdaptor">#</a> <b><u>Lux.FromFluxAdaptor</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">FromFluxAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(preserve_ps_st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, force_preserve</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Convert a Flux Model to Lux Model.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>This always ingores the <code>active</code> field of some of the Flux layers. This is almost never going to be supported.</p></div><p><strong>Keyword Arguments</strong></p><ul><li><p><code>preserve_ps_st</code>: Set to <code>true</code> to preserve the states and parameters of the l. This attempts the best possible way to preserve the original model. But it might fail. If you need to override possible failures, set <code>force_preserve</code> to <code>true</code>.</p></li><li><p><code>force_preserve</code>: Some of the transformations with state and parameters preservation haven&#39;t been implemented yet, in these cases, if <code>force_transform</code> is <code>false</code> a warning will be printed and a core Lux layer will be returned. Else, it will create a <a href="/api/Lux/switching_frameworks#Lux.FluxLayer"><code>FluxLayer</code></a>.</p></li></ul><p><strong>Example</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Flux</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Adapt, Lux, Metalhead, Random</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ResNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">18</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> adapt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">FromFluxAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), m</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">layers) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># or FromFluxAdaptor()(m.layers)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">224</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">224</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), m2);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">m2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span></code></pre></div><p><a href="https://github.com/LuxDL/Lux.jl/blob/47b1f847e1a6d9a23f55aaa5ace539c14286c123/src/transform/flux.jl#L1-L37" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.FluxLayer" href="#Lux.FluxLayer">#</a> <b><u>Lux.FluxLayer</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">FluxLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer)</span></span></code></pre></div><p>Serves as a compatibility layer between Flux and Lux. This uses <code>Optimisers.destructure</code> API internally.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>Lux was written to overcome the limitations of <code>destructure</code> + <code>Flux</code>. It is recommended to rewrite your l in Lux instead of using this layer.</p></div><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>Introducing this Layer in your model will lead to type instabilities, given the way <code>Optimisers.destructure</code> works.</p></div><p><strong>Arguments</strong></p><ul><li><code>layer</code>: Flux layer</li></ul><p><strong>Parameters</strong></p><ul><li><code>p</code>: Flattened parameters of the <code>layer</code></li></ul><p><a href="https://github.com/LuxDL/Lux.jl/blob/47b1f847e1a6d9a23f55aaa5ace539c14286c123/src/transform/flux.jl#L43-L66" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Lux-Models-to-Simple-Chains" tabindex="-1">Lux Models to Simple Chains <a class="header-anchor" href="#Lux-Models-to-Simple-Chains" aria-label="Permalink to &quot;Lux Models to Simple Chains {#Lux-Models-to-Simple-Chains}&quot;">​</a></h2><p><code>SimpleChains.jl</code> provides a way to train Small Neural Networks really fast on CPUs. See <a href="https://julialang.org/blog/2022/04/simple-chains/" target="_blank" rel="noreferrer">this blog post</a> for more details. This section describes how to convert <code>Lux</code> models to <code>SimpleChains</code> models while preserving the <a href="/manual/interface#lux-interface">layer interface</a>.</p><div class="tip custom-block"><p class="custom-block-title">Tip</p><p>Accessing these functions require manually loading <code>SimpleChains</code>, i.e., <code>using SimpleChains</code> must be present somewhere in the code for these to be used.</p></div><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Adapt.adapt-Tuple{ToSimpleChainsAdaptor, LuxCore.AbstractExplicitLayer}" href="#Adapt.adapt-Tuple{ToSimpleChainsAdaptor, LuxCore.AbstractExplicitLayer}">#</a> <b><u>Adapt.adapt</u></b> — <i>Method</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Adapt</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adapt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(from</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ToSimpleChainsAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, L</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractExplicitLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Adapt a Flux model to Lux model. See <a href="/api/Lux/switching_frameworks#Lux.ToSimpleChainsAdaptor"><code>ToSimpleChainsAdaptor</code></a> for more details.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/47b1f847e1a6d9a23f55aaa5ace539c14286c123/src/transform/simplechains.jl#L47-L51" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.ToSimpleChainsAdaptor" href="#Lux.ToSimpleChainsAdaptor">#</a> <b><u>Lux.ToSimpleChainsAdaptor</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ToSimpleChainsAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Adaptor for converting a Lux Model to SimpleChains. The returned model is still a Lux model, and satisfies the <code>AbstractExplicitLayer</code> interfacem but all internal calculations are performed using SimpleChains.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>There is no way to preserve trained parameters and states when converting to <code>SimpleChains.jl</code>.</p></div><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>Any kind of initialization function is not preserved when converting to <code>SimpleChains.jl</code>.</p></div><p><strong>Arguments</strong></p><ul><li><code>input_dims</code>: Tuple of input dimensions excluding the batch dimension. These must be of <code>static</code> type as <code>SimpleChains</code> expects.</li></ul><p><strong>Example</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SimpleChains</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> static</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Adapt, Lux, Random</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lux_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Conv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MaxPool</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">FlattenLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 84</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">84</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">adaptor </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ToSimpleChainsAdaptor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">static</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">static</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">static</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">simple_chains_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> adapt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(adaptor, lux_model) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># or adaptor(lux_model)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), simple_chains_model)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">28</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">simple_chains_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span></code></pre></div><p><a href="https://github.com/LuxDL/Lux.jl/blob/47b1f847e1a6d9a23f55aaa5ace539c14286c123/src/transform/simplechains.jl#L1-L42" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.SimpleChainsLayer" href="#Lux.SimpleChainsLayer">#</a> <b><u>Lux.SimpleChainsLayer</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SimpleChainsLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer)</span></span></code></pre></div><p>Wraps a <code>SimpleChains</code> layer into a <code>Lux</code> layer. All operations are performed using <code>SimpleChains</code> but the layer satisfies the <code>AbstractExplicitLayer</code> interface.</p><p><strong>Arguments</strong></p><ul><li><code>layer</code>: SimpleChains layer</li></ul><p><a href="https://github.com/LuxDL/Lux.jl/blob/47b1f847e1a6d9a23f55aaa5ace539c14286c123/src/transform/simplechains.jl#L79-L88" target="_blank" rel="noreferrer">source</a></p></div><br>`,19),n=[l];function p(h,r,k,d,o,c){return a(),i("div",null,n)}const u=s(t,[["render",p]]);export{E as __pageData,u as default};
