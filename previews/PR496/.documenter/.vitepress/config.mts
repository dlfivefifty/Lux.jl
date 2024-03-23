import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import { transformerMetaWordHighlight } from '@shikijs/transformers';

// https://vitepress.dev/reference/site-config
export default defineConfig({
    base: '/',// TODO: replace this in makedocs!
    title: 'Lux',
    description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    lastUpdated: true,
    cleanUrls: true,
    outDir: '../final_site', // This is required for MarkdownVitepress to work correctly...
    head: [['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }]],

    markdown: {
        math: true,
        config(md) {
            md.use(tabsMarkdownPlugin),
                md.use(mathjax3),
                md.use(footnote)
        },
        theme: {
            light: "github-light",
            dark: "github-dark"
        },
        codeTransformers: [transformerMetaWordHighlight(),],

    },
    themeConfig: {
        outline: 'deep',
        // https://vitepress.dev/reference/default-theme-config
        logo: { src: '/logo.png', width: 24, height: 24},
        search: {
            provider: 'local',
            options: {
                detailedView: true
            }
        },
        nav: [
            { text: 'Home', link: '/' },
            { text: 'Getting Started', link: '/introduction/index' },
            { text: 'Ecosystem', link: '/ecosystem' },
            { text: 'Tutorials', link: '/tutorials/index' },
            { text: 'Manual', link: '/manual/interface' },
            { text: 'API', link: '/api/Lux/layers' }
        ],

        sidebar: [
{ text: 'Lux.jl', link: '/index' },
{ text: 'Ecosystem', link: '/ecosystem' },
{ text: 'Getting Started', collapsed: false, items: [
{ text: 'Introduction', link: '/introduction/index' },
{ text: 'Overview', link: '/introduction/overview' },
{ text: 'Resources', link: '/introduction/resources' },
{ text: 'Citation', link: '/introduction/citation' }]
 },
{ text: 'Tutorials', collapsed: false, items: [
{ text: 'Overview', link: '/tutorials/index' },
{ text: 'Beginner', collapsed: false, items: [
{ text: 'Julia & Lux for the Uninitiated', link: '/tutorials/beginner/1_Basics' },
{ text: 'Fitting a Polynomial using MLP', link: '/tutorials/beginner/2_PolynomialFitting' },
{ text: 'Training a Simple LSTM', link: '/tutorials/beginner/3_SimpleRNN' },
{ text: 'MNIST Classification with SimpleChains', link: '/tutorials/beginner/4_SimpleChains' }]
 },
{ text: 'Intermediate', collapsed: false, items: [
{ text: 'MNIST Classification using Neural ODEs', link: '/tutorials/intermediate/1_NeuralODE' },
{ text: 'Bayesian Neural Network', link: '/tutorials/intermediate/2_BayesianNN' },
{ text: 'Training a HyperNetwork on MNIST and FashionMNIST', link: '/tutorials/intermediate/3_HyperNet' }]
 },
{ text: 'Advanced', collapsed: false, items: [
{ text: 'Training a Neural ODE to Model Gravitational Waveforms', link: '/tutorials/advanced/1_GravitationalWaveForm' }]
 }]
 },
{ text: 'Manual', collapsed: false, items: [
{ text: 'Markdown.Link(Any["Lux Interface"], "@id lux-interface")', link: '/manual/interface' },
{ text: 'Debugging Lux Models', link: '/manual/debugging' },
{ text: 'Dispatching on Custom Input Types', link: '/manual/dispatch_custom_input' },
{ text: 'Markdown.Link(Any["Freezing Model Parameters"], "@id freezing-model-parameters")', link: '/manual/freezing_model_parameters' },
{ text: 'GPU Management', link: '/manual/gpu_management' },
{ text: 'Markdown.Link(Any["Migrating from Flux to Lux"], "@id migrate-from-flux")', link: '/manual/migrate_from_flux' },
{ text: 'Initializing Weights', link: '/manual/weight_initializers' }]
 },
{ text: 'API Reference', collapsed: false, items: [
{ text: 'Lux', collapsed: false, items: [
{ text: 'Built-In Layers', link: '/api/Lux/layers' },
{ text: 'Utilities', link: '/api/Lux/utilities' },
{ text: 'Experimental Features', link: '/api/Lux/contrib' },
{ text: 'Switching between Deep Learning Frameworks', link: '/api/Lux/switching_frameworks' }]
 },
{ text: 'Accelerator Support', collapsed: false, items: [
{ text: 'LuxAMDGPU', link: '/api/Accelerator_Support/LuxAMDGPU' },
{ text: 'LuxCUDA', link: '/api/Accelerator_Support/LuxCUDA' },
{ text: 'Markdown.Link(Any["LuxDeviceUtils"], "@id LuxDeviceUtils-API")', link: '/api/Accelerator_Support/LuxDeviceUtils' }]
 },
{ text: 'Building Blocks', collapsed: false, items: [
{ text: 'LuxCore', link: '/api/Building_Blocks/LuxCore' },
{ text: 'LuxLib', link: '/api/Building_Blocks/LuxLib' },
{ text: 'Markdown.Link(Any["WeightInitializers"], "@id WeightInitializers-API")', link: '/api/Building_Blocks/WeightInitializers' }]
 },
{ text: 'Domain Specific Modeling', collapsed: false, items: [
{ text: 'Boltz', link: '/api/Domain_Specific_Modeling/Boltz' }]
 },
{ text: 'Testing Functionality', collapsed: false, items: [
{ text: 'LuxTestUtils', link: '/api/Testing_Functionality/LuxTestUtils' }]
 }]
 }
]
,
        editLink: { pattern: "github.com/LuxDL/Lux.jl/edit/main/docs/src/:path" },
        socialLinks: [
            { icon: 'github', link: 'https://github.com/LuxDL/Lux.jl' }
        ],
        footer: {
            message: 'Made with <a href="https://documenter.juliadocs.org/stable/" target="_blank"><strong>Documenter.jl</strong></a>, <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a> and <a href="https://luxdl.github.io/DocumenterVitepress.jl/stable" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>Released under the MIT License. Powered by the <a href="https://www.julialang.org">Julia Programming Language</a>.<br>',
            copyright: `© Copyright ${new Date().getUTCFullYear()} Avik Pal.`
        },
        head: [
            [
                "script",
                { async: "", src: "https://www.googletagmanager.com/gtag/js?id=G-Q8GYTEVTZ2" },
            ],
            [
                "script",
                {},
                `window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('js', new Date());
              gtag('config', 'G-Q8GYTEVTZ2');`,
            ],
        ],
    }
})