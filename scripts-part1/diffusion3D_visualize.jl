const USE_GPU = true
const BENCHMARK = false
const VISUALIZE = true

include("./diffusion3D_multixpu.jl")

diffusion_3D(256)