const USE_GPU = false
const VISUALIZE = false
const BENCHMARK = true
include("elastic_wave_3D.jl")

xc, P = elastic_wave_3D(128, 20.0)