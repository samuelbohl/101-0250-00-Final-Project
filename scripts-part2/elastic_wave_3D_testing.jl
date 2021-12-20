const USE_GPU = false
const VISUALIZE = false
include("elastic_wave_3D.jl")

xc, P = elastic_wave_3D(32, 1.0)