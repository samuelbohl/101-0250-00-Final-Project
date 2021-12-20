const USE_GPU = false
include("elastic_wave_3D.jl")

xc, P = elastic_wave_3D(32, 1.0, false)