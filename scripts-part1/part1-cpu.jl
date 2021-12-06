
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots, Printf, ImplicitGlobalGrid
using Distributed
import MPI

const USE_GPU = false
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

include("part1.jl")

diffusion_2D(; do_visu=true)
