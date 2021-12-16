# ~/.julia/bin/mpiexecjl -n 1 julia --project ./scripts-part1/scaling_experiment_multixpu.jl
using CUDA

const USE_GPU = true
include("./diffusion3D_multixpu_perf.jl")


T_eff = diffusion_3D(grid=(128,128,128), is_experiment=true)
println("Scaling Experiment MultiXPU: $(T_eff)")
