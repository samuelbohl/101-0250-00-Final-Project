# ~/.julia/bin/mpiexecjl -n 1 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl
# ~/.julia/bin/mpiexecjl -n 2 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl
# ~/.julia/bin/mpiexecjl -n 3 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl
# ~/.julia/bin/mpiexecjl -n 4 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl
using MAT

# Initialize Globals
const USE_GPU = true
const BENCHMARK = true
const VISUALIZE = false

# Include main script
include("./diffusion3D_multixpu.jl")

# Run simulation
T_eff, t_toc, nprocs, me = diffusion_3D(256)
println("Scaling Experiment MultiXPU: $(T_eff)")

# Prepare visualization
gr()
ENV["GKSwstype"] = "nul"

# Run visualization on process 0
if me == 0
    # read file
    file = matopen("$(@__DIR__)/../docs/img/scaling_mgpu.mat")
    T_effs = read(file, "T_effs")
    t_tocs = read(file, "t_tocs")
    close(file)

    # update Teff of this run
    T_effs[nprocs] = T_eff
    t_tocs[nprocs] = t_toc

    # write back to file
    file = matopen("$(@__DIR__)/../docs/img/scaling_mgpu.mat", "w")
    write(file, "T_effs", T_effs)
    write(file, "t_tocs", t_tocs)
    close(file)

    # plot
    plot([1,2,3,4], T_effs, xlabel="Number of GPUs", ylabel="T_eff (GB/s)", ylims=(0, extrema(T_effs)[2]), title="GPU Weak Scaling Experiment", label=false, lw=3)
    png("$(@__DIR__)/../docs/img/scaling_experiment_mgpu_teff")
    plot([1,2,3,4], t_tocs[1]./t_tocs, ylims=(0, 1.0), xlabel="Number of GPUs", ylabel="Parallal Efficiency", title="GPU Weak Scaling Experiment", label=false, lw=3)
    png("$(@__DIR__)/../docs/img/scaling_experiment_mgpu_pareff")
end



