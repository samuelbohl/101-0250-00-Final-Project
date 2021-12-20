# ~/.julia/bin/mpiexecjl -n 1 julia --project ./scripts-part1/scaling_experiment_multigpu.jl
# ~/.julia/bin/mpiexecjl -n 2 julia --project ./scripts-part1/scaling_experiment_multigpu.jl
# ~/.julia/bin/mpiexecjl -n 3 julia --project ./scripts-part1/scaling_experiment_multigpu.jl
# ~/.julia/bin/mpiexecjl -n 4 julia --project ./scripts-part1/scaling_experiment_multigpu.jl
using MAT

const USE_GPU = true
const BENCHMARK = true
const VISUALIZE = false
include("./diffusion3D_multixpu_perf.jl")

T_eff, t_toc, nprocs, me = diffusion_3D(256)
println("Scaling Experiment MultiXPU: $(T_eff)")

gr()
ENV["GKSwstype"] = "nul"

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
    plot([1,2,3,4], T_effs, xlabel="Number of GPUs", ylabel="T_eff (GB/s)", ylims=(0, extrema(T_effs)[2]), title="GPU Weak Scaling Experiment", label=false)
    png("$(@__DIR__)/../docs/img/scaling_experiment_mgpu_teff")
    plot([1,2,3,4], t_tocs[1]./t_tocs, ylims=(0, 1.0), xlabel="Number of GPUs", ylabel="Parallal Efficiency", title="GPU Weak Scaling Experiment", label=false)
    png("$(@__DIR__)/../docs/img/scaling_experiment_mgpu_pareff")
end



