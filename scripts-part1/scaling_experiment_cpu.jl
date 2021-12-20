# julia -t 1 ./scripts-part1/scaling_experiment_cpu.jl
# julia -t 4 ./scripts-part1/scaling_experiment_cpu.jl
# julia -t 8 ./scripts-part1/scaling_experiment_cpu.jl
# julia -t 16 ./scripts-part1/scaling_experiment_cpu.jl
const USE_GPU = false
const BENCHMARK = true
const VISUALIZE = false
include("./diffusion3D_xpu_perf.jl")

# Grid sizes for the scaling experiment
grid_sizes = 16 * 2 .^ (1:5)

# Array allocation
T_effs = zeros(5)

# Scaling experimetn loop
for it = 1:5
    T_effs[it] = diffusion_3D(grid_sizes[it])
end

# Visualization of Results
gr()
max_bandwith = 50.0 # Theoretical Max Memory Bandwidth of Intel Core i7-11700
num_threads = Threads.nthreads()

plot(grid_sizes, max_bandwith .* ones(5), label="Theoretical Max Bandwidth")
plot!(grid_sizes, T_effs, xlabel="cbrt(grid size)", ylabel="T_eff GB/s", title="CPU Scaling Experiment $(num_threads) Thread(s)", label="diffusion3D_xpu_perf.jl")
png("$(@__DIR__)/../docs/img/diffusion3D_xpu_perf_scaling_experiment_cpu_$(num_threads)threads")