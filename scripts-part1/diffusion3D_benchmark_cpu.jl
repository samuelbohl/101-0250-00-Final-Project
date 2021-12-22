# julia -O3 --check-bounds=no -t 1 ./scripts-part1/diffusion3D_benchmark_cpu.jl
# julia -O3 --check-bounds=no -t 4 ./scripts-part1/diffusion3D_benchmark_cpu.jl
# julia -O3 --check-bounds=no -t 8 ./scripts-part1/diffusion3D_benchmark_cpu.jl
# julia -O3 --check-bounds=no -t 16 ./scripts-part1/diffusion3D_benchmark_cpu.jl

# Initialize Globals
const USE_GPU = false
const BENCHMARK = true
const VISUALIZE = false

# Include main script
include("./diffusion3D_xpu.jl")

# Grid sizes for the scaling experiment
grid_sizes = 16 * 2 .^ (1:5)

# Array allocation
T_effs = zeros(5)

# Scaling experiment loop
for it = 1:5
    T_effs[it] = diffusion_3D(grid_sizes[it])
end

# Prepare visualisation
gr()
ENV["GKSwstype"]="nul"

# System variables 
max_bandwith = 50.0 # Theoretical Max Memory Bandwidth of Intel Core i7-11700
num_threads = Threads.nthreads()

# Plot the results
plot(grid_sizes, max_bandwith .* ones(5), label="Theoretical Max Bandwidth", lw=3)
plot!(grid_sizes, T_effs, xlabel="cbrt(grid size)", ylabel="T_eff GB/s", ylims=(0.0, max_bandwith+5), title="CPU Scaling Experiment $(num_threads) Thread(s)", label="diffusion3D_xpu.jl", lw=3)
png("$(@__DIR__)/../docs/img/diffusion3D_scaling_experiment_cpu_$(num_threads)threads")