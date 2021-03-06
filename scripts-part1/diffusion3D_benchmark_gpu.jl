# julia -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_gpu.jl

# Initialize Globals
const USE_GPU = true
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
max_bandwith = 360.0 # Theoretical Max Memory Bandwidth of RTX 3060

# Plot the Results
plot(grid_sizes, max_bandwith .* ones(5), label="Theoretical Max Bandwidth", lw=3)
plot!(grid_sizes, T_effs, xlabel="cbrt(grid size)", ylabel="T_eff GB/s", ylims=(0.0,  max_bandwith+5), title="GPU Scaling Experiment", label="diffusion3D_xpu.jl", lw=3)
png("$(@__DIR__)/../docs/img/diffusion3D_scaling_experiment_gpu")