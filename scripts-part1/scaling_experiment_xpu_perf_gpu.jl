# julia ./scripts-part1/scaling_experiment_xpu_perf_gpu.jl
const USE_GPU = true
include("./diffusion3D_xpu_perf.jl")

# Grid sizes for the scaling experiment
grid_sizes = 16 * 2 .^ (1:5)

# Array allocation
T_effs = zeros(5)

# Scaling experiment loop
for it = 1:5
    grid_tuple = grid_sizes[it], grid_sizes[it], grid_sizes[it]
    T_effs[it] = diffusion_3D(;grid=grid_tuple, is_experiment=true)
end

# Visualization of Results
gr()
max_bandwith = 360.0 # Theoretical Max Memory Bandwidth of RTX 3060

plot(grid_sizes, max_bandwith .* ones(5), label="Theoretical Max Bandwidth")
plot!(grid_sizes, T_effs, xlabel="cbrt(grid size)", ylabel="T_eff GB/s", title="GPU Scaling Experiment", label="diffusion3D_xpu_perf.jl")
png("$(@__DIR__)/../docs/img/diffusion3D_xpu_perf_scaling_experiment_gpu")