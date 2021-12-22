const USE_GPU = true
const VISUALIZE = false
const BENCHMARK = true
include("elastic_wave_3D.jl")

# Grid sizes for the scaling experiment
grid_size = 16 * 2 .^ (1:5)

# Array allocation
T_effs = zeros(5)

# Scaling experiment loop
for it = 1:5
    T_effs[it] = elastic_wave_3D(grid_size[it], 8192.0 / 8^it)
end

# Visualization of Results
gr()
max_bandwith = 360.0 # Theoretical Max Memory Bandwidth of NVIDIA GeForce RTX 3060

plot(grid_size, max_bandwith .* ones(5), label="Theoretical Max Bandwidth", lw=3)
plot!(grid_size, T_effs, xlabel="grid_size", ylabel="T_eff GB/s", ylims=(0.0, 400.0), title="GPU Scaling Experiment", label="elastic_wave_3D.jl", lw=3)
png("./docs/img/elastic_wave_3D_scaling_experiment_gpu")