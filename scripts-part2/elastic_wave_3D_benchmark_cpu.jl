const USE_GPU = false
const VISUALIZE = false
const BENCHMARK = true
include("elastic_wave_3D.jl")

# Grid sizes for the scaling experiment
grid_size = 16 * 2 .^ (1:5)

# Array allocation
T_effs = zeros(5)

# Scaling experiment loop
for it = 1:5
    T_effs[it] = elastic_wave_3D(grid_size[it], 4096.0 / 8^it)
end

# Visualization of Results
gr()
max_bandwith = 50.0 # Theoretical Max Memory Bandwidth of Intel Core i7-11700
num_threads = Threads.nthreads()

plot(grid_size, max_bandwith .* ones(5), label="Theoretical Max Bandwidth")
plot!(grid_size, T_effs, xlabel="grid_size", ylabel="T_eff GB/s", ylims=(0.0, max_bandwith), title="CPU Scaling Experiment $(num_threads) Thread(s)", label="elastic_wave_3D.jl", lw=3)
png("./docs/img/elastic_wave_3D_scaling_experiment_cpu_$(num_threads)threads")