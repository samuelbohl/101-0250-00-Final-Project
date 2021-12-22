const USE_GPU = true
const BENCHMARK = false
const VISUALIZE = false
const STEADY = true


include("diffusion3D_xpu.jl")

# Grid sizes for the work presicion diagram
grid_sizes = 16 * 2 .^ (1:5)
iters = zeros(5)

# tolerances for the work presicion diagram
tols = (1.0)./((10.0).^(3:15))

for it = 1:5
    iters[it] = diffusion_3D(grid_sizes[it])
end

# TODO: REST of work precision diagram

# Visualization of Results
gr()
ENV["GKSwstype"]="nul"

plot(grid_sizes.^3, iters, xlabel="Number of grid points", ylabel="Iterations", label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_1")

plot(grid_sizes, iters, xlabel="cbrt(Number of grid points)", ylabel="Iterations", label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_2")

