# julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_work_precision.jl

# Initialize Globals
const USE_GPU = true
const BENCHMARK = false
const VISUALIZE = false
const STEADY = true

# Include main script
include("diffusion3D_xpu.jl")

# Grid sizes for the work precision diagram
grid_sizes = 16 * 2 .^ (1:5)
iters = zeros(5)

# local point for observation
local_point = (5,5,5)
lp_vals = zeros(5)

# loop over grid sizes
for it = 1:5
    # run simulation with current grid size
    iters[it], xc, H = diffusion_3D(grid_sizes[it])

    # get the index of the nearest local point of interest
    lp_index = findmin(abs.(xc.-local_point[1]))[2], findmin(abs.(xc.-local_point[2]))[2], findmin(abs.(xc.-local_point[3]))[2]
    # save the value
    lp_vals[it] = H[lp_index[1],lp_index[2],lp_index[3]]
end

# get the well converged solution
iter, xc, H_wc = diffusion_3D(128; tol=1e-24)

# tolerances for the work precision diagram
tols_exp = (6:23)
tols = (1.0)./((10.0).^tols_exp)

# allocate arrays for convergence behavior
rel_conv = zeros(length(tols))
abs_conv = zeros(length(tols))

# loop over tolerances
for it = 1:length(tols)
    # get the solution with current tolerance
    iter, xc, H_c = diffusion_3D(128; tol=tols[it])

    # Calculate absolute difference
    abs_diff = H_wc - H_c
    abs_conv[it] = norm(abs_diff)

    # Calculate relative difference
    rel_diff = norm(abs_diff)/norm(H_wc)
    rel_conv[it] = rel_diff
end

# Visualization of Results
gr()
ENV["GKSwstype"]="nul"

# Plot: Number of grid points VS. Iterations
plot(grid_sizes.^3, iters, xlabel="Number of grid points", xaxis=:log, ylabel="Iterations", label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_1")

plot(grid_sizes, iters, xlabel="cbrt(Number of grid points)", ylabel="Iterations", label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_2")

# Plot: Number of grid points VS. Value at local_point
plot(grid_sizes.^3, lp_vals, xlabel="Number of grid points", xaxis=:log, ylabel="Value at $(local_point)", label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_3")

plot(grid_sizes, lp_vals, xlabel="cbrt(Number of grid points)", ylabel="Value at $(local_point)", label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_4")

# Plot: Tolerance VS. Relative/Absolute difference to well converged solution
plot(tols_exp, rel_conv, xlabel="tolerance [1e-x]", ylabel="rel diff to well converged sol", yaxis=:log, label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_5")

plot(tols_exp, abs_conv, xlabel="tolerance [1e-x]", ylabel="abs diff to well converged sol", yaxis=:log, label=false)
png("$(@__DIR__)/../docs/img/diffusion3D_workprecision_6")

