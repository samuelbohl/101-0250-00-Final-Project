# Testing part 1

# Initialize Globals
const USE_GPU = false
const BENCHMARK = false
const VISUALIZE = false

# Run the XPU script
include("../scripts-part1/diffusion3D_xpu.jl")
xc, H = diffusion_3D(32)

# Run the Multi XPU script
ParallelStencil.@reset_parallel_stencil()
include("../scripts-part1/diffusion3D_multixpu.jl")
xc2, H2 = diffusion_3D(32)

# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d_xpu = Dict(:X=> xc[inds], :H=>H[inds, inds, 15])
d_mxpu = Dict(:X=> xc2[inds], :H=>H2[inds, inds, 15])

@testset "Reference Test - Diffusion 3D" begin
    @test_reference "reftest-files/test_1.bson" d_xpu by=comp
    @test_reference "reftest-files/test_1.bson" d_mxpu by=comp
end
