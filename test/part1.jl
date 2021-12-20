# Testing part 1

#include("../scripts-part1/part1.jl") # modify to include the correct script

const USE_GPU = false
const BENCHMARK = false
const VISUALIZE = false

include("../scripts-part1/diffusion3D_xpu_perf.jl")


xc, H = diffusion_3D(32)

# Add unit and reference tests

# @testset "Testset nx,ny = [64, 128, 256, 512]" begin
#     C64 = diffusion_2D(64)[[5, Int(cld(0.6*64,1)), 54], Int(cld(0.5*64, 1))]'
#     C128 = diffusion_2D(128)[[5, Int(cld(0.6*128,1)), 118], Int(cld(0.5*128, 1))]'
#     C256 = diffusion_2D(256)[[5, Int(cld(0.6*256,1)), 246], Int(cld(0.5*256, 1))]'
#     C512 = diffusion_2D(512)[[5, Int(cld(0.6*512,1)), 502], Int(cld(0.5*512, 1))]'
#     @test C64 ≈ [1.28961441675812e-6 0.3403434055248243 0.000226725154067358]
#     @test C128 ≈ [1.42876853096198e-7 0.3606848631942946 2.784022638919167e-6]
#     @test C256 ≈ [3.82994869422046e-8 0.3515100977539851 2.070629144549965e-7]
#     @test C512 ≈ [1.56975129887789e-8 0.3467239448747831 4.938759153492403e-8]
# end


# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d = Dict(:X=> xc[inds], :H=>H[inds, inds, 15])

@testset "Ref-file" begin
    @test_reference "reftest-files/test_1.bson" d by=comp
end