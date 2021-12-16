# Testing part 1

#include("../scripts-part1/part1.jl") # modify to include the correct script

include("../scripts-part1/diffusion3D_multixpu.jl")

xc, H = diffusion_3D()

# Add unit and reference tests


# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d = Dict(:X=> xc[inds], :H=>H[inds, inds, 15])

@testset "Ref-file" begin
    @test_reference "reftest-files/test_1.bson" d by=comp
end