# Testing part 2

include("../scripts-part2/elastic_wave_3D_testing.jl")


# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d = Dict(:X=> xc[inds], :H=>P[inds, inds, 15])

@testset "Reference Test - Elastic Wave 3D" begin
    @test_reference "reftest-files/test_2.bson" d by=comp
end
