#using FinalProjectRepo
using Test, ReferenceTests, BSON

# make sure to turn off GPU usage, at least for Github Actions

include("part1.jl")
ParallelStencil.@reset_parallel_stencil()
include("part2.jl")
