using Test
using LinearAlgebra
using Yancc

include(joinpath(@__DIR__, "golden", "layout.jl"))

@testset "Yancc" begin
    include("test_grids.jl")
    include("test_collisions.jl")
    include("test_dke.jl")
    include("test_tokamak.jl")
end
