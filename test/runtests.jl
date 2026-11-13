using Test
using LinearAlgebra
using Yancc

include(joinpath(@__DIR__, "golden", "layout.jl"))

@testset "Yancc" begin
    include("test_grids.jl")
    include("test_collisions.jl")
    include("test_dke.jl")
    include("test_tokamak.jl")
    include("test_batch.jl")
    # opt-in: needs an NVIDIA GPU and an environment with CUDA and CUDSS
    if get(ENV, "YANCC_TEST_GPU", "0") == "1"
        include("test_cuda_ext.jl")
    end
end
