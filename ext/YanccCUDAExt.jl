# NVIDIA GPU backend of the batched tokamak solver (`solve_dke_tokamak_batch(...;
# device=:cuda)`): the device methods of the seams in src/batch/backend.jl.
#
#   - dense and sparse data are uploaded as `CuArray` and `CuSparseMatrixCSR`, after
#     which the operator, the Krylov iteration and the restriction and prolongation of
#     the preconditioner run on the device through broadcasts, cuBLAS and cuSPARSE;
#   - the small products per case go through `gemm_strided_batched!`;
#   - the plane and coarse matrices of all cases are factored by dense blocks
#     (src/batch/block_coarse.jl) in batched cuBLAS kernels; where a grid is too small
#     for blocks, or when asked for, as uniform sparse batches by cuDSS.
#
# Only methods for `Val{:cuda}` and `CuArray` arguments are added here.
module YanccCUDAExt

using Yancc
using CUDA
using CUDA: cuBLAS, cuSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

import Yancc: _to_device, _default_batchsize, _batched_gemm!, _batched_inverse!, _factor_batch, _solve_batch!, _synchronize, _release,
    _default_coarse_solver, _default_plane_solver, _free!

function __init__()
    Yancc._DEVICE_FUNCTIONAL[:cuda] = () -> CUDA.functional()
    return nothing
end

_to_device(::Val{:cuda}, A::Array{Float64}) = CuArray(A)
_to_device(::Val{:cuda}, A::Array{Int}) = CuArray(A)
_to_device(::Val{:cuda}, A::SparseMatrixCSC{Float64}) = cuSPARSE.CuSparseMatrixCSR(A)

# Bytes of device memory per unknown of one case: the Krylov basis of a typical solve,
# the work arrays of the operator and the preconditioner, the factors and the arrays
# that live only while they are built. With cuDSS for the planes about 4000 at
# 65 × 41 × 6 × 3 unknowns; with blocks of size `sb` and a border of `nbd` for the
# planes 3700 at 65 × 41 (sb = 82) and 4600 at 257 × 61 (sb = 122) with two pitch angles
# a block (w = 1), measured, which `8 (250 + (1 + 2w) sb + 2 nbd)` bounds.
const BYTES_PER_UNKNOWN = Ref(8 * 600)
const BYTES_PER_UNKNOWN_BESIDES_PLANES = Ref(8 * 250)

function _default_batchsize(::Val{:cuda}, ncases::Integer, unknowns::Integer, restart::Integer, planes)
    budget = CUDA.free_memory() ÷ 2
    perunknown = planes === nothing ? BYTES_PER_UNKNOWN[] :
                 BYTES_PER_UNKNOWN_BESIDES_PLANES[] + 8 * ((1 + 2planes.w) * planes.sb + 2planes.nbd)
    return clamp(budget ÷ (perunknown * unknowns), 1, ncases)
end

const CuStrided3 = Union{CuArray{Float64,3},SubArray{Float64,3,<:CuArray}}

function _batched_gemm!(tA::Char, tB::Char, α::Real, A::CuStrided3, B::CuStrided3, β::Real, C::CuStrided3)
    cuBLAS.gemm_strided_batched!(tA, tB, Float64(α), A, B, Float64(β), C)
    return C
end

_synchronize(::Val{:cuda}) = CUDA.synchronize()
# The arrays of a chunk are freed one by one (_free!), so a quick collection of the young
# objects is enough between chunks; a full one took a fifth of a chunk.
_release(::Val{:cuda}) = (GC.gc(false); nothing)
_free!(x::CuArray) = CUDA.unsafe_free!(x)
_default_coarse_solver(::Val{:cuda}) = :block
# Dense blocks of one pitch angle do more arithmetic than a sparse LU and win on all but
# small planes: for whole solves on an A100 the two are even at 65 × 41 (2700 unknowns a
# plane), blocks are 1.3 times faster at 129 × 41 and 2.2 times at 257 × 61.
const MIN_BLOCK_PLANE = Ref(2500)
_default_plane_solver(::Val{:cuda}, op) = prod(Yancc.state_size(op)[1:2]) >= MIN_BLOCK_PLANE[] ? :block : :sparse

function _batched_inverse!(C::CuArray{Float64,3}, A::CuArray{Float64,3})
    pivots, info, _ = cuBLAS.getrf_strided_batched!(A, true)
    cuBLAS.getri_strided_batched!(A, C, pivots)
    return C
end

# LU factors of a uniform batch, held by cuDSS. It factors without pivoting: accurate
# to rounding for the plane matrices, to about 1e-6 (1e-3 at worst) for ill conditioned
# ones, which is enough inside a preconditioner. Single precision is not: it has given
# NaN for plane matrices.
struct CudssLUBatch{S}
    solver::S
    m::Int
    nbatch::Int
end

const REORDERING = Ref("algo3")   # approximate minimum degree: faster here than the default

function _factor_batch(::Val{:cuda}, A::Yancc.UniformSparseBatch{<:CuMatrix{Float64}})
    nbatch = length(A)
    solver = CudssSolver(CuVector{Cint}(A.rowptr), CuVector{Cint}(A.colval), vec(A.nzval), "G", 'F')
    nbatch > 1 && cudss_set(solver, "ubatch_size", nbatch)
    cudss_set(solver, "reordering_alg", REORDERING[])
    x, b = CudssMatrix(Float64, A.m; nbatch), CudssMatrix(Float64, A.m; nbatch)
    cudss("analysis", solver, x, b; asynchronous=false)
    cudss("factorization", solver, x, b; asynchronous=false)
    return CudssLUBatch(solver, A.m, nbatch)
end

_free!(F::CudssLUBatch) = (finalize(F.solver.data); nothing)   # the factors live in cuDSS's data

function _solve_batch!(X::CuMatrix{Float64}, F::CudssLUBatch, R::CuMatrix{Float64})
    @assert size(X) == size(R) == (F.m, F.nbatch)
    ldiv!(X, F.solver, R)
    return X
end

end
