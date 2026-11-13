# Device seams of the batched solver.
#
# The batched operator, Krylov iteration and preconditioner are written for any array
# type that supports broadcasting, `mul!`, reductions over a dimension and
# `permutedims!`. The few operations that have no such generic form go through the
# functions below, which have host methods here and device methods in a package
# extension (`ext/YanccCUDAExt.jl`). Extensions only add methods for their own array
# types and `Val{:device}` tags; none of the methods defined here is overwritten.

# filled by an extension when its device can be used
const _DEVICE_FUNCTIONAL = Dict{Symbol,Any}()

"`Val` tag of the device named by `device` (`:cpu`, `:cuda` or `:auto`)."
function _pick_device(device::Symbol)
    device == :cpu && return Val(:cpu)
    if device == :auto
        for (name, functional) in _DEVICE_FUNCTIONAL
            functional() && return Val(name)
        end
        return Val(:cpu)
    end
    haskey(_DEVICE_FUNCTIONAL, device) ||
        error("device = :$device is not available: for :cuda run `using CUDA, CUDSS` before the solve")
    _DEVICE_FUNCTIONAL[device]() || error("device = :$device was loaded but is not functional on this machine")
    return Val(device)
end

"Copy of the host array or sparse matrix `A` on the device."
_to_device(::Val{:cpu}, A) = A

"Solver of the coarse matrices that suits the device: `:sparse` LU or dense `:block`s."
_default_coarse_solver(::Val{:cpu}) = :sparse

"Solver of the plane matrices of `op` that suits the device: `:sparse` LU or dense `:block`s."
_default_plane_solver(::Val{:cpu}, op) = :sparse

"Wait for the device to finish what it was given (for timings)."
_synchronize(::Val) = nothing

"Return the memory of arrays that are no longer referenced to the device."
_release(::Val) = nothing

"""
Number of cases to solve at once when the caller does not say; `planes` is the block
layout of the plane matrices, or `nothing` when they are factored sparse.
"""
_default_batchsize(::Val{:cpu}, ncases::Integer, unknowns::Integer, restart::Integer, planes) = ncases

"""
    _batched_gemm!(tA, tB, α, A, B, β, C)

`C[:, :, k] = α op(A[:, :, k]) op(B[:, :, k]) + β C[:, :, k]` for every `k`, with `op`
the identity (`'N'`) or the transpose (`'T'`).
"""
function _batched_gemm!(tA::Char, tB::Char, α::Real, A::AbstractArray{T,3}, B::AbstractArray{T,3}, β::Real,
    C::AbstractArray{T,3}) where {T}
    @assert size(A, 3) == size(B, 3) == size(C, 3)
    if tA == 'N' && tB == 'N'
        @views for k in axes(C, 3)
            mul!(C[:, :, k], A[:, :, k], B[:, :, k], α, β)
        end
    elseif tA == 'T' && tB == 'N'
        @views for k in axes(C, 3)
            mul!(C[:, :, k], transpose(A[:, :, k]), B[:, :, k], α, β)
        end
    else
        error("_batched_gemm!: unsupported combination $tA, $tB")
    end
    return C
end

"""
Sparse `m × m` matrices of one sparsity pattern, stored by rows, the values of matrix
`k` in `nzval[:, k]` (a host or a device array).
"""
struct UniformSparseBatch{V<:AbstractMatrix}
    m::Int
    rowptr::Vector{Int}
    colval::Vector{Int}
    nzval::V
end

Base.length(A::UniformSparseBatch) = size(A.nzval, 2)
SparseArrays.nnz(A::UniformSparseBatch) = length(A.colval)   # of each matrix
# matrix k on the host; its transpose is what compressed columns store
_matrix(A::UniformSparseBatch, k::Integer) = _matrix(A.m, A.rowptr, A.colval, Array(view(A.nzval, :, k)))
_matrix(m, rowptr, colval, values::Vector) = copy(transpose(SparseMatrixCSC(m, m, rowptr, colval, values)))

# sparse LU of every matrix of a batch, on the host
struct HostLUBatch{F}
    factors::Vector{F}
end

function _foreach_chunk(f, n::Integer)
    nchunks = min(n, Threads.nthreads())
    nchunks <= 1 && return f(1:n)
    @sync for chunk in Iterators.partition(1:n, cld(n, nchunks))
        Threads.@spawn f(chunk)
    end
    return nothing
end

"""
    _factor_batch(device, A::UniformSparseBatch)

LU factorization of every matrix of the batch, for [`_solve_batch!`](@ref).
"""
function _factor_batch(::Val{:cpu}, A::UniformSparseBatch)
    values = Array(A.nzval)
    first_factor = lu(_matrix(A.m, A.rowptr, A.colval, values[:, 1]))
    factors = Vector{typeof(first_factor)}(undef, length(A))
    factors[1] = first_factor
    _foreach_chunk(length(A) - 1) do chunk
        for k in chunk
            factors[k+1] = lu(_matrix(A.m, A.rowptr, A.colval, values[:, k+1]))
        end
    end
    return HostLUBatch(factors)
end

"`X[:, k] = A_k⁻¹ R[:, k]` for every matrix of the factored batch."
function _solve_batch!(X::AbstractMatrix, F::HostLUBatch, R::AbstractMatrix)
    @assert size(X, 2) == size(R, 2) == length(F.factors)
    _foreach_chunk(length(F.factors)) do chunk
        buffer = Vector{Float64}(undef, size(R, 1))
        for k in chunk
            ldiv!(buffer, F.factors[k], view(R, :, k))
            copyto!(view(X, :, k), buffer)
        end
    end
    return X
end

# host vector or array `v` as an array like `A` (on its device)
_like(A::AbstractArray, v::Array) = copyto!(similar(A, eltype(v), size(v)), v)

# 2-norm of every column, on the host
_column_norms(A::AbstractMatrix) = vec(Array(sqrt.(sum(abs2, A; dims=1))))
