# Direct solver for the moment coarse matrices of a batch, by dense blocks; also used
# for the plane matrices, see the end of the file.
#
# The coarse unknowns are ordered by poloidal point, `nl n` of them at each, followed by
# the surface averages and the sources. Poloidal points are coupled through the
# advection stencil only, which reaches `h` points either way, around the torus. Taking
# `h` consecutive points as one block of size `sb = h nl n`, and moving the last points
# (at least `h`, so that the rest is a whole number `K` of blocks) to the end with the
# averages and sources, gives
#
#     [ T  U ] [x]   [f]        T: block tridiagonal, K blocks of size sb
#     [ V  S ] [y] = [g]        S: border of size nbd
#
# which is solved by block elimination along T and the Schur complement of the border:
# `K` inverses of `sb × sb` matrices and products with them, each done for all cases of
# the batch at once. Pivoting is within the blocks.
#
# A sparse LU does the same work with less memory traffic on a host; on a GPU the
# batched dense kernels are several times faster than a batched sparse factorization,
# and more accurate than one without pivoting.

# `npoints` grid points of `mb` unknowns each, coupled `reach` points either way (around
# the end when `periodic`), and `extra` unknowns at the end: chain blocks of `reach`
# points and a border with the rest. Without periodicity the border may be empty.
function _chain_layout(reach::Integer, npoints::Integer, mb::Integer, extra::Integer; periodic::Bool)
    reach = max(reach, 1)
    tail = periodic ? reach + npoints % reach : npoints % reach   # points in the border
    K = (npoints - tail) ÷ reach
    K >= 1 || return nothing
    return (sb=reach * mb, K=K, nbd=tail * mb + extra)
end

# largest distance between coupled points of the stencils `sts` on `n` points, around
# the end when `periodic`
function _stencil_reach(sts, n::Integer; periodic::Bool)
    reach = 0
    for st in sts, i in 1:n, k in st.ptr[i]:(st.ptr[i+1]-1)
        distance = abs(st.idx[k] - i)
        reach = max(reach, periodic ? min(distance, n - distance) : distance)
    end
    return reach
end

"Block sizes of the coarse matrices, or `nothing` if the poloidal grid is too small."
function _coarse_block_layout(op::DKEOperator, nl::Integer)
    _, nt, _, nx, ns = state_size(op)
    reach = _stencil_reach(op.trajectories.Dt, nt; periodic=true)
    return _chain_layout(reach, nt, nl * nx * ns, nx * ns + 2ns; periodic=true)
end

"""
Maps from the ingredients of a case to the dense blocks of its coarse matrix: for each
part (`D`, `L`, `U` of the chain, the border columns `Ub` and rows `Vb`, the border
`S`), `G` gives the stored entries from the ingredients, which go to position `inblock`
of block `block` (from 0) of that part.
"""
struct BlockAssemblyMap{P<:NamedTuple}
    sb::Int
    K::Int
    nbd::Int
    ningredients::Int
    parts::P   # name => (G, inblock, block)
end

function BlockAssemblyMap(e::EntryList, m::Integer, ningredients::Integer, layout::NamedTuple, device::Val)
    sb, K, nbd = layout.sb, layout.K, layout.nbd
    nch = K * sb
    @assert nch + nbd == m
    names = (:D, :L, :U, :Ub, :Vb, :S)
    blocksize = (D=sb * sb, L=sb * sb, U=sb * sb, Ub=sb * nbd, Vb=nbd * sb, S=nbd * nbd)
    slots = NamedTuple{names}(ntuple(_ -> Int[], 6))
    entries = NamedTuple{names}(ntuple(_ -> Int[], 6))
    for (q, (r, c)) in enumerate(zip(e.rows, e.cols))
        if r <= nch && c <= nch
            kr, ir, kc, ic = (r - 1) ÷ sb + 1, (r - 1) % sb + 1, (c - 1) ÷ sb + 1, (c - 1) % sb + 1
            name = kc == kr ? :D : kc == kr - 1 ? :L : kc == kr + 1 ? :U :
                   error("coarse matrix: an entry couples blocks $kr and $kc, which are not neighbours")
            slot = ir + sb * (ic - 1) + sb * sb * (kr - 1)
        elseif r <= nch
            name, slot = :Ub, (r - 1) % sb + 1 + sb * (c - nch - 1) + sb * nbd * ((r - 1) ÷ sb)
        elseif c <= nch
            name, slot = :Vb, (r - nch) + nbd * ((c - 1) % sb) + nbd * sb * ((c - 1) ÷ sb)
        else
            name, slot = :S, (r - nch) + nbd * (c - nch - 1)
        end
        push!(slots[name], slot), push!(entries[name], q)
    end
    parts = map(names) do name
        stored = sort!(unique(slots[name]))
        rows = [searchsortedfirst(stored, x) for x in slots[name]]
        G = sparse(rows, e.ingredients[entries[name]], e.coefficients[entries[name]], length(stored), ningredients)
        n = blocksize[name]
        (G=_to_device(device, G), inblock=_to_device(device, (stored .- 1) .% n .+ 1), block=_to_device(device, (stored .- 1) .÷ n))
    end
    return BlockAssemblyMap(sb, K, nbd, Int(ningredients), NamedTuple{names}(parts))
end

# `nblocks` blocks of size `rows × cols` of all cases, [:, :, case, block], zero but for
# the stored entries of `part`
function _scatter(part::NamedTuple, Θ::AbstractMatrix, rows::Integer, cols::Integer, nblocks::Integer)
    B = size(Θ, 2)
    out = fill!(similar(Θ, rows, cols, B, nblocks), 0)
    size(part.G, 1) == 0 && return out
    cases = _like(part.inblock, reshape(collect(0:B-1), 1, B))
    positions = part.inblock .+ (rows * cols * B) .* part.block .+ (rows * cols) .* cases
    out[vec(positions)] = vec(part.G * Θ)
    return out
end

# the blocks of all cases, [:, :, case, block]
function _assemble_blocks(map::BlockAssemblyMap, Θ::AbstractMatrix)
    sb, K, nbd, B = map.sb, map.K, map.nbd, size(Θ, 2)
    p = map.parts
    return (sb=sb, K=K, nbd=nbd, D=_scatter(p.D, Θ, sb, sb, K), L=_scatter(p.L, Θ, sb, sb, K), U=_scatter(p.U, Θ, sb, sb, K),
        Ub=_scatter(p.Ub, Θ, sb, nbd, K), Vb=_scatter(p.Vb, Θ, nbd, sb, K), S=reshape(_scatter(p.S, Θ, nbd, nbd, 1), nbd, nbd, B))
end

"""
    _batched_inverse!(C, A)

`C[:, :, k] = inv(A[:, :, k])` for every `k`; `A` may be overwritten.
"""
function _batched_inverse!(C::AbstractArray{T,3}, A::AbstractArray{T,3}) where {T}
    for k in axes(A, 3)
        C[:, :, k] .= inv(A[:, :, k])
    end
    return C
end

struct BlockCoarseFactors{A4,A3,W<:NamedTuple}
    sb::Int
    K::Int
    nbd::Int
    Dinv::A4   # inverse of the pivot blocks of T, [:, :, case, block]
    Lhat::A4   # lower blocks times the inverse of the pivot above
    U::A4      # upper blocks
    Y::A4      # T⁻¹ U of the border columns
    V::A4      # border rows
    Sinv::A3   # inverse of the Schur complement of the border
    work::W
end

_slice(A::AbstractArray{<:Any,4}, k::Integer) = view(A, :, :, :, k)

"Factorization of the coarse matrices given by their blocks, which are overwritten."
function _factor_blocks(b::NamedTuple)
    sb, K, nbd, B = b.sb, b.K, b.nbd, size(b.S, 3)
    D, L, U, Y, V, S = b.D, b.L, b.U, b.Ub, b.Vb, b.S
    # the inverses of the pivots replace D, and L D⁻¹ replaces L, block by block
    Dinv, Lhat = D, L
    tmp = similar(D, sb, sb, B)
    _batched_inverse!(tmp, _slice(D, 1))
    _slice(Dinv, 1) .= tmp
    for k in 2:K
        _batched_gemm!('N', 'N', 1.0, _slice(L, k), _slice(Dinv, k - 1), 0.0, tmp)
        _slice(Lhat, k) .= tmp
        _batched_gemm!('N', 'N', -1.0, _slice(Lhat, k), _slice(U, k - 1), 1.0, _slice(D, k))
        _batched_inverse!(tmp, _slice(D, k))
        _slice(Dinv, k) .= tmp
    end
    Sinv = similar(S)
    if nbd > 0
        # Y = T⁻¹ U, in place
        tmp = similar(Y, sb, nbd, B)
        for k in 2:K
            _batched_gemm!('N', 'N', -1.0, _slice(Lhat, k), _slice(Y, k - 1), 1.0, _slice(Y, k))
        end
        for k in K:-1:1
            k < K && _batched_gemm!('N', 'N', -1.0, _slice(U, k), _slice(Y, k + 1), 1.0, _slice(Y, k))
            _batched_gemm!('N', 'N', 1.0, _slice(Dinv, k), _slice(Y, k), 0.0, tmp)
            _slice(Y, k) .= tmp
        end
        for k in 1:K
            _batched_gemm!('N', 'N', -1.0, _slice(V, k), _slice(Y, k), 1.0, S)
        end
        _batched_inverse!(Sinv, S)
    end
    work = (f=similar(S, sb, 1, B, K), x=similar(S, sb, 1, B, K), g=similar(S, nbd, 1, B), y=similar(S, nbd, 1, B),
        chain=similar(S, sb * K, B))
    return BlockCoarseFactors(sb, K, nbd, Dinv, Lhat, U, Y, V, Sinv, work)
end

function _solve_batch!(X::AbstractMatrix, F::BlockCoarseFactors, R::AbstractMatrix)
    sb, K, nbd, B = F.sb, F.K, F.nbd, size(R, 2)
    nch = sb * K
    f, x, g, y = F.work.f, F.work.x, F.work.g, F.work.y
    F.work.chain .= view(R, 1:nch, :)
    permutedims!(reshape(f, sb, B, K), reshape(F.work.chain, sb, K, B), (1, 3, 2))
    reshape(g, nbd, B) .= view(R, (nch+1):(nch+nbd), :)
    # x = T⁻¹ f
    for k in 2:K
        _batched_gemm!('N', 'N', -1.0, _slice(F.Lhat, k), _slice(f, k - 1), 1.0, _slice(f, k))
    end
    for k in K:-1:1
        k < K && _batched_gemm!('N', 'N', -1.0, _slice(F.U, k), _slice(x, k + 1), 1.0, _slice(f, k))
        _batched_gemm!('N', 'N', 1.0, _slice(F.Dinv, k), _slice(f, k), 0.0, _slice(x, k))
    end
    # border, then the correction of x
    if nbd > 0
        for k in 1:K
            _batched_gemm!('N', 'N', -1.0, _slice(F.V, k), _slice(x, k), 1.0, g)
        end
        _batched_gemm!('N', 'N', 1.0, F.Sinv, g, 0.0, y)
        for k in 1:K
            _batched_gemm!('N', 'N', -1.0, _slice(F.Y, k), y, 1.0, _slice(x, k))
        end
    end
    permutedims!(reshape(F.work.chain, sb, K, B), reshape(x, sb, B, K), (1, 3, 2))
    view(X, 1:nch, :) .= F.work.chain
    view(X, (nch+1):(nch+nbd), :) .= reshape(y, nbd, B)
    return X
end

# ------------------------------------------------------------------------------------
# Plane matrices by dense blocks. On a plane the unknowns are coupled along θ around the
# torus and along the pitch angle α, which does not wrap, through the advection stencil
# and the pitch angle scattering. Numbered with θ fastest, a block of `h` consecutive
# pitch angles (`h` the reach in α) with all their poloidal points couples to its two
# neighbours only, and the plane matrix is the chain of the coarse matrices with
# `sb = h nt`; the pitch angles left over when `h` does not divide `nalpha` form the
# border. The work is that of a banded LU of bandwidth `sb`, in batched dense kernels.

"Block sizes of the plane matrices, numbered with θ fastest, or `nothing` if α is too short."
function _plane_block_layout(op::DKEOperator)
    na, nt = state_size(op)[1:2]
    tr = op.trajectories
    D = op.collisions.D
    reach = max(_stencil_reach(tr.Da, na; periodic=false),
        maximum((abs(a - a2) for a in 1:na, a2 in 1:na if D[a, a2] != 0); init=0))
    return _chain_layout(reach, na, nt, 0; periodic=false)
end

"Maps to the blocks of the plane matrices, which are numbered with θ fastest."
struct PlaneBlockMap{M<:BlockAssemblyMap}
    na::Int
    nt::Int
    blocks::M
end

function PlaneBlockMap(e::EntryList, na::Integer, nt::Integer, ningredients::Integer, layout::NamedTuple, device::Val)
    θfirst(r) = (a = (r - 1) % na + 1; t = (r - 1) ÷ na + 1; t + nt * (a - 1))
    renumbered = EntryList(θfirst.(e.rows), θfirst.(e.cols), e.ingredients, e.coefficients)
    return PlaneBlockMap(Int(na), Int(nt), BlockAssemblyMap(renumbered, na * nt, ningredients, layout, device))
end

_assemble(map::PlaneBlockMap, Θ::AbstractMatrix) = (na=map.na, nt=map.nt, blocks=_assemble_blocks(map.blocks, Θ))

struct PlaneBlockFactors{F<:BlockCoarseFactors,A<:AbstractMatrix}
    na::Int
    nt::Int
    factors::F
    R::A   # right hand sides and solutions, θ fastest
    X::A
end

function _factor_planes(::Val, planes::NamedTuple)
    factors = _factor_blocks(planes.blocks)
    nplanes = size(planes.blocks.D, 3)
    R = similar(planes.blocks.D, planes.na * planes.nt, nplanes)
    return PlaneBlockFactors(planes.na, planes.nt, factors, R, similar(R))
end
_factor_planes(device::Val, planes::UniformSparseBatch) = _factor_batch(device, planes)

function _solve_batch!(X::AbstractMatrix, F::PlaneBlockFactors, R::AbstractMatrix)
    na, nt, nplanes = F.na, F.nt, size(R, 2)
    permutedims!(reshape(F.R, nt, na, nplanes), reshape(R, na, nt, nplanes), (2, 1, 3))
    _solve_batch!(F.X, F.factors, F.R)
    permutedims!(reshape(X, na, nt, nplanes), reshape(F.X, nt, na, nplanes), (2, 1, 3))
    return X
end
