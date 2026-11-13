# Bordered drift kinetic operators of many cases, applied together.
#
# The cases of a batch share the grids and stencils; the field, the plasma and the
# radial electric field may differ. With `nz = 1` the state of case `b` is
# `F[α, θ, j, b]`, `j = i + nx (s - 1)` the (speed, species) index, and the unknowns of
# all cases are the columns of an `(N + 2 ns) × B` matrix `[vec(f); sources]`.
#
# Every term is one of: a sparse matrix on the `(α, θ)` plane common to all cases times
# the state as an `(na nt) × (n B)` matrix; a dense matrix common to all cases times a
# reshaped state; a small dense matrix per case in a batched product; or a broadcast
# with coefficients per case. None of them reads single entries, so the same code runs
# on host arrays and on device arrays.

"""
    BatchedDKE(ops, borders, device; speed)

The bordered operators of `ops` in the form applied by `apply_bordered!(Y, bop, X)`,
with their row scalings, on `device` (a `Val`). `speed` keeps the advection in speed
(default: when a case has a radial electric field).
"""
struct BatchedDKE{D<:NamedTuple,W<:NamedTuple}
    dims::NTuple{4,Int}   # nalpha, ntheta, nx, nspecies
    nbatch::Int
    nL::Int
    krook::Float64
    has_speed::Bool
    has_exchange::Bool
    data::D
    work::W
end

state_size(bop::BatchedDKE) = bop.dims
system_size(bop::BatchedDKE) = prod(bop.dims) + 2 * bop.dims[4]

function _stencil_matrix(st::RowStencil)
    m = length(st.ptr) - 1
    rows = [i for i in 1:m for _ in st.ptr[i]:(st.ptr[i+1]-1)]
    return sparse(rows, st.idx, st.val, m, m)
end

_same_stencil(a::RowStencil, b::RowStencil) = a.ptr == b.ptr && a.idx == b.idx && a.val == b.val

# the cases of a batch must share everything that sets the structure of the operator
function _check_batch(ops::AbstractVector{<:DKEOperator})
    isempty(ops) && throw(ArgumentError("no cases to solve"))
    a = first(ops)
    @assert a.field.nzeta == 1 "the tokamak solver needs nzeta = 1"
    for (k, b) in enumerate(ops)
        same = state_size(b) == state_size(a) && b.weights == a.weights &&
               b.collisions.nL == a.collisions.nL && isempty(b.collisions.exchange) == isempty(a.collisions.exchange) &&
               all(>=(0), b.weights) && b.pitchgrid.wxi == a.pitchgrid.wxi &&
               b.collisions.D == a.collisions.D && b.collisions.Txi == a.collisions.Txi &&
               b.speedgrid.x == a.speedgrid.x && b.trajectories.Dx == a.trajectories.Dx &&
               all(_same_stencil(b.trajectories.Dt[d], a.trajectories.Dt[d]) for d in 1:2) &&
               all(_same_stencil(b.trajectories.Da[d], a.trajectories.Da[d]) for d in 1:2)
        same || throw(ArgumentError("case $k differs from case 1 in its grids, stencils, operator weights, " *
                                    "Legendre modes or energy exchange option; the cases of a batch must share these"))
    end
    return nothing
end

function BatchedDKE(ops::AbstractVector{<:DKEOperator{Float64}}, borders::AbstractVector{<:DKEBorder}, device::Val;
    speed::Bool=first(ops).weights[1] != 0 && any(op.trajectories.has_speed for op in ops))
    _check_batch(ops)
    op1 = first(ops)
    na, nt, _, nx, ns = state_size(op1)
    B = length(ops)
    np, n = na * nt, nx * ns
    N, nB = np * n, n * B
    nL = op1.collisions.nL
    w = op1.weights
    has_speed = speed
    has_exchange = !isempty(op1.collisions.exchange)

    wa, wt = zeros(np, nB), zeros(np, nB)
    wx = has_speed ? zeros(np, nB) : zeros(0, 0)
    nuscale, shift = zeros(1, nB), zeros(1, nB)
    KlocalT, KGHT = zeros(n, n, B), zeros(n, n, nL * B)
    Kexchange = has_exchange ? zeros(n, n, B) : zeros(0, 0, 0)
    S1, S2 = zeros(nx, ns, B), zeros(nx, ns, B)
    constraints = zeros(np * nx, 2, B)
    sw = zeros(1, nt, 1, B)
    scaling = zeros(N + 2ns, B)
    Threads.@threads for b in 1:B
        op, border = ops[b], borders[b]
        tr, col = op.trajectories, op.collisions
        cols = (1:n) .+ n * (b - 1)
        wa[:, cols] .= w[2] .* reshape(tr.wa, np, n)
        wt[:, cols] .= w[3] .* reshape(tr.wt, np, n)
        has_speed && (wx[:, cols] .= w[1] .* reshape(tr.wx, np, n))
        nuscale[1, cols] .= (-col.weights[1] / 2) .* vec(col.nus)
        shift[1, cols] .= diag(col.Klocal) .+ w[8]
        KlocalT[:, :, b] .= transpose(col.Klocal)
        for l in 1:nL
            KGHT[:, :, l+nL*(b-1)] .= transpose(view(col.KGH, :, :, l))
        end
        has_exchange && (Kexchange[:, :, b] .= col.Kexchange)
        S1[:, :, b] .= view(border.sources, :, :, 1)
        S2[:, :, b] .= view(border.sources, :, :, 2)
        constraints[:, :, b] .= reshape(border.constraints, np * nx, 2)
        sw[1, :, 1, b] .= vec(op.surface_weights)
        scaling[:, b] .= _row_scaling(op, border)
    end

    # (names distinct from those of the threaded loop above, which must stay local to it)
    tr1, col1 = op1.trajectories, op1.collisions
    It, Ia = sparse(1.0I, nt, nt), sparse(1.0I, na, na)
    dev(A) = _to_device(device, A)
    DxT = has_speed ? repeat(Matrix(transpose(kron(Matrix(1.0I, ns, ns), tr1.Dx))), 1, 1, B) : zeros(0, 0, 0)
    data = (
        # common to all cases: derivative along α and θ with either stencil, pitch
        # angle scattering, Legendre transform
        Da=(dev(kron(It, _stencil_matrix(tr1.Da[1]))), dev(kron(It, _stencil_matrix(tr1.Da[2])))),
        Dt=(dev(kron(_stencil_matrix(tr1.Dt[1]), Ia)), dev(kron(_stencil_matrix(tr1.Dt[2]), Ia))),
        Dpitch=dev(kron(It, sparse(col1.D))),
        Txi=dev(col1.Txi), Txi_inv=dev(col1.Txi_inv), txi1=dev(reshape(col1.Txi[:, 1], na, 1, 1, 1)),
        # per case
        wa=dev(wa), wt=dev(wt), wx=dev(wx), nuscale=dev(nuscale), shift=dev(shift), DxT=dev(DxT),
        KlocalT=dev(KlocalT), KGHT=dev(KGHT), Kexchange=dev(Kexchange),
        S1=dev(S1), S2=dev(S2), constraints=dev(constraints), sw=dev(sw), scaling=dev(scaling))

    like(dims...) = similar(data.wa, dims...)
    work = (F=like(np, nB), O=like(np, nB), T=like(np, nB), T2=like(np, nB), mean=like(1, nt * nB),
        Fm=like(nL, nt, n, B), Gm=like(nL, nt, n, B), P=like(nt, n, nL, B), Q=like(nt, n, nL, B),
        c=like(1, 1, n, B), d=like(n, 1, B), src=like(2ns, B), coef=like(nx, ns, B), con=like(2ns, B))
    return BatchedDKE((na, nt, nx, ns), B, nL, w[8], has_speed, has_exchange, data, work)
end

"""
    apply_bordered!(Y, bop::BatchedDKE, X)

The bordered system of every case acting on its column of `X`, see the method for a
single [`DKEOperator`](@ref). `X` and `Y` must not share memory.
"""
function apply_bordered!(Y::AbstractMatrix, bop::BatchedDKE, X::AbstractMatrix)
    na, nt, nx, ns = bop.dims
    B, nL = bop.nbatch, bop.nL
    np, n = na * nt, nx * ns
    N, nB = np * n, n * B
    d, w = bop.data, bop.work
    F, O, T = w.F, w.O, w.T
    reshape(F, N, B) .= view(X, 1:N, :)
    w.src .= view(X, (N+1):(N+2ns), :)
    F3, O3 = reshape(F, np, n, B), reshape(O, np, n, B)
    fill!(O, 0)

    # advection in α and θ, upwinded node by node
    for (D, W) in ((d.Da, d.wa), (d.Dt, d.wt))
        mul!(T, D[2], F)
        O .+= max.(W, 0) .* T
        mul!(T, D[1], F)
        O .+= min.(W, 0) .* T
    end
    # advection in speed
    if bop.has_speed
        _batched_gemm!('N', 'N', 1.0, F3, d.DxT, 0.0, reshape(T, np, n, B))
        O .+= d.wx .* T
    end

    # collisions local in pitch angle: energy scattering and local field particle part
    _batched_gemm!('N', 'N', 1.0, F3, d.KlocalT, 1.0, O3)

    # pitch angle scattering, of the part without the pitch average
    G = reshape(F, na, nt * nB)
    sum!(w.mean, G)
    reshape(T, na, nt * nB) .= G .- w.mean ./ na
    mul!(w.T2, d.Dpitch, T)
    O .+= d.nuscale .* w.T2

    # Rosenbluth potential part, diagonal in Legendre mode
    mul!(reshape(w.Fm, nL, nt * nB), d.Txi_inv, G)
    permutedims!(w.P, w.Fm, (2, 3, 1, 4))
    _batched_gemm!('N', 'N', 1.0, reshape(w.P, nt, n, nL * B), d.KGHT, 0.0, reshape(w.Q, nt, n, nL * B))
    permutedims!(w.Gm, w.Q, (3, 1, 2, 4))
    mul!(reshape(O, na, nt * nB), d.Txi, reshape(w.Gm, nL, nt * nB), true, true)

    # surface averaged exchange between species
    if bop.has_exchange
        sum!(w.c, d.sw .* view(w.Fm, 1:1, :, :, :))
        _batched_gemm!('N', 'N', 1.0, d.Kexchange, reshape(w.c, n, 1, B), 0.0, w.d)
        reshape(O, na, nt, n, B) .+= d.txi1 .* reshape(w.d, 1, 1, n, B)
    end
    bop.krook != 0 && (O .+= bop.krook .* F)

    # border: sources (columns) and constraints (rows)
    src = reshape(w.src, 2, ns, B)
    w.coef .= d.S1 .* view(src, 1:1, :, :) .+ d.S2 .* view(src, 2:2, :, :)
    O .+= reshape(w.coef, 1, nB)
    _batched_gemm!('T', 'N', 1.0, d.constraints, reshape(F, np * nx, ns, B), 0.0, reshape(w.con, 2, ns, B))
    view(Y, 1:N, :) .= reshape(O, N, B)
    view(Y, (N+1):(N+2ns), :) .= w.con
    return Y
end
