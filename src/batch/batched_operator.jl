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
    BatchedDKE(ops, device; speed)

The bordered operators of `ops` in the form applied by `apply_bordered!(Y, bop, X)`,
with their row scalings, on `device` (a `Val`). `speed` keeps the advection in speed
(default: when a case has a radial electric field).

The constraints and the row scalings are products of factors in pitch angle, θ and
speed; they are formed on the device from those factors, so that the host only fills
arrays of the size of the operator coefficients.
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
_free!(bop::BatchedDKE) = (_free!(bop.data); _free!(bop.work))
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

function BatchedDKE(ops::AbstractVector{<:DKEOperator{Float64}}, device::Val;
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
    sw = zeros(1, nt, 1, B)
    # factors of the constraints and the row scalings
    xg, wxg, wxig = op1.speedgrid.x, op1.speedgrid.wx, op1.pitchgrid.wxi   # (`wx` is taken: speed advection)
    dr, root_sw, speed_scale, border_scale = zeros(1, nt, 1, 1, B), zeros(1, nt, 1, 1, B), zeros(1, 1, nx, ns, B), zeros(2ns, B)
    moments = [sum(xg[i]^2 * wxg[i] * xg[i]^(2(k - 1)) for i in 1:nx) for k in 1:2]
    Threads.@threads for b in 1:B
        op = ops[b]
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
        drb = vec(volume_element(op.field))
        dr[1, :, 1, 1, b] .= drb
        sw[1, :, 1, b] .= vec(op.surface_weights)
        root_sw[1, :, 1, 1, b] .= sqrt.(vec(op.surface_weights))
        for (s, sp) in enumerate(op.species)
            for i in 1:nx
                F = sp(xg[i] * sp.v_thermal)
                S1[i, s, b], S2[i, s, b] = (xg[i]^2 - 5 / 2) * F, (-2 / 3 * xg[i]^2 + 1) * F   # as in DKEBorder
                speed_scale[1, 1, i, s, b] = sqrt(exp(-xg[i]^2) * xg[i]^2 * wxg[i]) / F         # as in _row_scaling
            end
            for k in 1:2
                border_scale[2(s-1)+k, b] = 1 / (4π^2 * sum(wxig) * sum(drb) * moments[k] * sp(0.0))
            end
        end
    end

    # (names distinct from those of the threaded loop above, which must stay local to it)
    tr1, col1 = op1.trajectories, op1.collisions
    It, Ia = sparse(1.0I, nt, nt), sparse(1.0I, na, na)
    dev(A) = _to_device(device, A)
    DxT = has_speed ? repeat(Matrix(transpose(kron(Matrix(1.0I, ns, ns), tr1.Dx))), 1, 1, B) : zeros(0, 0, 0)
    # constraints[α, θ, x, k, b] = 4π² wxig[α] dr[θ] x² wx x^(2(k-1)) (DKEBorder), and the
    # row scalings of _row_scaling, from their factors on the device
    pitch = dev(reshape(4π^2 .* wxig, na, 1, 1, 1, 1))
    speedk = dev(reshape([xg[i]^2 * wxg[i] * xg[i]^(2(k - 1)) for i in 1:nx, k in 1:2], 1, 1, nx, 2, 1))
    constraints = reshape(pitch .* dev(dr) .* speedk, np * nx, 2, B)
    scaling = similar(constraints, N + 2ns, B)
    view(scaling, 1:N, :) .= reshape(dev(reshape(sqrt.(wxig), na, 1, 1, 1, 1)) .* dev(root_sw) .* dev(speed_scale), N, B)
    view(scaling, (N+1):(N+2ns), :) .= dev(border_scale)
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
        S1=dev(S1), S2=dev(S2), constraints, sw=dev(sw), scaling)

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

"""
    _batched_rhs(ops, EparB, bop)

Right hand sides of the bordered systems of `ops` ([`dke_rhs`](@ref) and zero for the
constraints), one column per case, on the device of `bop`. The drive is a sum of two
products of factors in pitch angle, θ and speed, formed on the device.
"""
function _batched_rhs(ops::AbstractVector{<:DKEOperator{Float64}}, EparB::AbstractVector{<:Real}, bop::BatchedDKE)
    na, nt, nx, ns = bop.dims
    B, N = bop.nbatch, prod(bop.dims)
    op1 = first(ops)
    xi, x = op1.pitchgrid.xi, op1.speedgrid.x
    # drive = c1[x, s] (1 + ξ²) BxgradB/(2B³)[θ] + c2[x, s] ξ B[θ], see dke_rhs and
    # radial_magnetic_drift: -(A1 + x² A2) v_m·∇ρ F_M + A3 B v∥ F_M
    c1, c2 = zeros(1, 1, nx, ns, B), zeros(1, 1, nx, ns, B)
    h1, h2 = zeros(1, nt, 1, 1, B), zeros(1, nt, 1, 1, B)
    Threads.@threads for b in 1:B
        op = ops[b]
        field = op.field
        h1[1, :, 1, 1, b] .= vec(field.BxgradrhodotgradB) ./ (2 .* vec(field.Bmag) .^ 3)
        h2[1, :, 1, 1, b] .= vec(field.Bmag)
        for (s, (sp, (A1, A2, A3))) in enumerate(zip(op.species, thermodynamic_forces(op, EparB[b]))), i in 1:nx
            v = x[i] * sp.v_thermal
            F = sp(v)
            c1[1, 1, i, s, b] = (A1 + x[i]^2 * A2) * sp.species.mass * v^2 / sp.species.charge * F
            c2[1, 1, i, s, b] = A3 * v * F
        end
    end
    like(A) = _like(bop.data.scaling, A)
    R = fill!(similar(bop.data.scaling, N + 2ns, B), 0)
    view(R, 1:N, :) .= reshape(like(reshape(1 .+ xi .^ 2, na, 1, 1, 1, 1)) .* like(h1) .* like(c1) .+
                               like(reshape(collect(xi), na, 1, 1, 1, 1)) .* like(h2) .* like(c2), N, B)
    return R
end
