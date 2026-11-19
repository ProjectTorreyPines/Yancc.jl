# Assembly of the preconditioner matrices of all cases without a loop over their entries.
#
# Every entry of a plane matrix or of a moment coarse matrix is a fixed combination of a
# few quantities of its case (advection speeds at a node, collision matrices, sources,
# ...), the "ingredients". The combination depends on the grids and stencils only, so
# for a batch it is one sparse matrix `G` with a row per stored entry and a column per
# ingredient:
#
#     values[:, case] = G * ingredients[:, case]
#
# `G` and the sparsity pattern are built once on the host by the loops below, which
# mirror `plane_matrix` and `moment_coarse_matrix` (and are tested against them). The
# ingredients come from the arrays of the batched operator by reshapes, broadcasts and
# dense products, and the product with `G` gives the values of all cases at once, on
# whichever device holds the operator.

# contributions (row, column, ingredient, coefficient) to the entries of a matrix
struct EntryList
    rows::Vector{Int}
    cols::Vector{Int}
    ingredients::Vector{Int}
    coefficients::Vector{Float64}
end
EntryList() = EntryList(Int[], Int[], Int[], Float64[])

function _add!(e::EntryList, r::Integer, c::Integer, ingredient::Integer, coefficient::Real)
    push!(e.rows, r), push!(e.cols, c), push!(e.ingredients, ingredient), push!(e.coefficients, coefficient)
    return e
end

"""
Sparsity pattern by rows of `m × m` matrices and the map `G` from the ingredients of a
case to its stored values, see the top of this file.
"""
struct AssemblyMap{S}
    m::Int
    rowptr::Vector{Int}
    colval::Vector{Int}
    ningredients::Int
    G::S
end

function AssemblyMap(e::EntryList, m::Integer, ningredients::Integer, device::Val)
    keys = (e.rows .- 1) .* m .+ e.cols          # ordered by row, then column
    stored = sort!(unique(keys))
    slots = [searchsortedfirst(stored, k) for k in keys]
    rowptr = ones(Int, m + 1)
    for k in stored
        rowptr[(k-1)÷m+2] += 1
    end
    cumsum!(rowptr, rowptr)
    rowptr .-= (0:m)
    colval = [(k - 1) % m + 1 for k in stored]
    G = sparse(slots, e.ingredients, e.coefficients, length(stored), ningredients)
    return AssemblyMap(Int(m), rowptr, colval, Int(ningredients), _to_device(device, G))
end

SparseArrays.nnz(A::AssemblyMap) = length(A.colval)

# ------------------------------------------------------------------------------------
# plane matrices: ingredients of the plane of speed and species j of a case are
#   max(wt, 0), min(wt, 0), max(wa, 0), min(wa, 0) at each node, scatter, shift
# with wt, wa the advection speeds times their operator weights, scatter = -w_L ν_D / 2
# and shift = Klocal[j, j] + Krook weight; in a rotating plasma scatter and shift at each
# poloidal point.

function _plane_entries(op::DKEOperator)
    na, nt, _, _, _ = state_size(op)
    tr, col = op.trajectories, op.collisions
    np = na * nt
    rotating = col.rotating !== nothing
    scatter(t) = 4np + (rotating ? t : 1)
    shift(t) = 4np + (rotating ? nt + t : 2)
    idx(a, t) = a + na * (t - 1)
    e = EntryList()
    for t in 1:nt, a in 1:na
        r = idx(a, t)
        for d in 1:2   # backward stencil where the speed is positive
            st = tr.Dt[d]
            for k in st.ptr[t]:(st.ptr[t+1]-1)
                _add!(e, r, idx(a, st.idx[k]), (d == 2 ? 0 : np) + r, st.val[k])
            end
            st = tr.Da[d]
            for k in st.ptr[a]:(st.ptr[a+1]-1)
                _add!(e, r, idx(st.idx[k], t), (d == 2 ? 2np : 3np) + r, st.val[k])
            end
        end
        for a2 in 1:na
            col.D[a, a2] == 0 || _add!(e, r, idx(a2, t), scatter(t), col.D[a, a2])
        end
        _add!(e, r, r, shift(t), 1.0)
    end
    return e, np, 4np + (rotating ? 2nt : 2)
end

function _plane_ingredients(bop::BatchedDKE)
    na, nt, nx, ns = bop.dims
    np, nB = na * nt, nx * ns * bop.nbatch
    d = bop.data
    nc = size(d.nuscale, 1)   # 1, or the poloidal points in a rotating plasma
    Θ = similar(d.wa, 4np + 2nc, nB)
    view(Θ, 1:np, :) .= max.(d.wt, 0)
    view(Θ, (np+1):2np, :) .= min.(d.wt, 0)
    view(Θ, (2np+1):3np, :) .= max.(d.wa, 0)
    view(Θ, (3np+1):4np, :) .= min.(d.wa, 0)
    view(Θ, (4np+1):(4np+nc), :) .= d.nuscale
    view(Θ, (4np+nc+1):(4np+2nc), :) .= d.shift
    return Θ
end

# ------------------------------------------------------------------------------------
# moment coarse matrices, see `moment_coarse_matrix` for the unknowns. With
# q = (l', l) pairs of moments, the ingredients of a case are, in this order:
#   1                     the constant 1
#   TP, TM [q, θ, j]      W max(wt, 0) P and W min(wt, 0) P
#   MI [q, θ, j]          W (max(wa, 0) Da_bwd + min(wa, 0) Da_fwd) P
#   SP [q, θ, j]          W wx P (when a case has a radial electric field)
#   NU [j]                scatter
#   S1, S2 [j]            speed dependence of the two sources
#   CP [l, θ, i, k]       constraint weights projected on the moments
#   SW [θ]                surface average weights
#   KL [j, j']            Klocal[j', j]
#   KG [j, j', l]         KGH[j', j, l]
#   KX [j', j]            exchange correction (unless energy exchange is kept)
# In a rotating plasma NU, KL and KG are those of each poloidal point (θ the last index),
# and instead of KX, with the moments k of RotatingCollisions and r = θ + nt (j - 1):
#   XM, XK [k, r]         surface average weights times M and M K0
#   AG [r, k]             A G - K0 V
#   XA [r, k]             A

struct CoarseLayout
    nl::Int
    nt::Int
    n::Int
    nx::Int
    ns::Int
    speed::Bool
    exchange::Bool
    rotating::Bool
    nk::Int   # moments of the exchange correction of a rotating plasma
end

# unknowns of the coarse matrix after the moments and before the sources
_coarse_aux(c::CoarseLayout) = c.rotating ? 2 * c.nk : c.n
_coarse_aux(op::DKEOperator) = op.collisions.rotating === nothing ? prod(state_size(op)[4:5]) : 2 * size(op.collisions.rotating.V, 2)

function _offsets(c::CoarseLayout)
    q, n = c.nl^2, c.n
    per = c.rotating ? c.nt : 1
    sizes = (ONE=1, TP=q * c.nt * n, TM=q * c.nt * n, MI=q * c.nt * n, SP=c.speed ? q * c.nt * n : 0, NU=n * per, S1=n, S2=n,
        CP=c.nl * c.nt * c.nx * 2, SW=c.nt, KL=n^2 * per, KG=n^2 * c.nl * per, KX=c.exchange ? n^2 : 0,
        XM=c.nk * c.nt * n, XK=c.nk * c.nt * n, AG=c.nk * c.nt * n, XA=c.nk * c.nt * n)
    starts = cumsum((0, values(sizes)...))
    return NamedTuple{keys(sizes)}(starts[1:end-1]), starts[end], sizes
end

function _coarse_entries(op::DKEOperator, nl::Integer, speed::Bool)
    na, nt, _, nx, ns = state_size(op)
    tr, col, w = op.trajectories, op.collisions, op.weights
    n = nx * ns
    rot = col.rotating
    rotating = rot !== nothing
    nk = rotating ? size(rot.V, 2) : 0
    layout = CoarseLayout(nl, nt, n, nx, ns, speed && w[1] != 0, !rotating && !isempty(col.exchange), rotating, nk)
    naux = _coarse_aux(layout)
    o, ningredients, _ = _offsets(layout)
    P = col.Txi[:, 1:nl]
    W = [(2l - 1) / 2 * op.pitchgrid.wxi[a] * P[a, l] for l in 1:nl, a in 1:na]
    WDP, M = W * col.D * P, W * P
    q = nl^2
    cidx(l, t, j) = (t - 1) * nl * n + (l - 1) * n + j
    aux(j) = nl * nt * n + j
    src(s, k) = nl * nt * n + naux + 2 * (s - 1) + k
    block(lp, l, t, j) = (lp + nl * (l - 1)) + q * (t - 1) + q * nt * (j - 1)
    e = EntryList()
    for s in 1:ns, i in 1:nx
        j = i + nx * (s - 1)
        for t in 1:nt
            for d in 1:2   # poloidal advection, backward stencil where the speed is positive
                st = tr.Dt[d]
                for k in st.ptr[t]:(st.ptr[t+1]-1), lp in 1:nl, l in 1:nl
                    _add!(e, cidx(lp, t, j), cidx(l, st.idx[k], j), (d == 2 ? o.TP : o.TM) + block(lp, l, t, j), st.val[k])
                end
            end
            for lp in 1:nl, l in 1:nl   # mirror term and pitch angle scattering
                _add!(e, cidx(lp, t, j), cidx(l, t, j), o.MI + block(lp, l, t, j), 1.0)
                _add!(e, cidx(lp, t, j), cidx(l, t, j), o.NU + (rotating ? t + nt * (j - 1) : j), WDP[lp, l])
            end
            if layout.speed
                for ip in 1:nx, lp in 1:nl, l in 1:nl
                    jp = ip + nx * (s - 1)
                    _add!(e, cidx(lp, t, jp), cidx(l, t, j), o.SP + block(lp, l, t, jp), tr.Dx[ip, i])
                end
            end
            for k in 1:2, l in 1:nl   # sources and constraints
                _add!(e, cidx(l, t, j), src(s, k), (k == 1 ? o.S1 : o.S2) + j, M[l, 1])
                _add!(e, src(s, k), cidx(l, t, j), o.CP + l + nl * (t - 1) + nl * nt * (i - 1) + nl * nt * nx * (k - 1), 1.0)
            end
            rotating || _add!(e, aux(j), cidx(1, t, j), o.SW + t, -1.0)
        end
        rotating || _add!(e, aux(j), aux(j), o.ONE + 1, 1.0)
    end
    for t in 1:nt, l in 1:nl, lp in 1:nl, j in 1:n, jp in 1:n   # coupling of speeds and species
        point = rotating ? t - 1 : 0
        _add!(e, cidx(lp, t, jp), cidx(l, t, j), o.KL + j + n * (jp - 1) + n^2 * point, M[lp, l])
        _add!(e, cidx(lp, t, jp), cidx(l, t, j), o.KG + j + n * (jp - 1) + n^2 * (l - 1) + n^2 * nl * point, M[lp, l])
        jp == j && _add!(e, cidx(lp, t, jp), cidx(l, t, j), o.ONE + 1, M[lp, l] * w[8])
    end
    # exchange correction of a rotating plasma: unknowns m = ⟨M c⟩ and u = ⟨M K0 c⟩
    for k in 1:nk
        _add!(e, aux(k), aux(k), o.ONE + 1, 1.0)
        _add!(e, aux(nk + k), aux(nk + k), o.ONE + 1, 1.0)
        for t in 1:nt, j in 1:n
            r = t + nt * (j - 1)
            _add!(e, aux(k), cidx(1, t, j), o.XM + k + nk * (r - 1), -1.0)
            _add!(e, aux(nk + k), cidx(1, t, j), o.XK + k + nk * (r - 1), -1.0)
            for lp in 1:nl
                _add!(e, cidx(lp, t, j), aux(k), o.AG + r + nt * n * (k - 1), M[lp, 1])
                _add!(e, cidx(lp, t, j), aux(nk + k), o.XA + r + nt * n * (k - 1), -M[lp, 1])
            end
        end
    end
    if layout.exchange
        for t in 1:nt, lp in 1:nl, j in 1:n, jp in 1:n
            _add!(e, cidx(lp, t, jp), aux(j), o.KX + jp + n * (j - 1), M[lp, 1])
        end
    end
    # products that take the advection speeds at the nodes to the moment blocks
    dP = cat(_stencil_matrix(tr.Da[1]) * P, _stencil_matrix(tr.Da[2]) * P; dims=3)
    project(Q) = [W[lp, a] * Q[a, l] for lp in 1:nl, l in 1:nl, a in 1:na]
    projections = (WP=reshape(project(P), q, na), WDb=reshape(project(dP[:, :, 2]), q, na), WDf=reshape(project(dP[:, :, 1]), q, na),
        PT=Matrix(transpose(P)))
    return e, nl * nt * n + naux + 2ns, ningredients, layout, projections, P, W
end

function _coarse_ingredients(bop::BatchedDKE, layout::CoarseLayout, projections::NamedTuple)
    na, nt, nx, ns = bop.dims
    B, n, nl = bop.nbatch, nx * ns, layout.nl
    q, cols = nl^2, nt * n * B
    d = bop.data
    o, ningredients, sizes = _offsets(layout)
    Θ = similar(d.wa, ningredients, B)
    store!(name, A) = (view(Θ, (o[name]+1):(o[name]+sizes[name]), :) .= reshape(A, sizes[name], B); nothing)
    view(Θ, 1:1, :) .= 1
    nodes(A) = reshape(A, na, cols)
    tmp, blocks = similar(d.wa, na, cols), similar(d.wa, q, cols)
    tmp .= max.(nodes(d.wt), 0)
    store!(:TP, mul!(blocks, projections.WP, tmp))
    tmp .= min.(nodes(d.wt), 0)
    store!(:TM, mul!(blocks, projections.WP, tmp))
    tmp .= max.(nodes(d.wa), 0)
    mul!(blocks, projections.WDb, tmp)
    tmp .= min.(nodes(d.wa), 0)
    store!(:MI, mul!(blocks, projections.WDf, tmp, true, true))
    layout.speed && store!(:SP, mul!(blocks, projections.WP, nodes(d.wx)))
    store!(:NU, d.nuscale)
    store!(:S1, d.S1)
    store!(:S2, d.S2)
    store!(:CP, projections.PT * reshape(d.constraints, na, nt * nx * 2 * B))
    store!(:SW, d.sw)
    store!(:KL, d.KlocalT)
    modes = similar(d.wa, n, n, nl, B)
    modes .= view(reshape(d.KGHT, n, n, bop.nL, B), :, :, 1:nl, :)
    if layout.rotating   # field particle terms scale with the density of their own species: rows j' of KGH
        local_modes = similar(d.wa, n, n, nl, nt, B)
        local_modes .= reshape(modes, n, n, nl, 1, B) .* d.coldens
        store!(:KG, local_modes)
    else
        store!(:KG, modes)
    end
    layout.exchange && store!(:KX, d.Kexchange)
    if layout.nk > 0
        store!(:XM, d.XM)
        store!(:XK, d.XK)
        AG = copy(d.XKV)
        _batched_gemm!('N', 'N', 1.0, d.XA, d.XG, -1.0, AG)
        store!(:AG, AG)
        store!(:XA, d.XA)
    end
    return Θ
end
