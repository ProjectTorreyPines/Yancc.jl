# Solver for axisymmetric fields (nzeta = 1).
#
# GMRES on the bordered kinetic system with a two level right preconditioner:
#
#   1. coarse solve on the low Legendre moments in pitch angle at every poloidal
#      point, speed and species, together with the source amplitudes. These carry the
#      null space of the collision operator, which the constraints remove, and the
#      coupling between speeds and species that dominates at high collisionality.
#      At least three moments are needed: streaming and the mirror force leave a
#      parallel flow ∝ B undetermined until the l = 2 moment (viscosity) is kept;
#   2. sparse LU of the plane problem (advection in θ and α, pitch angle scattering
#      and the diagonal of the remaining collision terms) at each speed and species.
#
# The Krylov iteration is left with the coupling of the higher Legendre moments
# between speeds through energy scattering and the field particle operator.

"""
    plane_matrix(op, i, s)

Sparse matrix on the `(α, θ)` plane of speed node `i` and species `s`: advection in θ
and α, pitch angle scattering, and the diagonal of the terms that couple speeds.
"""
function plane_matrix(op::DKEOperator{Float64}, i::Integer, s::Integer)
    na, nt, nz, nx, ns = state_size(op)
    @assert nz == 1
    tr, col, w = op.trajectories, op.collisions, op.weights
    j = i + nx * (s - 1)
    shift = col.Klocal[j, j] + w[8]
    scatter = -col.weights[1] * col.nus[i, s] / 2
    rows, cols, vals = Int[], Int[], Float64[]
    idx(a, t) = a + na * (t - 1)
    for t in 1:nt, a in 1:na
        r = idx(a, t)
        wind = tr.wt[a, t, 1, i, s]
        st = wind > 0 ? tr.Dt[2] : tr.Dt[1]
        for k in st.ptr[t]:(st.ptr[t+1]-1)
            push!(rows, r), push!(cols, idx(a, st.idx[k])), push!(vals, w[3] * wind * st.val[k])
        end
        wind = tr.wa[a, t, 1, i, s]
        st = wind > 0 ? tr.Da[2] : tr.Da[1]
        for k in st.ptr[a]:(st.ptr[a+1]-1)
            push!(rows, r), push!(cols, idx(st.idx[k], t)), push!(vals, w[2] * wind * st.val[k])
        end
        for a2 in 1:na
            col.D[a, a2] == 0 && continue
            push!(rows, r), push!(cols, idx(a2, t)), push!(vals, scatter * col.D[a, a2])
        end
        push!(rows, r), push!(cols, r), push!(vals, shift)
    end
    return sparse(rows, cols, vals, na * nt, na * nt)
end

# Coarse space of the functions constant on each (pitch angle, θ) plane, with the
# source amplitudes. The operator maps it into itself.
struct ConstantCoarseSpace
    restriction::Vector{Float64} # weights of the plane average, [α, θ]
    KZ::Matrix{Float64}          # bordered operator applied to the coarse basis
    factor::LU{Float64,Matrix{Float64},Vector{Int}}
    rc::Vector{Float64}
end

function ConstantCoarseSpace(op::DKEOperator{Float64}, border::DKEBorder)
    na, nt, nz, nx, ns = state_size(op)
    N, np, n = na * nt * nx * ns, na * nt, nx * ns
    restriction = vec(op.pitchgrid.wxi ./ sum(op.pitchgrid.wxi) .* transpose(vec(op.surface_weights)))
    nc = n + 2ns
    KZ = zeros(N + 2ns, nc)
    e = zeros(N + 2ns)
    for j in 1:nc
        rng = j <= n ? (((j-1)*np+1):(j*np)) : ((N+j-n):(N+j-n))
        e[rng] .= 1
        apply_bordered!(view(KZ, :, j), op, border, e)
        e[rng] .= 0
    end
    E = zeros(nc, nc)
    for j in 1:nc
        _restrict!(view(E, :, j), view(KZ, :, j), restriction, n)
    end
    return ConstantCoarseSpace(restriction, KZ, lu(E), zeros(nc))
end

function _restrict!(rc::AbstractVector, r::AbstractVector, restriction::Vector{Float64}, n::Integer)
    np = length(restriction)
    @inbounds for j in 1:n
        rc[j] = dot(restriction, view(r, ((j-1)*np+1):(j*np)))
    end
    rc[(n+1):end] .= view(r, (n*np+1):length(r))
    return rc
end

# Coarse space of the first `nl` Legendre moments in pitch angle at every poloidal
# point, speed and species, with the source amplitudes: the functions on which the
# collision operator is small or couples speeds and species as strongly as pitch angle
# scattering does. Its Galerkin operator is sparse; the surface averaged exchange term
# enters through the auxiliary unknowns c̄ = ⟨c_{l=0}⟩.
struct MomentCoarseSpace{F}
    nl::Int
    P::Matrix{Float64}   # Legendre polynomials P_l(ξ_a), [α, l]
    W::Matrix{Float64}   # projection onto them, [l, α]
    factor::F
    rc::Vector{Float64}
    work::Vector{Float64}
end

"""
    moment_coarse_matrix(op, border, nl)

Galerkin projection of the bordered operator on the first `nl` Legendre moments at
every `(θ, x, s)`: unknowns `c[l, j, θ]` at index `(θ-1) nl n + (l-1) n + j` with
`j = i + nx (s-1)`, then the surface averages of `c[1, j, θ]`, then the sources.
"""
function moment_coarse_matrix(op::DKEOperator{Float64}, border::DKEBorder, nl::Integer)
    na, nt, nz, nx, ns = state_size(op)
    tr, col, w = op.trajectories, op.collisions, op.weights
    n = nx * ns
    @assert nl <= col.nL
    P = col.Txi[:, 1:nl]
    W = [(2l - 1) / 2 * op.pitchgrid.wxi[a] * P[a, l] for l in 1:nl, a in 1:na]
    cidx(l, t, j) = (t - 1) * nl * n + (l - 1) * n + j
    aux(j) = nl * nt * n + j
    src(s, k) = nl * nt * n + n + 2 * (s - 1) + k
    nc = nl * nt * n + n + 2ns
    rows, cols, vals = Int[], Int[], Float64[]
    add(r, c, v) = (push!(rows, r); push!(cols, c); push!(vals, v))

    # derivative of each polynomial along α with either stencil, and D P
    dP = zeros(na, nl, 2)
    for d in 1:2, l in 1:nl, a in 1:na
        st = tr.Da[d]
        dP[a, l, d] = sum(st.val[k] * P[st.idx[k], l] for k in st.ptr[a]:(st.ptr[a+1]-1); init=0.0)
    end
    WDP = W * col.D * P
    M = W * P   # identity up to the quadrature error of the pitch angle weights

    blk = zeros(nl, nl)
    for s in 1:ns, i in 1:nx
        j = i + nx * (s - 1)
        for t in 1:nt
            # poloidal advection
            for d in 1:2
                st = tr.Dt[d]
                fill!(blk, 0)
                for a in 1:na
                    wind = tr.wt[a, t, 1, i, s]
                    (wind > 0) == (d == 2) || continue
                    for lp in 1:nl, l in 1:nl
                        blk[lp, l] += W[lp, a] * w[3] * wind * P[a, l]
                    end
                end
                for k in st.ptr[t]:(st.ptr[t+1]-1), lp in 1:nl, l in 1:nl
                    blk[lp, l] == 0 || add(cidx(lp, t, j), cidx(l, st.idx[k], j), blk[lp, l] * st.val[k])
                end
            end
            # mirror term and pitch angle scattering
            fill!(blk, 0)
            for a in 1:na
                wind = tr.wa[a, t, 1, i, s]
                d = wind > 0 ? 2 : 1
                for lp in 1:nl, l in 1:nl
                    blk[lp, l] += W[lp, a] * w[2] * wind * dP[a, l, d]
                end
            end
            blk .-= col.weights[1] * col.nus[i, s] / 2 .* WDP
            for lp in 1:nl, l in 1:nl
                blk[lp, l] == 0 || add(cidx(lp, t, j), cidx(l, t, j), blk[lp, l])
            end
            # speed term
            if tr.has_speed && w[1] != 0
                for ip in 1:nx
                    fill!(blk, 0)
                    for a in 1:na, lp in 1:nl, l in 1:nl
                        blk[lp, l] += W[lp, a] * w[1] * tr.wx[a, t, 1, ip, s] * P[a, l]
                    end
                    for lp in 1:nl, l in 1:nl
                        add(cidx(lp, t, ip + nx * (s - 1)), cidx(l, t, j), blk[lp, l] * tr.Dx[ip, i])
                    end
                end
            end
            # sources and constraints
            for k in 1:2
                for l in 1:nl
                    add(cidx(l, t, j), src(s, k), M[l, 1] * border.sources[i, s, k])
                    v = sum(border.constraints[a, t, 1, i, k] * P[a, l] for a in 1:na)
                    add(src(s, k), cidx(l, t, j), v)
                end
            end
            add(aux(j), cidx(1, t, j), -op.surface_weights[t, 1])
        end
        add(aux(j), aux(j), 1.0)
    end
    # terms coupling speeds and species, local in θ
    for t in 1:nt, l in 1:nl, lp in 1:nl, j in 1:n, jp in 1:n
        v = M[lp, l] * (col.Klocal[jp, j] + col.KGH[jp, j, l] + (jp == j ? w[8] : 0.0))
        v == 0 || add(cidx(lp, t, jp), cidx(l, t, j), v)
    end
    if !isempty(col.exchange)
        for t in 1:nt, lp in 1:nl, j in 1:n, jp in 1:n
            v = M[lp, 1] * col.Kexchange[jp, j]
            v == 0 || add(cidx(lp, t, jp), aux(j), v)
        end
    end
    return sparse(rows, cols, vals, nc, nc), P, W
end

function MomentCoarseSpace(op::DKEOperator{Float64}, border::DKEBorder, nl::Integer)
    E, P, W = moment_coarse_matrix(op, border, nl)
    return MomentCoarseSpace(Int(nl), P, W, lu(E), zeros(size(E, 1)), zeros(system_size(op)))
end

# z = coarse solution, r = r - K z
function _coarse_correction!(z::AbstractVector, C::MomentCoarseSpace, r::AbstractVector, op::DKEOperator, border::DKEBorder)
    na, nt, nz, nx, ns = state_size(op)
    n, nl = nx * ns, C.nl
    N = na * nt * n
    R = reshape(view(r, 1:N), na, nt, n)
    Z = reshape(view(z, 1:N), na, nt, n)
    rc = C.rc
    cm = reshape(view(rc, 1:(nl*n*nt)), n, nl, nt)
    @inbounds for j in 1:n, t in 1:nt, l in 1:nl
        acc = 0.0
        for a in 1:na
            acc += C.W[l, a] * R[a, t, j]
        end
        cm[j, l, t] = acc
    end
    rc[(nl*n*nt+1):(nl*n*nt+n)] .= 0
    rc[(nl*n*nt+n+1):end] .= view(r, (N+1):length(r))
    c = C.factor \ rc
    cc = reshape(view(c, 1:(nl*n*nt)), n, nl, nt)
    @inbounds for j in 1:n, t in 1:nt, a in 1:na
        acc = 0.0
        for l in 1:nl
            acc += C.P[a, l] * cc[j, l, t]
        end
        Z[a, t, j] = acc
    end
    z[(N+1):end] .= view(c, (nl*n*nt+n+1):length(c))
    apply_bordered!(C.work, op, border, z)
    r .-= C.work
    return z
end

function _coarse_correction!(z::AbstractVector, C::ConstantCoarseSpace, r::AbstractVector, op::DKEOperator, border::DKEBorder)
    np = length(C.restriction)
    n = length(C.rc) - (length(r) - (length(r) ÷ np) * np)
    N = n * np
    c = _restrict!(C.rc, r, C.restriction, n)
    ldiv!(C.factor, c)
    mul!(r, C.KZ, c, -1.0, 1.0)
    @inbounds for j in 1:n
        z[((j-1)*np+1):(j*np)] .= c[j]
    end
    z[(N+1):end] .= view(c, (n+1):length(c))
    return z
end

"""
    TokamakPreconditioner(op; coarse=:moments, moments=4)

Preconditioner of the bordered kinetic system of an axisymmetric field, see
[`solve_dke_tokamak`](@ref). It may be reused for an operator with the same grids and
similar plasma parameters.

`coarse` selects the coarse space: `:moments`, the first `moments` Legendre moments in
pitch angle at every poloidal point, speed and species, or `:constants`, the functions
constant on each (pitch angle, θ) plane, which is cheaper to set up but loses its
effect as collisionality rises.
"""
struct TokamakPreconditioner{F,C}
    planes::Vector{F}   # LU of each plane, index i + nx (s - 1)
    coarse::C
    buffer::Vector{Float64}
end

function TokamakPreconditioner(op::DKEOperator{Float64}, border::DKEBorder=DKEBorder(op); coarse::Symbol=:moments,
    moments::Integer=4)
    na, nt, nz, nx, ns = state_size(op)
    @assert nz == 1 "the tokamak solver needs nzeta = 1"
    @assert coarse in (:moments, :constants)
    @assert moments >= 3 "the moment coarse space needs at least three moments"
    planes = [lu(plane_matrix(op, i, s)) for s in 1:ns for i in 1:nx]
    space = coarse == :moments ? MomentCoarseSpace(op, border, min(moments, op.collisions.nL)) : ConstantCoarseSpace(op, border)
    return TokamakPreconditioner(planes, space, zeros(na * nt))
end

# z = M⁻¹ r for the bordered operator of `op`; r is overwritten
function _precondition!(z::AbstractVector, P::TokamakPreconditioner, r::AbstractVector, op::DKEOperator, border::DKEBorder)
    np, n = length(P.buffer), length(P.planes)
    _coarse_correction!(z, P.coarse, r, op, border)
    @inbounds for j in 1:n
        rng = ((j-1)*np+1):(j*np)
        ldiv!(P.buffer, P.planes[j], view(r, rng))
        view(z, rng) .+= P.buffer
    end
    return z
end

# Scaling of the equations in which the residual is measured: for the kinetic equation
# the weights of ∫d³v ⟨r²/F_M⟩ / n, which treat all species alike; unit row sums for the
# constraints.
function _row_scaling(op::DKEOperator{Float64}, border::DKEBorder)
    na, nt, nz, nx, ns = state_size(op)
    N = na * nt * nz * nx * ns
    S = zeros(N + 2ns)
    Sf = reshape(view(S, 1:N), na, nt, nz, nx, ns)
    x, wx, wxi = op.speedgrid.x, op.speedgrid.wx, op.pitchgrid.wxi
    for (s, sp) in enumerate(op.species), i in 1:nx, z in 1:nz, t in 1:nt, a in 1:na
        Sf[a, t, z, i, s] = sqrt(exp(-x[i]^2) * x[i]^2 * wx[i] * wxi[a] * op.surface_weights[t, z]) / sp(x[i] * sp.v_thermal)
    end
    for (s, sp) in enumerate(op.species), k in 1:2
        S[N+2(s-1)+k] = 1 / (sum(view(border.constraints, :, :, :, :, k)) * sp(0.0))
    end
    return S
end

"Solution vector `[vec(f1); sources]` of the bordered system, e.g. as an initial guess."
function solution_vector(sol::DKESolution)
    ns = length(sol.particle_source)
    y = zeros(eltype(sol.f1), length(sol.f1) + 2ns)
    y[1:length(sol.f1)] .= vec(sol.f1)
    for s in 1:ns
        y[length(sol.f1)+2s-1] = sol.particle_source[s]
        y[length(sol.f1)+2s] = sol.heat_source[s]
    end
    return y
end

"""
    solve_dke_tokamak(op; EparB=0, rtol=1e-6, restart=80, maxiter=400,
                      preconditioner=TokamakPreconditioner(op), guess=nothing)

Solve the drift kinetic equation on a flux surface of an axisymmetric field
(`nzeta = 1`) by preconditioned GMRES. `rtol` is the relative residual of the kinetic
equation in the norm `∫d³v ⟨r²/F_M⟩ / n` summed over species.

`preconditioner` may come from an earlier, nearby operator on the same grids and
`guess` from its [`DKESolution`](@ref), as between the iterations of a transport solver.

Deep in the Pfirsch-Schlüter regime rounding limits the residual, which may stop above
a small `rtol`; there is a warning when it stops above 1e-6.
"""
function solve_dke_tokamak(op::DKEOperator{Float64}; EparB::Real=0.0, rtol::Real=1e-6, restart::Integer=80,
    maxiter::Integer=400, preconditioner::TokamakPreconditioner=TokamakPreconditioner(op),
    guess::Union{Nothing,DKESolution}=nothing)

    sol, status = _solve_dke_tokamak(op, preconditioner, guess; EparB, rtol, restart, maxiter)
    if status == :maxiter
        @warn "solve_dke_tokamak: relative residual $(sol.residual) after $(sol.iterations) iterations, rtol = $rtol"
    elseif status == :stagnated && sol.residual > 1e-6
        @warn "solve_dke_tokamak: rounding limits the relative residual to $(sol.residual), the system is ill conditioned at this collisionality"
    end
    return sol
end

function _solve_dke_tokamak(op::DKEOperator{Float64}, preconditioner::TokamakPreconditioner, guess; EparB, rtol, restart, maxiter)
    @assert op.field.nzeta == 1 "the tokamak solver needs nzeta = 1"
    border = DKEBorder(op)
    b = zeros(system_size(op))
    b[1:prod(state_size(op))] .= vec(dke_rhs(op, EparB))
    y = guess === nothing ? zeros(system_size(op)) : solution_vector(guess)
    y, iterations, residual, status = _gmres_bordered!(y, op, border, preconditioner, b; rtol, restart, maxiter)
    return DKESolution(op, y; iterations, residual), status
end

# GMRES on the bordered system with right hand side `b`, in the scaled norm; `b` is overwritten
function _gmres_bordered!(y::Vector{Float64}, op::DKEOperator{Float64}, border::DKEBorder, preconditioner::TokamakPreconditioner,
    b::Vector{Float64}; rtol, restart, maxiter)
    S = _row_scaling(op, border)
    b .*= S
    tmp = similar(b)
    A!(out, v) = (apply_bordered!(out, op, border, v); out .*= S; out)
    M!(z, v) = (tmp .= v ./ S; _precondition!(z, preconditioner, tmp, op, border))
    return gmres!(y, A!, b, M!; rtol, restart, maxiter)
end

"""
    TokamakCache()

State kept between solves on one flux surface whose plasma parameters change little,
as in the iterations of a transport solver: the preconditioner and the last solution,
see [`solve_dke_tokamak!`](@ref).
"""
mutable struct TokamakCache
    preconditioner::Union{Nothing,TokamakPreconditioner}
    solution::Union{Nothing,DKESolution{Float64}}
    dims::NTuple{5,Int}
    fresh_iterations::Int   # iterations of the solve for which the preconditioner was built
    plasma::Vector{Float64} # temperatures, densities and Erho for which it was built
    # last full solve: its inputs, solution vector and residual b - A y
    base_state::Vector{Float64}
    base_vector::Vector{Float64}
    base_residual::Vector{Float64}
    base_residual_norm::Float64
    solves::Int             # calls, preconditioners built and GMRES iterations so far
    rebuilds::Int
    iterations::Int
end
TokamakCache() = TokamakCache(nothing, nothing, (0, 0, 0, 0, 0), 0, Float64[], Float64[], Float64[], Float64[], 0.0, 0, 0, 0)

_plasma_state(op::DKEOperator) = Float64[(sp.temperature for sp in op.species)..., (sp.density for sp in op.species)...,
    (sp.temperature for sp in op.background)..., (sp.density for sp in op.background)..., op.Erho]

# largest relative change of a temperature or density, or of Erho, since `reference`
function _plasma_change(op::DKEOperator, reference::Vector{Float64})
    state = _plasma_state(op)
    length(state) == length(reference) || return Inf
    return maximum(abs(a - b) / max(abs(a), abs(b), floatmin()) for (a, b) in zip(state, reference))
end

# every input of the solve, with the scale against which a change of each is measured
function _input_state(op::DKEOperator, EparB::Real)
    state, scale = Float64[], Float64[]
    for sp in Iterators.flatten((op.species, op.background))
        append!(state, (sp.temperature, sp.density, sp.dTdrho, sp.dndrho))
        append!(scale, (sp.temperature, sp.density, abs(sp.dTdrho) + sp.temperature, abs(sp.dndrho) + sp.density))
    end
    T = op.species[1].temperature
    append!(state, (op.Erho, EparB))
    append!(scale, (abs(op.Erho) + T, abs(EparB) + floatmin()))
    return state, scale
end

function _input_change(op::DKEOperator, EparB::Real, reference::Vector{Float64})
    state, scale = _input_state(op, EparB)
    length(state) == length(reference) || return Inf
    return maximum(abs(state[k] - reference[k]) / scale[k] for k in eachindex(state))
end

"""
    solve_dke_tokamak!(cache, op; EparB=0, rtol=1e-6, restart=80, maxiter=400,
                       max_change=0.2, perturbation=1e-5, perturbation_rtol=1e-4)

[`solve_dke_tokamak`](@ref) starting from the last solution in `cache` and reusing its
preconditioner. The preconditioner is rebuilt and the solve starts afresh when the
grids have changed, when a temperature, a density or `Erho` differs by more than the
fraction `max_change` from the plasma the preconditioner was built for, or when the
solve does not converge within the iterations it took then (plus five).

An iterative solve stops anywhere below `rtol`, so on their own the fluxes are smooth
functions of the plasma parameters only to that level, which is not enough for a
transport solver that differentiates them by finite differences. When no input
differs by more than the fraction `perturbation` from those of the last full solve,
with solution `y₀` and residual `r₀ = b₀ - A₀ y₀`, the solution returned is therefore
`y₀ + δ` with `A δ = b - A y₀ - r₀` solved to `perturbation_rtol` relative to its own
right hand side: `δ` vanishes for unchanged inputs, and the fluxes are those of
`A⁻¹ (b - r₀)`, a smooth function of the inputs within `rtol` of the exact one.
"""
function solve_dke_tokamak!(cache::TokamakCache, op::DKEOperator{Float64}; EparB::Real=0.0, rtol::Real=1e-6,
    restart::Integer=80, maxiter::Integer=400, max_change::Real=0.2, perturbation::Real=1e-5, perturbation_rtol::Real=1e-4)
    dims = state_size(op)
    cache.solves += 1
    reusable = cache.preconditioner !== nothing && cache.solution !== nothing && cache.dims == dims
    budget = min(maxiter, cache.fresh_iterations + 5)

    # perturbation of the last full solve
    if reusable && cache.base_residual_norm <= rtol && _input_change(op, EparB, cache.base_state) <= perturbation
        border = DKEBorder(op)
        N = prod(dims)
        b = zeros(system_size(op))
        # same operations as for the base residual, so that unchanged inputs give exactly zero
        apply_bordered!(b, op, border, cache.base_vector)
        b .= .-b
        view(b, 1:N) .+= vec(dke_rhs(op, EparB))
        b .-= cache.base_residual
        delta, iterations, _, status = _gmres_bordered!(zeros(system_size(op)), op, border, cache.preconditioner, b;
            rtol=perturbation_rtol, restart, maxiter=budget)
        cache.iterations += iterations
        if status == :converged
            delta .+= cache.base_vector
            return DKESolution(op, delta; iterations, residual=cache.base_residual_norm)
        end
    end

    sol = nothing
    if reusable && _plasma_change(op, cache.plasma) <= max_change
        sol, status = _solve_dke_tokamak(op, cache.preconditioner, cache.solution; EparB, rtol, restart, maxiter=budget)
        cache.iterations += sol.iterations
        (status == :converged || (status == :stagnated && sol.residual <= 1e-6)) || (sol = nothing)
    end
    if sol === nothing
        cache.preconditioner = TokamakPreconditioner(op)
        cache.dims = dims
        cache.plasma = _plasma_state(op)
        sol = solve_dke_tokamak(op; EparB, rtol, restart, maxiter, preconditioner=cache.preconditioner)
        cache.fresh_iterations = sol.iterations
        cache.rebuilds += 1
        cache.iterations += sol.iterations
    end
    cache.solution = sol

    # keep the solve as the base of later perturbations
    border = DKEBorder(op)
    y = solution_vector(sol)
    r = zeros(system_size(op))
    apply_bordered!(r, op, border, y)
    r .= .-r
    view(r, 1:prod(dims)) .+= vec(dke_rhs(op, EparB))
    cache.base_state = _input_state(op, EparB)[1]
    cache.base_vector = y
    cache.base_residual = r
    cache.base_residual_norm = sol.residual
    return sol
end
