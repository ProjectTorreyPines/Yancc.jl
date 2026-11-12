# Drift kinetic operator, sources, constraints and drive (yancc/_trajectories.py DKE,
# yancc/_misc.py).
#
# The state is f1[α, θ, ζ, x, s], the departure of species s from its Maxwellian, at
# the nodes of the pitch angle, poloidal, toroidal and speed grids.

const DEFAULT_OPERATOR_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)

"""
    DKEOperator(field, pitchgrid, speedgrid, species, Erho; kw...)

Linearized drift kinetic operator `(advection - C)` on one flux surface.

Keywords:

  - `background`: Maxwellians included in the test particle collision terms only
  - `nL = 8`: Legendre modes of the Rosenbluth potentials
  - `p1 = "4d"`, `p2 = 4`: first derivative stencil and order of second derivatives;
    `p1_theta`, `p1_zeta`, `p1_alpha` override the stencil of one coordinate
  - `lnlambda = nothing`: Coulomb logarithm (pairwise values when `nothing`)
  - `operator_weights`: factors on the terms `(dx/dt, dα/dt, dθ/dt, dζ/dt, C_L, C_E, C_F, Krook)`
  - `energy_exchange = false`: keep the flux surface averaged exchange of density and
    energy between species
"""
struct DKEOperator{T}
    field::Field
    pitchgrid::UniformPitchAngleGrid
    speedgrid::MaxwellSpeedGrid
    species::Vector{LocalMaxwellian{T}}
    background::Vector{LocalMaxwellian{T}}
    Erho::T
    weights::NTuple{8,Float64}
    trajectories::Trajectories{T}
    collisions::CollisionOperator{T}
    surface_weights::Matrix{Float64}
end

function DKEOperator(field::Field, pitchgrid::UniformPitchAngleGrid, speedgrid::MaxwellSpeedGrid,
    species::AbstractVector{<:LocalMaxwellian}, Erho::Real=0.0;
    background::AbstractVector=LocalMaxwellian[], nL::Integer=8, p1::AbstractString=DEFAULT_P1, p2::Integer=DEFAULT_P2,
    p1_theta::AbstractString=p1, p1_zeta::AbstractString=p1, p1_alpha::AbstractString=p1,
    lnlambda=nothing, operator_weights=DEFAULT_OPERATOR_WEIGHTS, energy_exchange::Bool=false)

    T = promote_type(typeof(float(Erho)), (typeof(sp.temperature) for sp in species)...,
        (typeof(sp.temperature) for sp in background)...)
    sp = LocalMaxwellian{T}[_convert_maxwellian(T, s) for s in species]
    bg = LocalMaxwellian{T}[_convert_maxwellian(T, s) for s in background]
    w = NTuple{8,Float64}(Float64.(operator_weights))
    tr = Trajectories(field, pitchgrid, speedgrid, sp, T(Erho); p1, p1_theta, p1_zeta, p1_alpha)
    col = CollisionOperator(pitchgrid, speedgrid, sp; background=bg, nL, p2, lnlambda, weights=w[5:7], energy_exchange)
    dr = volume_element(field)
    return DKEOperator{T}(field, pitchgrid, speedgrid, sp, bg, T(Erho), w, tr, col, dr ./ sum(dr))
end

_convert_maxwellian(::Type{T}, s::LocalMaxwellian) where {T} =
    LocalMaxwellian{T}(s.species, T(s.temperature), T(s.density), T(s.dTdrho), T(s.dndrho), T(s.v_thermal))

"Dimensions `(nalpha, ntheta, nzeta, nx, nspecies)` of the state."
state_size(op::DKEOperator) =
    (op.pitchgrid.nalpha, op.field.ntheta, op.field.nzeta, op.speedgrid.nx, length(op.species))

"""
    apply!(out, op, f)

`out = (advection - C) f` for a state `f[α, θ, ζ, x, s]`.
"""
function apply!(out::AbstractArray{<:Any,5}, op::DKEOperator, f::AbstractArray{<:Any,5})
    fill!(out, zero(eltype(out)))
    apply_trajectories!(out, op.trajectories, f, op.weights)
    apply_collisions!(out, op.collisions, f, op.surface_weights)
    op.weights[8] != 0 && (out .+= op.weights[8] .* f)
    return out
end

"""
    DKEBorder(op)

Border of the kinetic system: particle and heat sources (columns) and density and
energy constraints (rows), two per species, which remove the null space of the
operator (Conlin & Landreman 2026, eqs 42-44).

  - `sources[x, s, j]`: speed dependence of source `j` of species `s`, constant in
    pitch angle and on the surface
  - `constraints[α, θ, ζ, x, j]`: weights of the density (`j = 1`) and energy (`j = 2`)
    moments, the same for every species
"""
struct DKEBorder{T}
    sources::Array{T,3}
    constraints::Array{Float64,5}
end

function DKEBorder(op::DKEOperator{T}) where {T}
    na, nt, nz, nx, ns = state_size(op)
    x = op.speedgrid.x
    sources = zeros(T, nx, ns, 2)
    for (s, sp) in enumerate(op.species), i in 1:nx
        F = sp(x[i] * sp.v_thermal)
        sources[i, s, 1] = (x[i]^2 - 5 / 2) * F
        sources[i, s, 2] = (-2 / 3 * x[i]^2 + 1) * F
    end
    dr = volume_element(op.field)
    constraints = zeros(na, nt, nz, nx, 2)
    for i in 1:nx, z in 1:nz, t in 1:nt, a in 1:na
        d3v = 2π * op.pitchgrid.wxi[a] * x[i]^2 * op.speedgrid.wx[i]
        constraints[a, t, z, i, 1] = 2π * d3v * dr[t, z]
        constraints[a, t, z, i, 2] = 2π * d3v * dr[t, z] * x[i]^2
    end
    return DKEBorder{T}(sources, constraints)
end

"Number of unknowns of the bordered system."
system_size(op::DKEOperator) = prod(state_size(op)) + 2 * length(op.species)

"""
    apply_bordered!(y, op, border, v)

Bordered system `[A B; C 0]` acting on `v = [vec(f); sources]`, with the two source
amplitudes of each species (particle, heat) stored species by species.
"""
function apply_bordered!(y::AbstractVector, op::DKEOperator, border::DKEBorder, v::AbstractVector)
    dims = state_size(op)
    na, nt, nz, nx, ns = dims
    N = prod(dims)
    f = reshape(view(v, 1:N), dims)
    out = reshape(view(y, 1:N), dims)
    apply!(out, op, f)
    nr = na * nt * nz
    F = reshape(f, nr * nx, ns)
    Cp = reshape(view(border.constraints, :, :, :, :, 1), nr * nx)
    Ce = reshape(view(border.constraints, :, :, :, :, 2), nr * nx)
    O = reshape(out, nr, nx, ns)
    @inbounds for s in 1:ns
        sp, sh = v[N+2s-1], v[N+2s]
        for i in 1:nx
            @views O[:, i, s] .+= border.sources[i, s, 1] * sp + border.sources[i, s, 2] * sh
        end
        y[N+2s-1] = dot(Cp, view(F, :, s))
        y[N+2s] = dot(Ce, view(F, :, s))
    end
    return y
end

"""
    radial_magnetic_drift(op)

Radial magnetic drift `v_m·∇ρ` at every node, `[α, θ, ζ, x, s]` [1/s].
"""
function radial_magnetic_drift(field::Field, pitchgrid::UniformPitchAngleGrid, speedgrid::MaxwellSpeedGrid,
    species::AbstractVector{<:LocalMaxwellian{T}}) where {T}
    na, nt, nz, nx, ns = pitchgrid.nalpha, field.ntheta, field.nzeta, speedgrid.nx, length(species)
    vm = zeros(T, na, nt, nz, nx, ns)
    for (s, sp) in enumerate(species), i in 1:nx, z in 1:nz, t in 1:nt, a in 1:na
        v = speedgrid.x[i] * sp.v_thermal
        vm[a, t, z, i, s] = -(sp.species.mass * v^2 / sp.species.charge * (1 + pitchgrid.xi[a]^2) /
                              (2 * field.Bmag[t, z]^3) * field.BxgradrhodotgradB[t, z])
    end
    return vm
end
radial_magnetic_drift(op::DKEOperator) = radial_magnetic_drift(op.field, op.pitchgrid, op.speedgrid, op.species)

"""
    thermodynamic_forces(op, EparB)

Forces `(A1, A2, A3)` of each species (Conlin & Landreman 2026, eqs 25-27):
`A1 = dln(n)/dρ - q E_ρ/T - 3/2 dln(T)/dρ`, `A2 = dln(T)/dρ`, `A3 = q ⟨E∥B⟩/(T ⟨B²⟩)`.
"""
function thermodynamic_forces(op::DKEOperator{T}, EparB::Real=0.0) where {T}
    return map(op.species) do sp
        q = sp.species.charge / ELEMENTARY_CHARGE
        dlnn = sp.dndrho / sp.density
        dlnT = sp.dTdrho / sp.temperature
        (dlnn - q * op.Erho / sp.temperature - 3 / 2 * dlnT, dlnT, q / sp.temperature * EparB / op.field.B2mag_fsa)
    end
end

"""
    dke_rhs(op, EparB=0)

Drive of the kinetic equation, `[α, θ, ζ, x, s]` (Conlin & Landreman 2026, eq 24):
`-(A1 + x² A2) v_m·∇ρ F_M + A3 B v∥ F_M`.
"""
function dke_rhs(op::DKEOperator{T}, EparB::Real=0.0) where {T}
    na, nt, nz, nx, ns = state_size(op)
    vm = radial_magnetic_drift(op)
    forces = thermodynamic_forces(op, EparB)
    rhs = zeros(promote_type(T, typeof(float(EparB))), na, nt, nz, nx, ns)
    for (s, sp) in enumerate(op.species), i in 1:nx
        x = op.speedgrid.x[i]
        v = x * sp.v_thermal
        F = sp(v)
        A1, A2, A3 = forces[s]
        for z in 1:nz, t in 1:nt, a in 1:na
            rhs[a, t, z, i, s] = -(A1 + x^2 * A2) * vm[a, t, z, i, s] * F +
                                 A3 * op.field.Bmag[t, z] * v * op.pitchgrid.xi[a] * F
        end
    end
    return rhs
end
