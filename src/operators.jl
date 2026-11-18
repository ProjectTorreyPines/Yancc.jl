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
  - `p1 = "4d"` (`"4z"` with `rotation`), `p2 = 4`: first derivative stencil and order of
    second derivatives; `p1_theta`, `p1_zeta`, `p1_alpha` override the stencil of one coordinate
  - `lnlambda = nothing`: Coulomb logarithm (pairwise values when `nothing`)
  - `operator_weights`: factors on the terms `(dx/dt, dα/dt, dθ/dt, dζ/dt, C_L, C_E, C_F, Krook)`
  - `energy_exchange = false`: keep the flux surface averaged exchange of density and
    energy between species
  - `rotation = nothing`: a [`Rotation`](@ref) of the plasma; the densities and density
    gradients of the species are then those at its reference point, the velocities are
    those in the rotating frame, and `rotation` of the operator holds the
    [`RotatingEquilibrium`](@ref)
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
    rotation::Union{Nothing,RotatingEquilibrium{T}}
end
DKEOperator{T}(field, pitchgrid, speedgrid, species, background, Erho, weights, trajectories, collisions, surface_weights) where {T} =
    DKEOperator{T}(field, pitchgrid, speedgrid, species, background, Erho, weights, trajectories, collisions, surface_weights, nothing)

function DKEOperator(field::Field, pitchgrid::UniformPitchAngleGrid, speedgrid::MaxwellSpeedGrid,
    species::AbstractVector{<:LocalMaxwellian}, Erho::Real=0.0;
    rotation::Union{Nothing,Rotation}=nothing, background::AbstractVector=LocalMaxwellian[], nL::Integer=8,
    p1::AbstractString=(rotation === nothing ? DEFAULT_P1 : DEFAULT_P1_ROTATION), p2::Integer=DEFAULT_P2,
    p1_theta::AbstractString=p1, p1_zeta::AbstractString=p1, p1_alpha::AbstractString=p1,
    lnlambda=nothing, operator_weights=DEFAULT_OPERATOR_WEIGHTS, energy_exchange::Bool=false)

    T = promote_type(typeof(float(Erho)), (typeof(sp.temperature) for sp in species)...,
        (typeof(sp.temperature) for sp in background)..., rotation === nothing ? Float64 : typeof(rotation.omega))
    sp = LocalMaxwellian{T}[_convert_maxwellian(T, s) for s in species]
    bg = LocalMaxwellian{T}[_convert_maxwellian(T, s) for s in background]
    w = NTuple{8,Float64}(Float64.(operator_weights))
    dr = volume_element(field)
    sw = dr ./ sum(dr)
    eq = rotation === nothing ? nothing : RotatingEquilibrium(field, sp, bg, Rotation{T}(rotation.omega, rotation.domega_drho, rotation.theta_ref))
    tr = Trajectories(field, pitchgrid, speedgrid, sp, T(Erho); p1, p1_theta, p1_zeta, p1_alpha, rotation=eq)
    col = CollisionOperator(pitchgrid, speedgrid, sp; background=bg, nL, p2, lnlambda, weights=w[5:7], energy_exchange,
        rotation=eq, surface_weights=sw)
    return DKEOperator{T}(field, pitchgrid, speedgrid, sp, bg, T(Erho), w, tr, col, sw, eq)
end

_convert_maxwellian(::Type{T}, s::LocalMaxwellian) where {T} =
    LocalMaxwellian{T}(_convert_species(T, T, s.species), T(s.temperature), T(s.density), T(s.dTdrho), T(s.dndrho), T(s.v_thermal))

"""
    map_parameters(f, S, op)

`DKEOperator{S}` with `f` applied to every number of `op` that depends on the plasma
profiles; the field, the grids and the stencils are shared with `op`. With
`f = ForwardDiff.value` this is the operator at the values of an operator of dual numbers.
"""
function map_parameters(f, ::Type{S}, op::DKEOperator) where {S}
    m(a::AbstractArray) = S[f(v) for v in a]
    sp(s::LocalMaxwellian) = LocalMaxwellian{S}(_convert_species(f, S, s.species), f(s.temperature), f(s.density), f(s.dTdrho),
        f(s.dndrho), f(s.v_thermal))
    tr, c, eq, rc = op.trajectories, op.collisions, op.rotation, op.collisions.rotating
    rotating = rc === nothing ? nothing : RotatingCollisions{S}(m(rc.nus), m(rc.Klocal), m(rc.KGH), m(rc.V), m(rc.M), m(rc.A), m(rc.KV), m(rc.MK), m(rc.G))
    rotation = eq === nothing ? nothing : RotatingEquilibrium{S}(f(eq.omega), f(eq.domega_drho), eq.theta_ref, m(eq.Phi), m(eq.lambda),
        m(eq.dlambda), m(eq.density), m(eq.bg_density), m(eq.dlnN))
    return DKEOperator{S}(op.field, op.pitchgrid, op.speedgrid, LocalMaxwellian{S}[sp(s) for s in op.species],
        LocalMaxwellian{S}[sp(s) for s in op.background], f(op.Erho), op.weights,
        Trajectories{S}(m(tr.wt), m(tr.wz), m(tr.wa), m(tr.wx), tr.has_speed, tr.Dt, tr.Dz, tr.Da, tr.Dx),
        CollisionOperator{S}(c.nL, c.weights, m(c.nus), c.D, m(c.CE), m(c.CD), m(c.GH), c.Txi, c.Txi_inv, m(c.exchange),
            m(c.Klocal), m(c.KGH), m(c.Kexchange), rotating),
        op.surface_weights, rotation)
end

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
    radial_drift(op)

Radial drift `dρ/dt` of the guiding centres at every node, `[α, θ, ζ, x, s]` [1/s]: the
magnetic drift and, in a rotating plasma, the drift in the gradient of the potential
energy of the rotating frame (E×B and centrifugal) and the Coriolis drift,

    dρ/dt = (m/(q ψ')) L[I v∥/B + ω R²]
          = v_m·∇ρ - B_ζ T ∂λ/∂θ /(q √g B²) + 2 m ω v∥ R ∂R/∂θ /(q √g B),

with `L` the collisionless operator and `ψ' = √g B^θ` (conservation of the canonical
angular momentum; Sugama & Horton 1997, eq 11, `W_a1`; Belli, Candy & Angioni 2014, eqs
12, 29).
"""
function radial_drift(op::DKEOperator)
    drift = radial_magnetic_drift(op)
    eq = op.rotation
    eq === nothing && return drift
    na, nt, nz, nx, ns = state_size(op)
    fd = op.field
    for (s, sp) in enumerate(op.species), i in 1:nx, t in 1:nt, a in 1:na
        vpar = op.speedgrid.x[i] * sp.v_thermal * op.pitchgrid.xi[a]
        Tq = sp.temperature * ELEMENTARY_CHARGE / sp.species.charge
        mq = sp.species.mass / sp.species.charge
        B = fd.Bmag[t, 1]
        drift[a, t, 1, i, s] += (-fd.B_sub_z[t, 1] * Tq * eq.dlambda[t, s] / B + 2 * mq * eq.omega * vpar * fd.R[t, 1] * fd.dRdt[t, 1]) /
                                (fd.sqrtg[t, 1] * B)
    end
    return drift
end

"""
    angular_momentum_drift(op, drift=radial_drift(op))

`(m²/(2 q ψ')) L[(I v∥/B + ω R²)² + μ R² B_p²/(m B)]` at every node, `[α, θ, ζ, x, s]`: the
toroidal angular momentum of a guiding centre times its radial drift, plus the part of the
gyromotion. It multiplies the rotation shear in the drive and gives the flux of angular
momentum (Sugama & Horton 1997, eqs 11 and 13, `W_aV`). Needs the major radius of the field.
"""
function angular_momentum_drift(op::DKEOperator{T}, drift::AbstractArray=radial_drift(op)) where {T}
    na, nt, nz, nx, ns = state_size(op)
    fd = op.field
    @assert nz == 1 && !isempty(fd.R) "the angular momentum needs an axisymmetric field with its major radius"
    omega = op.rotation === nothing ? zero(T) : op.rotation.omega
    out = zeros(T, na, nt, nz, nx, ns)
    for (s, sp) in enumerate(op.species), i in 1:nx, t in 1:nt, a in 1:na
        v, xi = op.speedgrid.x[i] * sp.v_thermal, op.pitchgrid.xi[a]
        m, q = sp.species.mass, sp.species.charge
        B, I, R = fd.Bmag[t, 1], fd.B_sub_z[t, 1], fd.R[t, 1]
        # ∂/∂θ of R² B_p²/(2B) = (R² B² - I²)/(2B)
        dh = R * fd.dRdt[t, 1] * B + (R^2 / 2 + I^2 / (2B^2)) * fd.dBdt[t, 1]
        out[a, t, 1, i, s] = m * (I * v * xi / B + omega * R^2) * drift[a, t, 1, i, s] +
                             m^2 / (2q) * v^3 * xi * (1 - xi^2) * dh / (fd.sqrtg[t, 1] * B^2)
    end
    return out
end

"""
    thermodynamic_forces(op, EparB)

Forces `(A1, A2, A3)` of each species (Conlin & Landreman 2026, eqs 25-27):
`A1 = dln(n)/dρ - q E_ρ/T - 3/2 dln(T)/dρ`, `A2 = dln(T)/dρ`, `A3 = q ⟨E∥B⟩/(T ⟨B²⟩)`.

In a rotating plasma `n` is the density at the reference point and `A1` contains in
addition the radial derivative of the potential energy there over the temperature,
`-(m/T) (ω ω' R*² + ω² R* ∂R*/∂ρ)`, so that `A1 + 3/2 A2` is the gradient of the flux
function `N` of `n = N exp(-E/T)` (Belli & Candy 2018, eqs 25, 31, 32).
"""
function thermodynamic_forces(op::DKEOperator{T}, EparB::Real=0.0) where {T}
    return map(op.species) do sp
        q = sp.species.charge / ELEMENTARY_CHARGE
        dlnn = sp.dndrho / sp.density
        dlnT = sp.dTdrho / sp.temperature
        (dlnn - q * op.Erho / sp.temperature - 3 / 2 * dlnT, dlnT, q / sp.temperature * EparB / op.field.B2mag_fsa)
    end
end

function _rotating_forces(op::DKEOperator, EparB::Real)
    forces = thermodynamic_forces(op, EparB)
    return [(A1 + op.rotation.dlnN[s], A2, A3) for (s, (A1, A2, A3)) in enumerate(forces)]
end

"""
    dke_rhs(op, EparB=0)

Drive of the kinetic equation, `[α, θ, ζ, x, s]` (Conlin & Landreman 2026, eq 24):
`-(A1 + x² A2) v_m·∇ρ F_M + A3 B v∥ F_M`.

In a rotating plasma (Sugama & Horton 1997, eqs 8-11, with `X_V = -dω/dψ`)

    -[A1 + (x² + λ) A2] dρ/dt F_M - (dω/dρ)/T (angular momentum drift) F_M + A3 B v∥ F_M,

with the Maxwellian of the local density, the drifts of [`radial_drift`](@ref) and
[`angular_momentum_drift`](@ref), and `λ` the potential energy over the temperature.
"""
function dke_rhs(op::DKEOperator{T}, EparB::Real=0.0) where {T}
    op.rotation === nothing || return _rotating_rhs(op, EparB)
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

function _rotating_rhs(op::DKEOperator{T}, EparB::Real) where {T}
    na, nt, nz, nx, ns = state_size(op)
    eq = op.rotation
    drift = radial_drift(op)
    momentum = angular_momentum_drift(op, drift)
    forces = _rotating_forces(op, EparB)
    rhs = zeros(promote_type(T, typeof(float(EparB))), na, nt, nz, nx, ns)
    for (s, sp) in enumerate(op.species), i in 1:nx
        x = op.speedgrid.x[i]
        v = x * sp.v_thermal
        A1, A2, A3 = forces[s]
        shear = eq.domega_drho / (sp.temperature * ELEMENTARY_CHARGE)
        for t in 1:nt
            F = sp(v) * eq.density[t, s]
            for a in 1:na
                rhs[a, t, 1, i, s] = -(A1 + (x^2 + eq.lambda[t, s]) * A2) * drift[a, t, 1, i, s] * F - shear * momentum[a, t, 1, i, s] * F +
                                     A3 * op.field.Bmag[t, 1] * v * op.pitchgrid.xi[a] * F
            end
        end
    end
    return rhs
end
