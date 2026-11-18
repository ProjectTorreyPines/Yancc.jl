# Solution of the drift kinetic equation and its moments (yancc/solution.py).

"""
    DKESolution

Solution on one flux surface. All fluxes are flux surface averages of the flux
through the surface per unit of the field's radial label, e.g. `⟨Γ·∇ρ⟩`; multiply by
`dV/dρ` for the total flow through the surface.

  - `f1[α, θ, ζ, x, s]`: departure from the Maxwellian
  - `particle_flux[s]`: `⟨∫d³v f v_m·∇ρ⟩` [m⁻³ s⁻¹]
  - `heat_flux[s]`: total energy flux `⟨∫d³v (m v²/2) f v_m·∇ρ⟩` [W m⁻³]; the conductive
    heat flux is `heat_flux - 5/2 T particle_flux`

In a rotating plasma (`rotation` of the operator) the velocities are those of the rotating
frame, the radial drift is [`radial_drift`](@ref), and the energy carried by `heat_flux` is
`m v²/2 + T λ`, the kinetic energy in the rotating frame plus the potential energy relative
to the reference point (Sugama & Horton 1997, eq 13; Belli & Candy 2018, eq 37). `Vpar` is
the parallel flow in the rotating frame.

  - `momentum_flux[s]`: `⟨∫d³v m v∥ B f v_m·∇ρ⟩`
  - `angular_momentum_flux[s]`: radial flux of toroidal angular momentum [N m⁻² ... per unit
    of the radial label: kg s⁻²], conjugate to `-dω/dρ` (zero when the field has no major radius)
  - `Vpar[θ, ζ, s]`: parallel flow [m/s]
  - `VparB[s]`: `⟨V∥ B⟩` [T m/s]
  - `JparB`: `⟨J∥ B⟩ = Σ_s n_s q_s ⟨V∥ B⟩` [A T m⁻²], bootstrap (and Ohmic) current
  - `J_rho`: radial current `Σ_s q_s particle_flux[s]`
  - `particle_source[s]`, `heat_source[s]`: amplitudes of the solvability sources
  - `iterations`, `residual`: Krylov iterations and relative residual (0 for a direct solve)
"""
struct DKESolution{T}
    f1::Array{T,5}
    particle_flux::Vector{T}
    heat_flux::Vector{T}
    momentum_flux::Vector{T}
    angular_momentum_flux::Vector{T}
    Vpar::Array{T,3}
    VparB::Vector{T}
    JparB::T
    J_rho::T
    particle_source::Vector{T}
    heat_source::Vector{T}
    iterations::Int
    residual::Float64
end

function DKESolution(op::DKEOperator, y::AbstractVector; iterations::Integer=0, residual::Real=0.0)
    dims = state_size(op)
    na, nt, nz, nx, ns = dims
    N = prod(dims)
    f1 = Array(reshape(view(y, 1:N), dims))
    S = eltype(f1)
    eq = op.rotation
    vm = radial_drift(op)
    has_momentum = nz == 1 && !isempty(op.field.R)
    pm = has_momentum ? angular_momentum_drift(op, vm) : vm
    dr = op.surface_weights
    x, wx, xi, wxi = op.speedgrid.x, op.speedgrid.wx, op.pitchgrid.xi, op.pitchgrid.wxi
    B = op.field.Bmag

    particle_flux, heat_flux, momentum_flux, angular_momentum_flux, VparB = (zeros(S, ns) for _ in 1:5)
    Vpar = zeros(S, nt, nz, ns)
    JparB = zero(S)
    for (s, sp) in enumerate(op.species), i in 1:nx
        vth, m = sp.v_thermal, sp.species.mass
        v = x[i] * vth
        for z in 1:nz, t in 1:nt
            # Maxwellian of the local density and potential energy in the rotating frame
            density = eq === nothing ? sp.density : sp.density * eq.density[t, s]
            F0 = eq === nothing ? sp(v) : sp(v) * eq.density[t, s]
            energy = eq === nothing ? m * v^2 / 2 : m * v^2 / 2 + sp.temperature * ELEMENTARY_CHARGE * eq.lambda[t, s]
            for a in 1:na
                d3v = 2π * vth^3 * wxi[a] * x[i]^2 * wx[i]
                f = F0 + f1[a, t, z, i, s]
                g = f * vm[a, t, z, i, s] * d3v * dr[t, z]
                particle_flux[s] += g
                heat_flux[s] += energy * g
                momentum_flux[s] += m * v * xi[a] * B[t, z] * g
                has_momentum && (angular_momentum_flux[s] += f * pm[a, t, z, i, s] * d3v * dr[t, z])
                Vpar[t, z, s] += v * xi[a] * f * d3v / density
            end
        end
    end
    for (s, sp) in enumerate(op.species)
        VparB[s] = sum(B[t, z] * Vpar[t, z, s] * dr[t, z] for t in 1:nt, z in 1:nz)
        JparB += eq === nothing ? sp.density * sp.species.charge * VparB[s] :
                 sp.density * sp.species.charge * sum(eq.density[t, s] * B[t, 1] * Vpar[t, 1, s] * dr[t, 1] for t in 1:nt)
    end
    J_rho = sum(sp.species.charge * particle_flux[s] for (s, sp) in enumerate(op.species))
    particle_source = S[y[N+2s-1] for s in 1:ns]
    heat_source = S[y[N+2s] for s in 1:ns]
    return DKESolution{S}(f1, particle_flux, heat_flux, momentum_flux, angular_momentum_flux, Vpar, VparB, JparB, J_rho,
        particle_source, heat_source, iterations, residual)
end

"Conductive heat flux `heat_flux - 5/2 T particle_flux` of each species [W m⁻³]."
conductive_heat_flux(op::DKEOperator, sol::DKESolution) =
    [sol.heat_flux[s] - 5 / 2 * sp.temperature * ELEMENTARY_CHARGE * sol.particle_flux[s] for (s, sp) in enumerate(op.species)]

"""
    energy_flux(sol, op; frame=:rotating)

Energy flux of each species of a rotating plasma with the potential energy measured from
the flux surface average of the potential and from the axis of rotation,

    ⟨∫d³v f [m v²/2 + q (Φ - ⟨Φ⟩) - m ω² R²/2] dρ/dt⟩,

the energy of Belli, Candy & Angioni 2014, eq 7, and of the energy flux of Belli & Candy
2018, eq 37. `sol.heat_flux` measures the potential energy from the reference point
instead; the two differ by the potential energy there times the particle flux.

`frame = :laboratory` adds `ω` times the flux of angular momentum: the kinetic energy of
the laboratory frame is `m v²/2 + m ω R (v_ζ + ω R) - m ω² R²/2`.

Without rotation both are `sol.heat_flux`.
"""
function energy_flux(sol::DKESolution, op::DKEOperator; frame::Symbol=:rotating)
    op.rotation === nothing && return copy(sol.heat_flux)
    return energy_flux(op, sol.particle_flux, sol.heat_flux, sol.angular_momentum_flux; frame)
end

"`energy_flux` of the fluxes of particles, energy (`heat_flux`) and angular momentum of each species, e.g. those of [`nondissipative_fluxes`](@ref)."
function energy_flux(op::DKEOperator, particle_flux::AbstractVector, heat_flux::AbstractVector, angular_momentum_flux::AbstractVector;
    frame::Symbol=:rotating)
    @assert frame in (:rotating, :laboratory)
    eq = op.rotation
    eq === nothing && return copy(heat_flux)
    Rref = op.field.R[eq.theta_ref, 1]
    Phi_fsa = sum(op.surface_weights[t, 1] * eq.Phi[t] for t in eachindex(eq.Phi))
    return map(eachindex(op.species)) do s
        sp = op.species[s]
        # potential energy of the reference point [J]
        E = -sp.species.charge * Phi_fsa - sp.species.mass * eq.omega^2 * Rref^2 / 2
        heat_flux[s] + E * particle_flux[s] + (frame == :laboratory ? eq.omega * angular_momentum_flux[s] : zero(eq.omega))
    end
end

# Nodes and weights of the Gauss quadrature whose Jacobi matrix has the diagonal `d` and the
# off-diagonal `e`; `mu0` is the integral of the weight function (Golub & Welsch 1969).
function _gauss_rule(d::Vector{Float64}, e::Vector{Float64}, mu0::Float64)
    F = eigen(SymTridiagonal(d, e))
    return F.values, mu0 .* F.vectors[1, :] .^ 2
end
_gauss_hermite(n::Integer) = _gauss_rule(zeros(n), [sqrt(k / 2) for k in 1:(n-1)], sqrt(pi))          # ∫ e^{-s²} ds
_gauss_laguerre(n::Integer) = _gauss_rule([2k + 1.0 for k in 0:(n-1)], [Float64(k) for k in 1:(n-1)], 1.0)   # ∫₀^∞ e^{-y} dy

"""
    nondissipative_fluxes(op)

`(; particle_flux, heat_flux, angular_momentum_flux)` of each species: the fluxes that
do not depend on collisions and that a flux surface without up-down symmetry carries in
a rotating plasma (the fluxes `Γ^H`, `q^H`, `Π^H` of Sugama & Horton 1997, related to the
parallel gyroviscosity; NEO's "gyroviscous" fluxes). They are to be added to those of a
[`DKESolution`](@ref), in the same units and conventions (`heat_flux` with the energy
`m v²/2 + T λ`; see [`energy_flux`](@ref)), and vanish on an up-down symmetric surface.

The shear drive of the kinetic equation is `-(ω'/T) F dπ/dt = -L[F G]` with `G = (ω'/T) G₂`,
`G₂ = (m²/(2 q ψ')) [(I v∥/B + ω R²)² + μ R² B_p²/(m B)]`, so that the solution is
`f = f̃ - F G`, where `f̃` is driven by the gradients and, through the collision operator
acting on `F G`, by the shear (the form of Hinton & Wong 1985). The physical fluxes are the
moments of the collision operator (momentum and energy balance); written as moments of
`f`, they are those of the solution plus moments of the Maxwellian,

    Γ^H = (ω'/T) ⟨∫d³v F G₂ dρ/dt⟩,   Q^H = (ω'/T) ⟨∫d³v F G₂ (m v²/2 + T λ) dρ/dt⟩,
    Π^H = -⟨∫d³v F [A1 + (x² + λ) A2] G₂ dρ/dt⟩,

with the forces of the drive. The coefficients are antisymmetric (no entropy production;
Sugama & Horton 1997, text between eqs 26 and 28) and the particle fluxes including `Γ^H`
are ambipolar (their eq 28), which those of the solution alone are not without up-down
symmetry. The velocity integrals are done exactly (Gauss-Hermite and Gauss-Laguerre rules).

Zero without `rotation`; for the gyroviscous momentum flux of a plasma at rest pass
`Rotation(0, 0)`. The fluxes driven by a parallel electric field are not included.
"""
function nondissipative_fluxes(op::DKEOperator{T}) where {T}
    ns = length(op.species)
    particle_flux, heat_flux, angular_momentum_flux = zeros(T, ns), zeros(T, ns), zeros(T, ns)
    eq = op.rotation
    eq === nothing && return (; particle_flux, heat_flux, angular_momentum_flux)
    fd = op.field
    forces = _rotating_forces(op, 0.0)
    sH, wH = _gauss_hermite(8)
    yL, wL = _gauss_laguerre(5)
    for (s, sp) in enumerate(op.species), t in 1:fd.ntheta
        m, q, vth = sp.species.mass, sp.species.charge, sp.v_thermal
        TJ = sp.temperature * ELEMENTARY_CHARGE
        A1, A2, _ = forces[s]
        B, I, R, sqrtg = fd.Bmag[t, 1], fd.B_sub_z[t, 1], fd.R[t, 1], fd.sqrtg[t, 1]
        psip = sqrtg * fd.B_sup_t[t, 1]
        lam = eq.lambda[t, s]
        weight = op.surface_weights[t, 1] * sp.density * eq.density[t, s] / sqrt(pi)
        for (j, y) in enumerate(yL), (i, sn) in enumerate(sH)
            u, w2 = vth * sn, vth^2 * y           # v∥ and v⊥²
            drift = -m / q * (2u^2 + w2) / (2B^3) * fd.BxgradrhodotgradB[t, 1] +
                    (-I * TJ / q * eq.dlambda[t, s] / B + 2 * m / q * eq.omega * u * R * fd.dRdt[t, 1]) / (sqrtg * B)
            G2 = m^2 / (2 * q * psip) * ((I * u / B + eq.omega * R^2)^2 + w2 * (R^2 * B^2 - I^2) / (2B^2))
            x2 = sn^2 + y
            g = weight * wH[i] * wL[j] * G2 * drift
            particle_flux[s] += eq.domega_drho / TJ * g
            heat_flux[s] += eq.domega_drho / TJ * TJ * (x2 + lam) * g
            angular_momentum_flux[s] -= (A1 + (x2 + lam) * A2) * g
        end
    end
    return (; particle_flux, heat_flux, angular_momentum_flux)
end
