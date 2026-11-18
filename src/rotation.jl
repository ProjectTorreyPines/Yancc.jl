# Sonic toroidal rotation of an axisymmetric plasma: the equilibrium in the rotating frame.
#
# With a toroidal flow ω(ρ) R² ∇ζ of the order of the ion thermal speed the lowest order
# distribution of every species is a Maxwellian in the frame rotating with ω, whose density
# varies on the flux surface (Hinton & Wong, Phys. Fluids 28, 3082 (1985); Sugama & Horton,
# Phys. Plasmas 4, 2215 (1997), eqs 3-6; Belli, Candy & Angioni, Plasma Phys. Control. Fusion
# 56, 124002 (2014), eqs 6-10):
#
#   n_s(θ) = n_s(θ*) exp(-λ_s(θ)),   λ_s = [Z_s e Φ(θ) - m_s ω² (R(θ)² - R(θ*)²)/2] / T_s,
#
# with the potential Φ(θ), zero at the reference point θ*, set by quasineutrality. The
# densities and density gradients of the species are those at θ*, as in Belli & Candy,
# Phys. Plasmas 25, 032301 (2018), eq 10.

"""
    Rotation(omega, domega_drho=0; theta_ref=1)

Toroidal rotation of the plasma: angular velocity `omega` [rad/s] along `+ζ` of the
[`Field`](@ref), common to all species, and its derivative `domega_drho` with respect to
the radial label of the field. The densities and density gradients of the species are
then those at the poloidal node `theta_ref` (`1` is `θ = 0`, the outboard midplane of the
usual poloidal angles).

`omega R` may be of the order of the thermal speeds (sonic rotation): the centrifugal and
Coriolis forces, the potential and the densities varying on the surface, and the drive
by the rotation shear are kept. The field must be axisymmetric and carry `R` and `dRdrho`.
"""
struct Rotation{T<:Real}
    omega::T
    domega_drho::T
    theta_ref::Int
    Rotation{T}(omega, domega_drho, theta_ref) where {T} = new{T}(omega, domega_drho, theta_ref)
end
function Rotation(omega::Real, domega_drho::Real=0.0; theta_ref::Integer=1)
    o, d = promote(float(omega), float(domega_drho))
    return Rotation{typeof(o)}(o, d, Int(theta_ref))
end

"""
    RotatingEquilibrium(field, species, background, rotation)

Lowest order state of a rotating plasma on the surface nodes:

  - `Phi[θ]`: potential [V], zero at the reference node
  - `lambda[θ, s]`, `dlambda[θ, s]`: potential energy of a kinetic species in the rotating
    frame over its temperature, relative to the reference node, and its derivative in θ
  - `density[θ, s]`, `bg_density[θ, b]`: densities of the kinetic and of the background
    species relative to the reference node
  - `dlnN[s]`: `-(m_s/T_s) (ω ω' R*² + ω² R* ∂R*/∂ρ)`, the radial derivative of the potential
    energy at the reference point over the temperature, which the gradient of the density
    there contains
"""
struct RotatingEquilibrium{T}
    omega::T
    domega_drho::T
    theta_ref::Int
    Phi::Vector{T}
    lambda::Matrix{T}
    dlambda::Matrix{T}
    density::Matrix{T}
    bg_density::Matrix{T}
    dlnN::Vector{T}
end

# Potential [V] at which the charge density equals that of the reference point:
# Σ Z_s n_s [exp(c_s - Z_s φ/T_s) - 1] = 0, with Z in elementary charges, T in eV and
# c_s the centrifugal energy over T_s. The left side decreases monotonically with φ.
function _quasineutral_potential(Z::AbstractVector, n::AbstractVector, T::AbstractVector, c::AbstractVector)
    phi = zero(promote_type(eltype(Z), eltype(n), eltype(T), eltype(c)))
    cap = minimum(T[s] / abs(Z[s]) for s in eachindex(Z))   # largest step: a change of one in every exponent
    converged = 0
    for _ in 1:1000
        f = zero(phi)
        df = zero(phi)
        for s in eachindex(Z)
            ex = exp(c[s] - Z[s] * phi / T[s])
            f += Z[s] * n[s] * (ex - 1)
            df -= Z[s]^2 * n[s] / T[s] * ex
        end
        step = -f / df
        abs(step) > cap && (step *= cap / abs(step))
        phi += step
        # two more steps once converged, so that the derivatives carried by dual numbers converge too
        abs(step) <= 1e-12 * cap && (converged += 1)
        converged >= 3 && return phi
    end
    error("Rotation: the quasineutrality equation for the potential did not converge")
end

function RotatingEquilibrium(field::Field, species::AbstractVector{<:LocalMaxwellian{T}}, background::AbstractVector,
    rotation::Rotation) where {T}
    @assert field.nzeta == 1 "toroidal rotation needs an axisymmetric field (nzeta = 1)"
    @assert !isempty(field.R) && !isempty(field.dRdrho) "toroidal rotation needs the major radius: Field(...; R, dRdrho)"
    nt, ns = field.ntheta, length(species)
    ref = rotation.theta_ref
    @assert 1 <= ref <= nt
    S = promote_type(T, typeof(rotation.omega))
    omega, domega = S(rotation.omega), S(rotation.domega_drho)
    all_species = collect(Iterators.flatten((species, background)))
    Z = [sp.species.charge / ELEMENTARY_CHARGE for sp in all_species]
    n = [sp.density for sp in all_species]
    Tev = [sp.temperature for sp in all_species]
    mT = [sp.species.mass / (sp.temperature * ELEMENTARY_CHARGE) for sp in all_species]   # m/T [s²/m²]
    R, dRdt = vec(field.R), vec(field.dRdt)
    Rref = R[ref]

    Phi = zeros(S, nt)
    lambda, dlambda, density = (zeros(S, nt, length(all_species)) for _ in 1:3)
    for t in 1:nt
        c = [mT[s] * omega^2 * (R[t]^2 - Rref^2) / 2 for s in eachindex(all_species)]
        Phi[t] = t == ref ? zero(S) : _quasineutral_potential(Z, n, Tev, c)
        for s in eachindex(all_species)
            lambda[t, s] = Z[s] * Phi[t] / Tev[s] - c[s]
            density[t, s] = exp(-lambda[t, s])
        end
        # ∂Φ/∂θ from the derivative of the quasineutrality equation
        num = sum(Z[s] * n[s] * density[t, s] * mT[s] for s in eachindex(all_species))
        den = sum(Z[s]^2 * n[s] * density[t, s] / Tev[s] for s in eachindex(all_species))
        dPhi = omega^2 * R[t] * dRdt[t] * num / den
        for s in eachindex(all_species)
            dlambda[t, s] = Z[s] * dPhi / Tev[s] - mT[s] * omega^2 * R[t] * dRdt[t]
        end
    end
    dlnN = S[-mT[s] * (omega * domega * Rref^2 + omega^2 * Rref * field.dRdrho[ref]) for s in 1:ns]
    return RotatingEquilibrium{S}(omega, domega, ref, Phi, lambda[:, 1:ns], dlambda[:, 1:ns], density[:, 1:ns], density[:, (ns+1):end], dlnN)
end

RotatingEquilibrium(field::Field, species::AbstractVector, background::AbstractVector, ::Nothing) = nothing
