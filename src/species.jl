# Species, Maxwellians and collision frequencies (yancc/species.py).

# CODATA 2022, as in scipy.constants
const ELEMENTARY_CHARGE = 1.602176634e-19
const EPSILON_0 = 8.8541878188e-12
const HBAR = 1.0545718176461565e-34
const PROTON_MASS = 1.67262192595e-27

"""
    Species(mass, charge)

Particle species with `mass` in proton masses and `charge` in elementary charges.
Stored in kg and C. The charge may be a dual number, e.g. an average charge state that
depends on the temperature.
"""
struct Species{T<:Real}
    mass::T
    charge::T
    function Species(mass::Real, charge::Real)
        m, q = promote(float(mass) * PROTON_MASS, float(charge) * ELEMENTARY_CHARGE)
        return new{typeof(m)}(m, q)
    end
    Species{T}(mass_kg::Real, charge_C::Real, ::Val{:SI}) where {T} = new{T}(mass_kg, charge_C)
end

# species of number type `T`, with `f` applied to the mass and the charge
_convert_species(f, ::Type{T}, s::Species) where {T} = Species{T}(f(s.mass), f(s.charge), Val(:SI))

const Electron = Species(1 / 1836.15267343, -1)
const Hydrogen = Species(1, 1)
const Deuterium = Species(2, 1)
const Tritium = Species(3, 1)
const Helium4 = Species(4, 2)

"""
    LocalMaxwellian(species, temperature, density, dTdrho, dndrho)

Maxwellian on one flux surface: `temperature` in eV, `density` in m⁻³ and their
derivatives with respect to the radial coordinate `rho` of the `Field`.
"""
struct LocalMaxwellian{T<:Real}
    species::Species{T}
    temperature::T
    density::T
    dTdrho::T
    dndrho::T
    v_thermal::T
end

function LocalMaxwellian(species::Species, temperature::Real, density::Real, dTdrho::Real, dndrho::Real)
    T, n, dT, dn, _ = promote(float(temperature), float(density), float(dTdrho), float(dndrho), species.charge)
    S = typeof(T)
    sp = _convert_species(S, S, species)
    vth = sqrt(2 * T * ELEMENTARY_CHARGE / sp.mass)
    return LocalMaxwellian{S}(sp, T, n, dT, dn, vth)
end

"""
    from_scale_lengths(species, temperature, density, aLT, aLn)

Maxwellian from normalized inverse gradient scale lengths `aLT = -dln(T)/drho`,
`aLn = -dln(n)/drho`.
"""
from_scale_lengths(species::Species, temperature::Real, density::Real, aLT::Real, aLn::Real) =
    LocalMaxwellian(species, temperature, density, -temperature * aLT, -density * aLn)

"Maxwellian distribution at speed `v` [m/s], in s³/m⁶."
(m::LocalMaxwellian)(v) = m.density / (sqrt(π) * m.v_thermal)^3 * exp(-v^2 / m.v_thermal^2)

"Chandrasekhar function."
chandrasekhar(x) = (erf(x) - 2x / sqrt(π) * exp(-x^2)) / (2x^2)

function debye_length(a::LocalMaxwellian, b::LocalMaxwellian)
    den = a.density / (a.temperature * ELEMENTARY_CHARGE) * a.species.charge^2 +
          b.density / (b.temperature * ELEMENTARY_CHARGE) * b.species.charge^2
    return sqrt(EPSILON_0 / den)
end

"""
    coulomb_logarithm(a, b)

`ln(b_max/b_min)` with `b_max` the Debye length of the pair and `b_min` the larger of
the 90° impact parameter and the thermal de Broglie length.
"""
function coulomb_logarithm(a::LocalMaxwellian, b::LocalMaxwellian)
    mr = a.species.mass * b.species.mass / (a.species.mass + b.species.mass)
    v2 = a.v_thermal^2 + b.v_thermal^2
    bperp = abs(a.species.charge * b.species.charge / (4π * EPSILON_0 * mr * v2))
    bdebroglie = HBAR / (2 * mr * sqrt(v2))
    return log(debye_length(a, b) / max(bperp, bdebroglie))
end

"Prefactor `Γ_ab = q_a² q_b² lnΛ / (4π ε₀² m_a²)` of the collision frequencies."
function gamma_ab(a::LocalMaxwellian, b::LocalMaxwellian, lnlambda=nothing)
    ll = lnlambda === nothing ? coulomb_logarithm(a, b) : lnlambda
    return a.species.charge^2 * b.species.charge^2 * ll / (4π * EPSILON_0^2 * a.species.mass^2)
end

"Deflection frequency of species `a` at speed `v` on a Maxwellian background `b`."
function nuD_ab(a::LocalMaxwellian, b::LocalMaxwellian, v, lnlambda=nothing)
    y = v / b.v_thermal
    return gamma_ab(a, b, lnlambda) * b.density / v^3 * (erf(y) - chandrasekhar(y))
end

"Parallel (energy) diffusion frequency of species `a` at speed `v` on background `b`."
function nupar_ab(a::LocalMaxwellian, b::LocalMaxwellian, v, lnlambda=nothing)
    return 2 * gamma_ab(a, b, lnlambda) * b.density / v^3 * chandrasekhar(v / b.v_thermal)
end
