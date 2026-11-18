# Helpers relating the reference data written by generate.py to the Julia layout.
#
# The Python code stores the state as (species, speed, pitch, theta, zeta) in C order;
# Yancc.jl uses f[pitch, theta, zeta, speed, species].

using NPZ

golden(name) = npzread(joinpath(@__DIR__, name * ".npz"))

"State array from a flat vector in the Python layout; `dims = (na, nt, nz, nx, ns)`."
function pystate(v::AbstractVector, dims)
    na, nt, nz, nx, ns = dims
    return permutedims(reshape(v, nz, nt, na, nx, ns), (3, 2, 1, 4, 5))
end

"The deterministic state of generate.py's `test_vector`."
test_vector(n) = [sin(0.37 * i + 0.2) * (1 + 0.3 * cos(0.0171 * i)) for i in 0:(n-1)]

"Largest difference relative to the largest reference entry."
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

function golden_species(d, prefix="")
    return [LocalMaxwellian(Species(d[prefix*"mass"][i] / Yancc.PROTON_MASS, d[prefix*"charge"][i] / Yancc.ELEMENTARY_CHARGE),
        d[prefix*"temperature"][i], d[prefix*"density"][i], d[prefix*"dTdrho"][i], d[prefix*"dndrho"][i])
            for i in eachindex(d[prefix*"mass"])]
end

function golden_field(d, prefix="field_"; derivatives=true)
    g(k) = d[prefix*k]
    kw = derivatives ? (; dBdt=g("dBdt"), dBdz=g("dBdz")) : (;)
    # the major radius of the nodes, which a rotating plasma needs
    haskey(d, prefix * "R") && (kw = (; kw..., R=g("R"), dRdrho=g("dRdrho")))
    return Field(g("rho"), g("B_sup_t"), g("B_sup_z"), g("B_sub_t"), g("B_sub_z"), g("Bmag"), g("sqrtg");
        Psi=g("Psi"), iota=g("iota"), R_major=g("R_major"), a_minor=g("a_minor"), NFP=Int(g("NFP")), B0=g("B0"), kw...)
end

"""
    miller_field(; r, a, R0, q, Bunit, kappa=1, delta=0, zeta=0, shift=0, s_kappa=0, s_delta=0,
                 s_zeta=0, s_zmag=0, ipccw=1, btccw=1, ntheta)

`Field` of a Miller local equilibrium (Miller et al., Phys. Plasmas 5, 973 (1998)) with the major
radius of its nodes, on `ntheta` points uniform in the angle `θ` of

    R = R0 + r cos(θ + asin(δ) sin θ - ζ sin 2θ),    Z = Z0 + κ r sin θ

(the squareness as in Arbon, Candy & Belli, Plasma Phys. Control. Fusion 63, 012001 (2021)).
Lengths in m, `Bunit = (q/r) dψ/dr` in T, `shift = dR0/dr`, `s_kappa = (r/κ) dκ/dr`,
`s_delta = r dδ/dr`, `s_zeta = r dζ/dr`, `s_zmag = dZ0/dr`. The radial label is `ρ = r/a`, `ζ` of
the field is the cylindrical angle of the right-handed `(R, φ, Z)`, counter-clockwise seen from
above, and `ipccw`, `btccw = 1` are a current and a toroidal field in that direction.
"""
function miller_field(; r::Real, a::Real, R0::Real, q::Real, Bunit::Real, kappa::Real=1.0, delta::Real=0.0, zeta::Real=0.0,
    shift::Real=0.0, s_kappa::Real=0.0, s_delta::Real=0.0, s_zeta::Real=0.0, s_zmag::Real=0.0, ipccw::Real=1, btccw::Real=1,
    ntheta::Integer)
    theta = range(0, 2π; length=ntheta + 1)[1:ntheta]
    x = asin(delta)
    R, Rr, Jr, dl = (zeros(ntheta) for _ in 1:4)
    for (k, t) in enumerate(theta)
        argr = t + x * sin(t) - zeta * sin(2t)
        R[k] = R0 + r * cos(argr)
        Rr[k] = shift + cos(argr) - sin(argr) * (sin(t) * s_delta / cos(x) - s_zeta * sin(2t))
        R_t = -r * sin(argr) * (1 + x * cos(t) - 2 * zeta * cos(2t))
        Z_r = s_zmag + kappa * (1 + s_kappa) * sin(t)
        Z_t = kappa * r * cos(t)
        Jr[k] = Rr[k] * Z_t - R_t * Z_r
        dl[k] = hypot(R_t, Z_t)
    end
    dpsidr = r * abs(Bunit) / abs(q)
    F = 2π * r * abs(Bunit) / (sum(Jr ./ R) * 2π / ntheta)
    Bp = dpsidr .* (dl ./ Jr) ./ R
    Bmag = hypot.(Bp, F ./ R)
    st, sz = -sign(ipccw), sign(btccw)
    col(v) = reshape(collect(float.(v)), ntheta, 1)
    return Field(r / a, col(st .* Bp ./ dl), col(sz .* F ./ R .^ 2), col(st .* Bp .* dl), col(sz .* F .+ 0 .* R),
        col(Bmag), col(-a .* R .* Jr); R_major=R0, a_minor=a, iota=st * sz / abs(q), R=col(R), dRdrho=col(a .* Rr))
end
