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
    return Field(g("rho"), g("B_sup_t"), g("B_sup_z"), g("B_sub_t"), g("B_sub_z"), g("Bmag"), g("sqrtg");
        Psi=g("Psi"), iota=g("iota"), R_major=g("R_major"), a_minor=g("a_minor"), NFP=Int(g("NFP")), B0=g("B0"), kw...)
end
