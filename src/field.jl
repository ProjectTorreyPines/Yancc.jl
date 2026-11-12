# Magnetic field on one flux surface (yancc/field.py).

"""
    spectral_derivative_matrix(n, period_factor=1)

Fourier differentiation matrix on `n` equally spaced points of a `2π/period_factor`
periodic angle. For even `n` the Nyquist mode has zero derivative.
"""
function spectral_derivative_matrix(n::Integer, period_factor::Real=1)
    D = zeros(n, n)
    K = (n - 1) ÷ 2
    for j in 1:n, l in 1:n
        j == l && continue
        D[j, l] = 2 / n * sum(k * sin(2π * k * (l - j) / n) for k in 1:K; init=0.0) * period_factor
    end
    return D
end

"""
    Field(rho, B_sup_t, B_sup_z, B_sub_t, B_sub_z, Bmag, sqrtg; kw...)

Magnetic field on the flux surface `rho`, given as `(ntheta, nzeta)` arrays on uniform
grids `θ ∈ [0, 2π)`, `ζ ∈ [0, 2π/NFP)` of arbitrary poloidal and toroidal angles:
contravariant (`B_sup_*`) and covariant (`B_sub_*`) components, `|B|` and the Jacobian
`sqrtg` of `(rho, θ, ζ)` [m³]. All radial derivatives and fluxes computed with this
field refer to the same radial label `rho`.

An axisymmetric field uses `nzeta = 1`.

Keywords: `Psi` (toroidal flux through the last closed surface [Wb]), `iota`,
`R_major`, `a_minor` (used for diagnostics and normalizations only, estimated from the
field when omitted), `NFP`, `dBdt`, `dBdz` (angular derivatives of `|B|`, spectral by
default) and `B0`.
"""
struct Field
    rho::Float64
    ntheta::Int
    nzeta::Int
    NFP::Int
    theta::Vector{Float64}
    zeta::Vector{Float64}
    wtheta::Vector{Float64}
    wzeta::Vector{Float64}
    B_sup_t::Matrix{Float64}
    B_sup_z::Matrix{Float64}
    B_sub_t::Matrix{Float64}
    B_sub_z::Matrix{Float64}
    sqrtg::Matrix{Float64}
    Bmag::Matrix{Float64}
    dBdt::Matrix{Float64}
    dBdz::Matrix{Float64}
    bdotgradB::Matrix{Float64}
    BxgradrhodotgradB::Matrix{Float64}
    Bmag_fsa::Float64
    B2mag_fsa::Float64
    Psi::Float64
    iota::Float64
    R_major::Float64
    a_minor::Float64
    B0::Float64
    I::Float64
    G::Float64
    psi_rho::Float64
    psi_r::Float64
end

function Field(rho::Real, B_sup_t::AbstractMatrix, B_sup_z::AbstractMatrix, B_sub_t::AbstractMatrix,
    B_sub_z::AbstractMatrix, Bmag::AbstractMatrix, sqrtg::AbstractMatrix;
    Psi=nothing, iota=nothing, R_major=nothing, a_minor=nothing, NFP::Integer=1,
    dBdt=nothing, dBdz=nothing, B0=nothing)

    ntheta, nzeta = size(sqrtg)
    for a in (B_sup_t, B_sup_z, B_sub_t, B_sub_z, Bmag)
        @assert size(a) == (ntheta, nzeta)
    end
    Bmag = Matrix{Float64}(Bmag)
    sqrtg = Matrix{Float64}(sqrtg)
    if dBdt === nothing
        dBdt = spectral_derivative_matrix(ntheta) * Bmag
    end
    if dBdz === nothing
        dBdz = Bmag * spectral_derivative_matrix(nzeta, NFP)'
    end
    B0 = B0 === nothing ? sum(Bmag) / length(Bmag) : B0
    bdotgradB = (B_sup_t .* dBdt .+ B_sup_z .* dBdz) ./ Bmag
    BxgradrhodotgradB = (B_sub_z .* dBdt .- B_sub_t .* dBdz) ./ sqrtg
    mean(a) = sum(a) / length(a)
    fsa(a) = mean(a .* sqrtg) / mean(sqrtg)
    I = mean(B_sub_t)
    G = mean(B_sub_z)
    # Psi and iota follow from the flux integrals of sqrtg B^ζ and sqrtg B^θ; R_major
    # and a_minor are circular torus estimates.
    Psi = Psi === nothing ? π * mean(sqrtg .* B_sup_z) / rho : Psi
    iota = iota === nothing ? mean(sqrtg .* B_sup_t) / mean(sqrtg .* B_sup_z) : iota
    R_major = R_major === nothing ? abs(G) / B0 : R_major
    a_minor = a_minor === nothing ? sqrt(mean(abs.(sqrtg)) / (R_major * rho)) : a_minor
    psi_rho = Psi / π * rho
    theta = collect(range(0, 2π; length=ntheta + 1))[1:ntheta]
    zeta = collect(range(0, 2π / NFP; length=nzeta + 1))[1:nzeta]
    return Field(rho, ntheta, nzeta, NFP, theta, zeta, fill(2π / ntheta, ntheta), fill(2π / NFP / nzeta, nzeta),
        Matrix{Float64}(B_sup_t), Matrix{Float64}(B_sup_z), Matrix{Float64}(B_sub_t), Matrix{Float64}(B_sub_z),
        sqrtg, Bmag, Matrix{Float64}(dBdt), Matrix{Float64}(dBdz), bdotgradB, BxgradrhodotgradB,
        fsa(Bmag), fsa(Bmag .^ 2), Psi, iota, R_major, a_minor, B0, I, G, psi_rho, psi_rho / a_minor)
end

"""
    field_from_boozer(rho, Bmag, I, G, iota, Psi; kw...)

Field in Boozer coordinates from `|B|` on uniform Boozer angles, the Boozer currents
`I` (toroidal) and `G` (poloidal) [T m], the rotational transform `iota` and the
toroidal flux `Psi` through the last closed surface [Wb], with `rho = sqrt(ψ_t/Psi)`.
"""
function field_from_boozer(rho::Real, Bmag::AbstractMatrix, I::Real, G::Real, iota::Real, Psi::Real; kw...)
    sqrtg0 = (G + iota * I) ./ Bmag .^ 2
    sqrtg = sqrtg0 .* (Psi * rho / π)
    return Field(rho, iota ./ sqrtg0, 1 ./ sqrtg0, fill(float(I), size(Bmag)), fill(float(G), size(Bmag)), Bmag, sqrtg;
        Psi, iota, kw...)
end

"Flux surface average of a `(ntheta, nzeta)` array."
flux_surface_average(field::Field, f::AbstractMatrix) = sum(f .* field.sqrtg) / sum(field.sqrtg)

"Volume element `sqrtg dθ dζ` on the surface grid."
volume_element(field::Field) = field.sqrtg .* field.wtheta .* field.wzeta'
