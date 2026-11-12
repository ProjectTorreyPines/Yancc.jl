# Linearized Fokker-Planck collision operator (yancc/_collisions.py).
#
# C = C_L + C_E + C_F (Conlin & Landreman 2026, eqs 10-13): pitch angle scattering,
# energy scattering and the field particle term through the Rosenbluth potentials.
# The operator applied to the state is -C, so that the kinetic equation reads
# (advection - C) f = drive.

"""
    pitch_projection(Txi, wxi)

Node values in pitch angle to Legendre coefficients: the projection weighted by the
pitch quadrature. With more modes than nodes the minimum norm weighted fit is used.
"""
function pitch_projection(Txi::AbstractMatrix, wxi::AbstractVector)
    if size(Txi, 2) > size(Txi, 1)
        sw = sqrt.(wxi)
        return pinv(sw .* Txi) .* sw'
    end
    TW = Txi' .* wxi'
    return (TW * Txi) \ TW
end

"""
    pitch_scattering_matrix(pitchgrid, p2)

Finite difference Lorentz operator `∂²/∂α² + cot(α) ∂/∂α` on the pitch grid with
reflecting boundaries. Row sums are set to zero so isotropic functions are
annihilated exactly.
"""
function pitch_scattering_matrix(pitchgrid::UniformPitchAngleGrid, p2::Integer)
    na = pitchgrid.nalpha
    h = π / na
    D1 = fdfwd_matrix(string(p2) * "z", na, h, :symmetric)
    D2 = fd2_matrix(p2, na, h, :symmetric)
    sina = sqrt.(1 .- pitchgrid.xi .^ 2)
    cosa = -pitchgrid.xi
    D = (cosa ./ sina) .* D1 .+ D2
    return D - Diagonal(vec(sum(D; dims=2)))
end

"""
    energy_scattering_matrices(speedgrid, species, background, lnlambda)

`M[y, x, s]`: energy scattering `C_E` of species `s` acting on node values in speed,
summed over all field species and background Maxwellians. Galerkin discretization of
the flux form `C_E f = x⁻² ∂/∂x [a (f' + 2 τ x f)]`, `a = ν∥ x⁴/2`, `τ = T_a/T_b`.
"""
function energy_scattering_matrices(speedgrid::MaxwellSpeedGrid, species::AbstractVector{<:LocalMaxwellian{T}},
    background::AbstractVector, lnlambda) where {T}
    nx, ns = speedgrid.nx, length(species)
    x = speedgrid.x
    Vp, dVp = maxwell_vandermonde(x, nx)
    gq = speedgrid.wx .* exp.(-x .^ 2)
    mass = Vp' * Diagonal(gq .* x .^ 2) * Vp
    M = zeros(T, nx, nx, ns)
    for (s, spa) in enumerate(species)
        diffusion = zeros(T, nx)
        relaxation = zeros(T, nx)
        for spb in Iterators.flatten((species, background)), i in 1:nx
            a = nupar_ab(spa, spb, x[i] * spa.v_thermal, lnlambda) * x[i]^4 / 2
            tau = spa.temperature / spb.temperature
            diffusion[i] += a
            relaxation[i] += 2 * (tau - 1) * a * x[i]
        end
        stiffness = dVp' * Diagonal(gq .* diffusion) * dVp + dVp' * Diagonal(gq .* relaxation) * Vp
        modal = -(mass \ stiffness)
        M[:, :, s] .= speedgrid.xvander * modal * speedgrid.xvander_inv
    end
    return M
end

"""
    CollisionOperator(pitchgrid, speedgrid, species; background, nL, p2, lnlambda, weights, energy_exchange)

Discrete linearized Fokker-Planck operator for the kinetic `species`, with additional
`background` Maxwellians entering the test particle terms only.

  - `nus[x, s]`: deflection frequency
  - `D`: Lorentz operator in pitch angle
  - `CE[y, x, s]`: energy scattering
  - `CD[y, x, p, s]`, `GH[y, l, x, p, s]`: field particle term of species `s` acting on
    species `p`: the part local in velocity, and the Rosenbluth potential part per
    Legendre mode `l`
  - `exchange[y, x, p, s]`: correction removing the flux surface averaged exchange of
    density and energy between species (empty when `energy_exchange = true`)

`weights` scale `(C_L, C_E, C_F)`. `lnlambda = nothing` uses pairwise Coulomb
logarithms.
"""
struct CollisionOperator{T}
    nL::Int
    weights::NTuple{3,Float64}
    nus::Matrix{T}
    D::Matrix{Float64}
    CE::Array{T,3}
    CD::Array{T,4}
    GH::Array{T,5}
    Txi::Matrix{Float64}
    Txi_inv::Matrix{Float64}
    exchange::Array{T,4}
    # assembled for the matrix-vector product, acting on the (speed, species) index
    Klocal::Matrix{T}       # -(w_E C_E + w_F CD)
    KGH::Array{T,3}         # -w_F GH per Legendre mode
    Kexchange::Matrix{T}
end

function CollisionOperator(pitchgrid::UniformPitchAngleGrid, speedgrid::MaxwellSpeedGrid,
    species::AbstractVector{<:LocalMaxwellian{T}};
    background::AbstractVector=LocalMaxwellian{T}[], nL::Integer=8, p2::Integer=DEFAULT_P2, lnlambda=nothing,
    weights=(1.0, 1.0, 1.0), energy_exchange::Bool=false,
    potentials::RosenbluthPotentials=RosenbluthPotentials(speedgrid, species, nL)) where {T}

    nx, ns = speedgrid.nx, length(species)
    x = speedgrid.x
    wL, wE, wF = Float64.(weights)

    nus = zeros(T, nx, ns)
    for (s, spa) in enumerate(species), spb in Iterators.flatten((species, background)), i in 1:nx
        nus[i, s] += nuD_ab(spa, spb, x[i] * spa.v_thermal, lnlambda)
    end
    D = pitch_scattering_matrix(pitchgrid, p2)
    CE = energy_scattering_matrices(speedgrid, species, background, lnlambda)

    CD = zeros(T, nx, nx, ns, ns)
    GH = zeros(T, nx, nL, nx, ns, ns)
    for (b, spb) in enumerate(species), (a, spa) in enumerate(species)
        va, vb = spa.v_thermal, spb.v_thermal
        ma, mb = spa.species.mass, spb.species.mass
        gamma = gamma_ab(spa, spb, lnlambda)
        v = x .* va
        Fa = spa.(v)
        # field distribution evaluated at the speeds of the test species grid
        xq = x .* (va / vb)
        Dab = first(maxwell_vandermonde(xq, nx)) .* exp.(-xq .^ 2)
        CD[:, :, a, b] .= (gamma * 4π * ma / mb .* Fa) .* (Dab * speedgrid.xvander_inv)
        pG = gamma .* Fa .* 2 .* v .^ 2 ./ va^4
        pH = -2 / va^2 * gamma .* Fa
        pdH = -2 .* v ./ va^2 .* (1 - ma / mb) .* gamma .* Fa
        for l in 1:nL
            modal = pG .* potentials.ddG[:, l, :, a, b] .+ pH .* potentials.H[:, l, :, a, b] .+
                    pdH .* potentials.dH[:, l, :, a, b]
            GH[:, l, :, a, b] .= modal * speedgrid.xvander_inv
        end
    end

    Txi = legendre_vandermonde(pitchgrid.xi, nL)
    Txi_inv = pitch_projection(Txi, pitchgrid.wxi)

    n = nx * ns
    idx(i, s) = i + nx * (s - 1)
    Klocal = zeros(T, n, n)
    KGH = zeros(T, n, n, nL)
    for b in 1:ns, a in 1:ns, j in 1:nx, i in 1:nx
        Klocal[idx(i, a), idx(j, b)] = -wF * CD[i, j, a, b] - (a == b ? wE * CE[i, j, a] : zero(T))
        for l in 1:nL
            KGH[idx(i, a), idx(j, b), l] = -wF * GH[i, l, j, a, b]
        end
    end

    if energy_exchange
        exchange = zeros(T, 0, 0, 0, 0)
        Kexchange = zeros(T, n, n)
    else
        # C_E and C_F map isotropic functions to isotropic functions through their
        # l = 0 blocks; C_L annihilates them.
        Kexchange = species_exchange_correction(speedgrid, species, Klocal .+ KGH[:, :, 1])
        exchange = permutedims(reshape(Kexchange, nx, ns, nx, ns), (1, 3, 2, 4))
    end
    return CollisionOperator{T}(nL, (wL, wE, wF), nus, D, CE, CD, GH, Txi, Txi_inv, exchange, Klocal, KGH, Kexchange)
end

"""
    species_exchange_correction(speedgrid, species, K0)

`(I - P) K0 (I - P) - K0`, with `K0` the isotropic block of the collision operator on
the (speed, species) index and `P` the projection of each species onto its density
and temperature perturbations along its density and energy moments. Added to the flux
surface average of the operator, it removes the exchange of density and energy
between species, which belongs to the slower evolution of the Maxwellians.
"""
function species_exchange_correction(speedgrid::MaxwellSpeedGrid, species::AbstractVector{<:LocalMaxwellian{T}},
    K0::AbstractMatrix) where {T}
    nx, ns = speedgrid.nx, length(species)
    x = speedgrid.x
    wv = x .^ 2 .* speedgrid.wx
    V = zeros(T, nx * ns, 2ns)
    Dm = zeros(T, 2ns, nx * ns)
    for (s, sp) in enumerate(species)
        F = sp.(x .* sp.v_thermal)
        a = sum(wv .* x .^ 2 .* F) / sum(wv .* F)
        mu = wv .* (x .^ 2 .- a)
        fhat = F ./ sum(wv .* F)
        phi = (x .^ 2 .- a) .* F
        phi = phi ./ sum(mu .* phi)
        rng = (1:nx) .+ nx * (s - 1)
        V[rng, s] .= fhat
        V[rng, ns+s] .= phi
        Dm[s, rng] .= wv
        Dm[ns+s, rng] .= mu
    end
    DK = Dm * K0
    W = K0 * V - V * (DK * V)
    return -(V * DK + W * Dm)
end

"""
    apply_collisions!(out, C, f, surface_weights)

Add `-C f` to `out` for a state `f[α, θ, ζ, x, s]`. `surface_weights[θ, ζ]` are the
normalized flux surface average weights.
"""
function apply_collisions!(out::AbstractArray{<:Any,5}, C::CollisionOperator, f::AbstractArray{<:Any,5},
    surface_weights::AbstractMatrix)
    na, nt, nz, nx, ns = size(f)
    wL = C.weights[1]
    nr = nt * nz
    n = nx * ns

    # terms local in pitch angle: energy scattering and local field particle part
    F = reshape(f, na * nr, n)
    O = reshape(out, na * nr, n)
    mul!(O, F, transpose(C.Klocal), true, true)

    # pitch angle scattering; D annihilates constants, removing the pitch average
    # first keeps rounding proportional to the anisotropic part
    G = reshape(f, na, nr * n)
    O2 = reshape(out, na, nr, n)
    D = C.D
    band = maximum(abs(a - b) for b in 1:na, a in 1:na if D[a, b] != 0; init=0)
    g = Vector{eltype(f)}(undef, na)
    @inbounds for j in 1:n
        scale = -wL * C.nus[j] / 2
        for r in 1:nr
            column = r + nr * (j - 1)
            mean = zero(eltype(f))
            for a in 1:na
                mean += G[a, column]
            end
            mean /= na
            for a in 1:na
                g[a] = G[a, column] - mean
            end
            for a in 1:na
                acc = zero(eltype(f))
                for b in max(1, a - band):min(na, a + band)
                    acc += D[a, b] * g[b]
                end
                O2[a, r, j] += scale * acc
            end
        end
    end

    # Rosenbluth potential part, diagonal in Legendre mode
    Fm = reshape(C.Txi_inv * G, C.nL, nr, n)
    Y = similar(Fm, eltype(out))
    @inbounds for l in 1:C.nL
        @views mul!(Y[l, :, :], Fm[l, :, :], transpose(C.KGH[:, :, l]))
    end
    mul!(reshape(out, na, nr * n), C.Txi, reshape(Y, C.nL, nr * n), true, true)

    # surface averaged exchange between species
    if !isempty(C.exchange)
        c = vec(vec(surface_weights)' * @view(Fm[1, :, :]))
        d = C.Kexchange * c
        @inbounds for j in 1:n
            @views O2[:, :, j] .+= C.Txi[:, 1] .* d[j]
        end
    end
    return out
end
