# Velocity space grids (yancc/velocity_grids.py).

# Coefficients of the monic three term recurrence
#   p_{k+1}(x) = (x - a_k) p_k(x) - b_k p_{k-1}(x),  p_0 = 1,
# of the Maxwell polynomials, orthogonal on [0, ∞) with weight exp(-x^2).
# b_0 is the integral of the weight. Tabulated in yancc for up to 21 terms.
const MAXWELL_AK = [
    5.6418958354775617e-01, 9.8842539284680075e-01, 1.2859676193639398e00, 1.5247208440801154e00,
    1.7301922743094393e00, 1.9134998431431025e00, 2.0806203364008327e00, 2.2352283805046405e00,
    2.3797824435046384e00, 2.5160256434438635e00, 2.6452479250569576e00, 2.7684359535042509e00,
    2.8863645940326998e00, 2.9996556533536003e00, 3.1088171759249232e00, 3.2142706360711242e00,
    3.3163702970830897e00, 3.4154173324133374e00, 3.5116703446156343e00, 3.6053533459055607e00,
    3.6966619115045933e00]
const MAXWELL_BK = [
    8.8622692545275805e-01, 1.8169011381620928e-01, 3.4132512895943917e-01, 5.0496215298800162e-01,
    6.7026419463961828e-01, 8.3617049928031195e-01, 1.0023478510110115e00, 1.1686711647442711e00,
    1.3350829222423350e00, 1.5015525993447623e00, 1.6680623621881157e00, 1.8346010527937686e00,
    2.0011613185512132e00, 2.1677381117632635e00, 2.3343278495405015e00, 2.5009279171337027e00,
    2.6675363609572034e00, 2.8341516916678309e00, 3.0007727537827202e00, 3.1673986369644243e00,
    3.3340286142031150e00]

const MAX_NX = length(MAXWELL_AK) - 1

"""
    maxwell_polynomials!(p, dp, z)

Values `p[k+1] = p_k(z)` and derivatives `dp[k+1] = p_k'(z)` of the monic Maxwell
polynomials for `k = 0:length(p)-1`.
"""
function maxwell_polynomials!(p::AbstractVector, dp::AbstractVector, z)
    n = length(p)
    p[1] = one(z)
    dp[1] = zero(z)
    n == 1 && return p, dp
    p[2] = z - MAXWELL_AK[1]
    dp[2] = one(z)
    @inbounds for k in 2:(n-1)
        p[k+1] = (z - MAXWELL_AK[k]) * p[k] - MAXWELL_BK[k] * p[k-1]
        dp[k+1] = p[k] + (z - MAXWELL_AK[k]) * dp[k] - MAXWELL_BK[k] * dp[k-1]
    end
    return p, dp
end

"Matrix of `p_k(z_i)` (columns `k = 0:n-1`) and of the derivatives `p_k'(z_i)`."
function maxwell_vandermonde(z::AbstractVector{T}, n::Integer) where {T}
    V = zeros(T, length(z), n)
    dV = zeros(T, length(z), n)
    p = zeros(T, n)
    dp = zeros(T, n)
    for (i, zi) in enumerate(z)
        maxwell_polynomials!(p, dp, zi)
        V[i, :] .= p
        dV[i, :] .= dp
    end
    return V, dV
end

"""
    MaxwellSpeedGrid(nx)

Grid in normalized speed `x = v/v_th`: the `nx` Gauss nodes of the Maxwell polynomials.
Functions are represented by their node values and include the weight, i.e.
`f(x) = exp(-x^2) p(x)` with `p` a polynomial of degree `nx - 1`.

  - `x`, `wx`: nodes and weights with `∫_0^∞ f dx ≈ sum(wx .* f.(x))`
  - `xvander[i, k]`: `exp(-x_i^2) p_{k-1}(x_i)`; `xvander_inv` its inverse (node values to
    polynomial coefficients)
  - `Dx`, `D2x`: first and second derivative matrices acting on node values
"""
struct MaxwellSpeedGrid
    nx::Int
    x::Vector{Float64}
    wx::Vector{Float64}
    xvander::Matrix{Float64}
    xvander_inv::Matrix{Float64}
    Dx::Matrix{Float64}
    D2x::Matrix{Float64}
end

function MaxwellSpeedGrid(nx::Integer)
    @assert 2 <= nx <= MAX_NX "MaxwellSpeedGrid requires 2 <= nx <= $MAX_NX"
    # Golub-Welsch: nodes are the eigenvalues of the Jacobi matrix, weights follow
    # from the first components of its eigenvectors
    J = SymTridiagonal(MAXWELL_AK[1:nx], sqrt.(MAXWELL_BK[2:nx]))
    E = eigen(J)
    x = E.values
    wgauss = MAXWELL_BK[1] .* abs2.(E.vectors[1, :])
    weight = exp.(-x .^ 2)
    wx = wgauss ./ weight

    Vp, _ = maxwell_vandermonde(x, nx)
    xvander = Vp .* weight
    xvander_inv = inv(xvander)

    # Differentiation of the polynomial interpolant through the nodes (barycentric
    # form), then d/dx (w p) = w (p' - 2 x p) and
    # d2/dx2 (w p) = w (p'' - 4 x p' + (4 x^2 - 2) p).
    c = [prod(x[i] - x[k] for k in 1:nx if k != i) for i in 1:nx]
    DL = zeros(nx, nx)
    for i in 1:nx
        for j in 1:nx
            i == j && continue
            DL[i, j] = c[i] / (c[j] * (x[i] - x[j]))
        end
        DL[i, i] = -sum(DL[i, j] for j in 1:nx if j != i)
    end
    X = Diagonal(x)
    W = Diagonal(weight)
    Wi = Diagonal(1 ./ weight)
    Dx = W * (DL - 2 * X) * Wi
    D2x = W * (DL * DL - 4 * X * DL + Diagonal(4 .* x .^ 2 .- 2)) * Wi
    return MaxwellSpeedGrid(nx, x, wx, xvander, xvander_inv, Dx, D2x)
end

"""
    UniformPitchAngleGrid(nalpha)

Uniform cell-centred grid in pitch angle `α ∈ (0, π)`, with `ξ = v∥/v = -cos(α)`.
`wxi` are the Fejér type 1 quadrature weights, `∫_{-1}^{1} f dξ ≈ sum(wxi .* f)`.
"""
struct UniformPitchAngleGrid
    nalpha::Int
    alpha::Vector{Float64}
    xi::Vector{Float64}
    wxi::Vector{Float64}
end

function UniformPitchAngleGrid(nalpha::Integer)
    alpha = [(2k + 1) * π / (2nalpha) for k in 0:(nalpha-1)]
    xi = -cos.(alpha)
    wxi = [2 / nalpha * (1 - 2 * sum(cos(2j * a) / (4j^2 - 1) for j in 1:(nalpha÷2); init=0.0)) for a in alpha]
    return UniformPitchAngleGrid(nalpha, alpha, xi, wxi)
end

"Matrix of Legendre polynomials `P_l(ξ_i)`, columns `l = 0:nL-1`."
function legendre_vandermonde(xi::AbstractVector{T}, nL::Integer) where {T}
    P = zeros(T, length(xi), nL)
    for (i, z) in enumerate(xi)
        P[i, 1] = one(T)
        nL > 1 && (P[i, 2] = z)
        for l in 2:(nL-1)
            P[i, l+1] = ((2l - 1) * z * P[i, l] - (l - 1) * P[i, l-1]) / l
        end
    end
    return P
end

"Gauss-Legendre nodes and weights on [0, 1]."
function gauss_legendre_unit(n::Integer)
    beta = [k / sqrt(4k^2 - 1) for k in 1:(n-1)]
    E = eigen(SymTridiagonal(zeros(n), beta))
    return (E.values .+ 1) ./ 2, abs2.(E.vectors[1, :])
end
