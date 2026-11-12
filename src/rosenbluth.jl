# Rosenbluth potentials of the speed basis functions (yancc/_collisions.py,
# RosenbluthPotentials; Conlin & Landreman 2026, appendix).
#
# For f(v) = P_l(ξ) L_k(x) exp(-x^2), with L_k the monic Maxwell polynomials, the
# potentials ∇²H = -4π f, ∇²G = 2H have the Legendre components (eqs 66-71)
#   H_l = 4π/(2l+1) [x^-(l+1) I2 + x^l I1]
#   G_l = -4π/(4l²-1) [x^l I3 - c x^(l+2) I1 - c x^-(l+1) I4 + x^-(l-1) I2],  c = (2l-1)/(2l+3)
#   I1 = ∫_x^∞ z^(1-l) f,  I2 = ∫_0^x z^(l+2) f,  I3 = ∫_x^∞ z^(3-l) f,  I4 = ∫_0^x z^(l+4) f
# in units of the thermal speed of the field species.

const GL_ORDER = 32
const GL_ZMAX = 10.0
const GL_LOWER_PANELS = 2
const GL_UPPER_GEOMETRIC_PANELS = 4
const GL_UPPER_UNIFORM_PANELS = 2
const GL_NODES, GL_WEIGHTS = gauss_legendre_unit(GL_ORDER)

function _panels!(z::Vector{T}, w::Vector{T}, edges) where {T}
    for i in 1:(length(edges)-1)
        lo, hi = edges[i], edges[i+1]
        for j in 1:GL_ORDER
            push!(z, lo + (hi - lo) * GL_NODES[j])
            push!(w, (hi - lo) * GL_WEIGHTS[j])
        end
    end
    return z, w
end

"Nodes and weights for integrals over [0, x] of a function decaying like exp(-z²)."
function lower_speed_quadrature(x::T) where {T}
    top = min(x, GL_ZMAX)
    edges = [top * i / GL_LOWER_PANELS for i in 0:GL_LOWER_PANELS]
    return _panels!(T[], T[], edges)
end

"""
Nodes and weights for integrals over [x, ∞) of a function decaying like exp(-z²).
Geometrically graded panels from `x` to `max(x, 1)` resolve negative powers of `z`
for small `x`; uniform panels cover the tail.
"""
function upper_speed_quadrature(x::T) where {T}
    mid = max(x, one(x))
    ratio = (mid / x)^(1 / GL_UPPER_GEOMETRIC_PANELS)
    geometric = [x * ratio^i for i in 0:GL_UPPER_GEOMETRIC_PANELS]
    far = mid + min(8.0, 22.0 / x + 1.0)
    uniform = [mid + (far - mid) * i / GL_UPPER_UNIFORM_PANELS for i in 0:GL_UPPER_UNIFORM_PANELS]
    z, w = _panels!(T[], T[], geometric)
    return _panels!(z, w, uniform)
end

"""
    rosenbluth_lk(x, nL, nx)

`H`, `dH/dx` and `d²G/dx²` at normalized speed `x` for Legendre modes `l = 0:nL-1` and
speed basis functions `k = 0:nx-1`, as `(nL, nx)` matrices.
"""
function rosenbluth_lk(x::T, nL::Integer, nx::Integer) where {T<:Real}
    zl, wl = lower_speed_quadrature(x)
    zu, wu = upper_speed_quadrature(x)
    Pl, _ = maxwell_vandermonde(zl, nx)
    Pu, _ = maxwell_vandermonde(zu, nx)
    el = wl .* exp.(-zl .^ 2)
    eu = wu .* exp.(-zu .^ 2)
    px, dpx = maxwell_polynomials!(zeros(T, nx), zeros(T, nx), x)
    ex = exp(-x^2)

    H = zeros(T, nL, nx)
    dH = zeros(T, nL, nx)
    ddG = zeros(T, nL, nx)
    for l in 0:(nL-1), k in 1:nx
        I1 = sum(eu[i] * zu[i]^(1 - l) * Pu[i, k] for i in eachindex(zu))
        I3 = sum(eu[i] * zu[i]^(3 - l) * Pu[i, k] for i in eachindex(zu))
        I2 = sum(el[i] * zl[i]^(l + 2) * Pl[i, k] for i in eachindex(zl))
        I4 = sum(el[i] * zl[i]^(l + 4) * Pl[i, k] for i in eachindex(zl))
        # integrand z^q L_k exp(-z²) and its derivative at z = x
        g(q) = x^q * px[k] * ex
        dg(q) = (q * x^(q - 1) * px[k] + x^q * dpx[k] - 2 * x^(q + 1) * px[k]) * ex
        dI1, dI2, dI3, dI4 = -g(1 - l), g(l + 2), -g(3 - l), g(l + 4)
        ddI1, ddI2, ddI3, ddI4 = -dg(1 - l), dg(l + 2), -dg(3 - l), dg(l + 4)

        H[l+1, k] = 4π / (2l + 1) * (x^(-(l + 1)) * I2 + x^l * I1)
        dH[l+1, k] = 4π / (2l + 1) * (-(l + 1) * x^(-(l + 2)) * I2 + x^(-(l + 1)) * dI2 + l * x^(l - 1) * I1 + x^l * dI1)
        c = (2l - 1) / (2l + 3)
        t1 = l * (l - 1) * x^(l - 2) * I3 + 2 * l * x^(l - 1) * dI3 + x^l * ddI3
        t2 = -c * ((l + 2) * (l + 1) * x^l * I1 + 2 * (l + 2) * x^(l + 1) * dI1 + x^(l + 2) * ddI1)
        t3 = -c * ((l + 1) * (l + 2) * x^(-(l + 3)) * I4 - 2 * (l + 1) * x^(-(l + 2)) * dI4 + x^(-(l + 1)) * ddI4)
        t4 = l * (l - 1) * x^(-(l + 1)) * I2 - 2 * (l - 1) * x^(-l) * dI2 + x^(-(l - 1)) * ddI2
        ddG[l+1, k] = -4π / (4l^2 - 1) * (t1 + t2 + t3 + t4)
    end
    return H, dH, ddG
end

"""
    RosenbluthPotentials(speedgrid, species, nL)

Potentials of every speed basis function of field species `b`, evaluated on the speed
grid of test species `a`: arrays `H`, `dH` (dH/dv) and `ddG` (d²G/dv²) indexed
`[x, l, k, a, b]`, in SI units per unit basis coefficient.
"""
struct RosenbluthPotentials{T}
    nL::Int
    H::Array{T,5}
    dH::Array{T,5}
    ddG::Array{T,5}
end

function RosenbluthPotentials(speedgrid::MaxwellSpeedGrid, species::AbstractVector{<:LocalMaxwellian{T}}, nL::Integer=8) where {T}
    ns, nx = length(species), speedgrid.nx
    H = zeros(T, nx, nL, nx, ns, ns)
    dH = zeros(T, nx, nL, nx, ns, ns)
    ddG = zeros(T, nx, nL, nx, ns, ns)
    for (b, spb) in enumerate(species), (a, spa) in enumerate(species)
        va, vb = spa.v_thermal, spb.v_thermal
        for i in 1:nx
            # speed of species a in units of the thermal speed of b
            h, dh, ddg = rosenbluth_lk(speedgrid.x[i] * va / vb, nL, nx)
            H[i, :, :, a, b] .= h .* vb^2
            dH[i, :, :, a, b] .= dh .* vb
            ddG[i, :, :, a, b] .= ddg .* vb^2
        end
    end
    return RosenbluthPotentials{T}(nL, H, dH, ddG)
end
