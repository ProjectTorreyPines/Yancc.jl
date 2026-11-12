# Collisionless trajectories of the drift kinetic equation (yancc/_trajectories.py;
# Conlin & Landreman 2026, eqs 5-8, with ξ = -cos α).
#
#   dθ/dt = v ξ B^θ/B - E_ρ B_ζ/(B² √g)
#   dζ/dt = v ξ B^ζ/B + E_ρ B_θ/(B² √g)
#   dα/dt = -v sinα b·∇B/(2B) - E_ρ ξ sinα (B×∇ρ·∇B)/(2B³)
#   dx/dt = -E_ρ (1+ξ²) x (B×∇ρ·∇B)/(2B³)
#
# Each is discretized with upwinded finite differences, the direction chosen node by
# node from the sign of the advection speed.

"Rows of a finite difference matrix in compressed form."
struct RowStencil
    ptr::Vector{Int}
    idx::Vector{Int}
    val::Vector{Float64}
end

function RowStencil(M::AbstractMatrix)
    ptr, idx, val = Int[1], Int[], Float64[]
    for i in axes(M, 1)
        for j in axes(M, 2)
            if M[i, j] != 0
                push!(idx, j)
                push!(val, M[i, j])
            end
        end
        push!(ptr, length(idx) + 1)
    end
    return RowStencil(ptr, idx, val)
end

"""
    Trajectories(field, pitchgrid, speedgrid, species, Erho; p1, p1_theta, p1_zeta, p1_alpha)

Advection speeds `wt, wz, wa, wx[α, θ, ζ, x, s]` of the four phase space coordinates
and the forward/backward difference stencils in θ, ζ and α. `Erho = -∂Φ/∂ρ` [V].
`p1` names the first derivative stencil; `p1_theta`, `p1_zeta` and `p1_alpha` override
it for one coordinate.
"""
struct Trajectories{T}
    wt::Array{T,5}
    wz::Array{T,5}
    wa::Array{T,5}
    wx::Array{T,5}
    has_speed::Bool
    Dt::NTuple{2,RowStencil}   # (forward, backward)
    Dz::NTuple{2,RowStencil}
    Da::NTuple{2,RowStencil}
    Dx::Matrix{Float64}
end

function Trajectories(field::Field, pitchgrid::UniformPitchAngleGrid, speedgrid::MaxwellSpeedGrid,
    species::AbstractVector{<:LocalMaxwellian{T}}, Erho::Real; p1::AbstractString=DEFAULT_P1,
    p1_theta::AbstractString=p1, p1_zeta::AbstractString=p1, p1_alpha::AbstractString=p1) where {T}

    na, nt, nz, nx, ns = pitchgrid.nalpha, field.ntheta, field.nzeta, speedgrid.nx, length(species)
    half(p) = length(FD_COEFFS_1[p]) ÷ 2
    @assert nt > half(p1_theta) "ntheta must exceed $(half(p1_theta)) for stencil $p1_theta"
    @assert na > half(p1_alpha) "nalpha must exceed $(half(p1_alpha)) for stencil $p1_alpha"
    @assert nz == 1 || nz > half(p1_zeta) "nzeta must be 1 or exceed $(half(p1_zeta)) for stencil $p1_zeta"

    S = promote_type(T, typeof(float(Erho)))
    wt, wz, wa, wx = (zeros(S, na, nt, nz, nx, ns) for _ in 1:4)
    B = field.Bmag
    for s in 1:ns, i in 1:nx, z in 1:nz, t in 1:nt, a in 1:na
        v = speedgrid.x[i] * species[s].v_thermal
        xi = pitchgrid.xi[a]
        sina = sqrt(1 - xi^2)
        exb = -Erho / (B[t, z]^2 * field.sqrtg[t, z])
        drift = -Erho * field.BxgradrhodotgradB[t, z] / (2 * B[t, z]^3)
        wt[a, t, z, i, s] = field.B_sup_t[t, z] / B[t, z] * v * xi + field.B_sub_z[t, z] * exb
        wz[a, t, z, i, s] = field.B_sup_z[t, z] / B[t, z] * v * xi - field.B_sub_t[t, z] * exb
        wa[a, t, z, i, s] = -field.bdotgradB[t, z] / (2 * B[t, z]) * sina * v + xi * sina * drift
        wx[a, t, z, i, s] = (1 + xi^2) * speedgrid.x[i] * drift
    end

    ht, hz, ha = 2π / nt, 2π / nz / field.NFP, π / na
    stencils(p, n, h, bc) = (RowStencil(fdfwd_matrix(p, n, h, bc)), RowStencil(fdbwd_matrix(p, n, h, bc)))
    Dt = stencils(p1_theta, nt, ht, :periodic)
    Dz = nz > 1 ? stencils(p1_zeta, nz, hz, :periodic) : (RowStencil(zeros(1, 1)), RowStencil(zeros(1, 1)))
    Da = stencils(p1_alpha, na, ha, :symmetric)
    return Trajectories{S}(wt, wz, wa, wx, !iszero(Erho), Dt, Dz, Da, speedgrid.Dx)
end

# out += weight * w * (upwinded derivative of f along dimension dim)
function _advect!(out::AbstractArray{<:Any,5}, f::AbstractArray{<:Any,5}, w::AbstractArray{<:Any,5},
    D::NTuple{2,RowStencil}, dim::Int, weight::Real)
    weight == 0 && return out
    fwd, bwd = D
    stride = prod(size(f)[1:(dim-1)]; init=1)
    R = CartesianIndices(f)
    L = LinearIndices(f)
    @inbounds for I in R
        wind = w[I]
        st = wind > 0 ? bwd : fwd
        i = I[dim]
        base = L[I] - (i - 1) * stride
        acc = zero(eltype(out))
        for k in st.ptr[i]:(st.ptr[i+1]-1)
            acc += st.val[k] * f[base+(st.idx[k]-1)*stride]
        end
        out[I] += weight * wind * acc
    end
    return out
end

"""
    apply_trajectories!(out, tr, f, weights)

Add the advection terms to `out`; `weights` scale the `(x, α, θ, ζ)` terms.
"""
function apply_trajectories!(out::AbstractArray{<:Any,5}, tr::Trajectories, f::AbstractArray{<:Any,5}, weights)
    na, nt, nz, nx, ns = size(f)
    _advect!(out, f, tr.wa, tr.Da, 1, weights[2])
    _advect!(out, f, tr.wt, tr.Dt, 2, weights[3])
    nz > 1 && _advect!(out, f, tr.wz, tr.Dz, 3, weights[4])
    if tr.has_speed && weights[1] != 0
        F = reshape(f, na * nt * nz, nx, ns)
        O = reshape(out, na * nt * nz, nx, ns)
        W = reshape(tr.wx, na * nt * nz, nx, ns)
        @inbounds for s in 1:ns
            @views O[:, :, s] .+= weights[1] .* W[:, :, s] .* (F[:, :, s] * transpose(tr.Dx))
        end
    end
    return out
end
