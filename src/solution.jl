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
  - `momentum_flux[s]`: `⟨∫d³v m v∥ B f v_m·∇ρ⟩`
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
    vm = radial_magnetic_drift(op)
    dr = op.surface_weights
    x, wx, xi, wxi = op.speedgrid.x, op.speedgrid.wx, op.pitchgrid.xi, op.pitchgrid.wxi
    B = op.field.Bmag

    particle_flux, heat_flux, momentum_flux, VparB = (zeros(S, ns) for _ in 1:4)
    Vpar = zeros(S, nt, nz, ns)
    for (s, sp) in enumerate(op.species), i in 1:nx
        vth, m = sp.v_thermal, sp.species.mass
        v = x[i] * vth
        F0 = sp(v)
        for z in 1:nz, t in 1:nt, a in 1:na
            d3v = 2π * vth^3 * wxi[a] * x[i]^2 * wx[i]
            f = F0 + f1[a, t, z, i, s]
            g = f * vm[a, t, z, i, s] * d3v * dr[t, z]
            particle_flux[s] += g
            heat_flux[s] += m * v^2 / 2 * g
            momentum_flux[s] += m * v * xi[a] * B[t, z] * g
            Vpar[t, z, s] += v * xi[a] * f * d3v / sp.density
        end
    end
    for s in 1:ns
        VparB[s] = sum(B[t, z] * Vpar[t, z, s] * dr[t, z] for t in 1:nt, z in 1:nz)
    end
    JparB = sum(sp.density * sp.species.charge * VparB[s] for (s, sp) in enumerate(op.species))
    J_rho = sum(sp.species.charge * particle_flux[s] for (s, sp) in enumerate(op.species))
    particle_source = S[y[N+2s-1] for s in 1:ns]
    heat_source = S[y[N+2s] for s in 1:ns]
    return DKESolution{S}(f1, particle_flux, heat_flux, momentum_flux, Vpar, VparB, JparB, J_rho,
        particle_source, heat_source, iterations, residual)
end

"Conductive heat flux `heat_flux - 5/2 T particle_flux` of each species [W m⁻³]."
conductive_heat_flux(op::DKEOperator, sol::DKESolution) =
    [sol.heat_flux[s] - 5 / 2 * sp.temperature * ELEMENTARY_CHARGE * sol.particle_flux[s] for (s, sp) in enumerate(op.species)]
