# Sonic toroidal rotation (src/rotation.jl): checks that need no reference code, then NEO
# (golden/rotation.npz: the problems as Yancc poses them and NEO's results in SI).

using ForwardDiff

const Carbon = Species(12, 6)
const E_CHARGE = Yancc.ELEMENTARY_CHARGE

# circular Miller surface r/a = 0.5, R/a = 3, q = 2 with a = 1 m, B_unit = 2 T
rotating_field(ntheta; r=0.5, ipccw=-1, btccw=-1) = miller_field(; r, a=1.0, R0=3.0, q=2.0, Bunit=2.0, ipccw, btccw, ntheta)

# D + C + e at 2 keV; scale lengths in ρ: 1/L_n = 1, 1/L_T = 3
function rotating_species(; T=2e3, ne=5e19, nC=0.0167 * 5e19, dlnn=(1.0, 1.0, 1.0), dlnT=(3.0, 3.0, 3.0))
    return [from_scale_lengths(Deuterium, T, ne - 6nC, dlnT[1], dlnn[1]), from_scale_lengths(Carbon, T, nC, dlnT[2], dlnn[2]),
        from_scale_lengths(Electron, T, ne, dlnT[3], dlnn[3])]
end

# angular velocity at which deuterium at temperature T [eV] has the Mach number `mach` at R = 3.5 m
mach_omega(mach; T=2e3) = mach * sqrt(2 * T * E_CHARGE / Deuterium.mass) / 3.5

function rotating_operator(; ntheta=13, nalpha=17, nx=4, omega=mach_omega(0.5), shear=0.0, species=rotating_species(), kw...)
    return DKEOperator(rotating_field(ntheta), UniformPitchAngleGrid(nalpha), MaxwellSpeedGrid(nx), species, 0.0;
        lnlambda=16.0, p1="4z", rotation=Rotation(omega, shear), kw...)
end

@testset "rotation: equilibrium" begin
    field = rotating_field(16)
    R, Rref = vec(field.R), field.R[1]
    omega, T = mach_omega(0.8), 2e3
    mT(sp) = sp.mass / (T * E_CHARGE)

    # D + e: e Φ/T = (m_i - m_e) ω² (R² - R*²)/(4T)
    sp = [from_scale_lengths(Deuterium, T, 5e19, 3.0, 1.0), from_scale_lengths(Electron, T, 5e19, 3.0, 1.0)]
    eq = Yancc.RotatingEquilibrium(field, sp, [], Rotation(omega, 0.0))
    phi = (mT(Deuterium) - mT(Electron)) .* omega^2 .* (R .^ 2 .- Rref^2) ./ 4
    @test eq.Phi ./ T ≈ phi rtol = 1e-10
    @test eq.density[:, 1] ≈ eq.density[:, 2] rtol = 1e-12
    @test eq.density[:, 1] ≈ exp.((mT(Deuterium) + mT(Electron)) .* omega^2 .* (R .^ 2 .- Rref^2) ./ 4) rtol = 1e-10
    @test eq.Phi[1] == 0 && all(eq.lambda[1, :] .== 0)

    # trace tungsten in that plasma, as a kinetic and as a background species
    W = Species(184, 40)
    trace = from_scale_lengths(W, T, 5e9, 3.0, 1.0)
    lambda = 40 .* phi .- mT(W) .* omega^2 .* (R .^ 2 .- Rref^2) ./ 2
    eq3 = Yancc.RotatingEquilibrium(field, [sp; trace], [], Rotation(omega, 0.0))
    @test eq3.lambda[:, 3] ≈ lambda rtol = 1e-7
    @test eq3.density[:, 3] ≈ exp.(-lambda) rtol = 1e-6
    @test minimum(eq3.density[:, 3]) < 1e-3   # pushed to the outboard side
    eqb = Yancc.RotatingEquilibrium(field, sp, [trace], Rotation(omega, 0.0))
    @test eqb.bg_density[:, 1] ≈ eq3.density[:, 3] rtol = 1e-12

    # D + C + e: quasineutral at every node, and the derivative of the potential energy
    sp = rotating_species()
    eq = Yancc.RotatingEquilibrium(field, sp, [], Rotation(omega, 0.0))
    Z = [s.species.charge / E_CHARGE for s in sp]
    n = [s.density for s in sp]
    @test maximum(abs, eq.density * (Z .* n)) < 1e-10 * n[3]
    @test eq.dlambda ≈ Yancc.spectral_derivative_matrix(16) * eq.lambda rtol = 1e-8
    @test eq.lambda[:, 2] ≈ 6 .* eq.Phi ./ T .- mT(Carbon) .* omega^2 .* (R .^ 2 .- Rref^2) ./ 2 rtol = 1e-12

    # gradient of the potential energy at the reference point against a difference of two surfaces
    shear, h = -0.7 * omega, 1e-5
    eq = Yancc.RotatingEquilibrium(field, sp, [], Rotation(omega, shear))
    energy(d) = -(omega + shear * d)^2 * rotating_field(16; r=0.5 + d).R[1]^2 / 2
    @test eq.dlnN ≈ [mT(s.species) for s in sp] .* (energy(h) - energy(-h)) / 2h rtol = 1e-8

    # dual numbers through the Newton iteration
    dn = ForwardDiff.derivative(w -> Yancc.RotatingEquilibrium(field, sp, [], Rotation(w, 0.0)).density[9, 2], omega)
    fd = (Yancc.RotatingEquilibrium(field, sp, [], Rotation(omega * (1 + 1e-6), 0.0)).density[9, 2] -
          Yancc.RotatingEquilibrium(field, sp, [], Rotation(omega * (1 - 1e-6), 0.0)).density[9, 2]) / (2e-6 * omega)
    @test dn ≈ fd rtol = 1e-7
end

@testset "rotation: none" begin
    # ω = 0 is the operator without rotation
    sp = rotating_species()
    field = rotating_field(9)
    grids = (UniformPitchAngleGrid(11), MaxwellSpeedGrid(4))
    op0 = DKEOperator(field, grids..., sp, 0.0; lnlambda=16.0, p1="4z")
    opr = DKEOperator(field, grids..., sp, 0.0; lnlambda=16.0, p1="4z", rotation=Rotation(0.0, 0.0))
    f = reshape(cos.(1:prod(Yancc.state_size(op0))), Yancc.state_size(op0))
    out0, outr = similar(f), similar(f)
    Yancc.apply!(out0, op0, f)
    Yancc.apply!(outr, opr, f)
    @test outr ≈ out0 rtol = 1e-12
    @test Yancc.dke_rhs(opr) ≈ Yancc.dke_rhs(op0) rtol = 1e-13
    s0, sr = solve_dke_tokamak(op0; rtol=1e-10), solve_dke_tokamak(opr; rtol=1e-10)
    @test sr.particle_flux ≈ s0.particle_flux rtol = 1e-7
    @test sr.heat_flux ≈ s0.heat_flux rtol = 1e-7
    @test sr.JparB ≈ s0.JparB rtol = 1e-7
    # a field without a major radius cannot rotate
    @test_throws AssertionError DKEOperator(circular_field(9, 0.15), grids..., sp, 0.0; rotation=Rotation(1e4, 0.0))
end

@testset "rotation: collisionless operator" begin
    # L F = 0 and L[E F] = 0 for the Maxwellian of the local density, and the drifts are
    # L of the canonical angular momentum and of its square (Sugama & Horton 1997, eq 11)
    collisionless = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    errors = map((17, 33)) do ntheta
        op = rotating_operator(; ntheta, nalpha=33, nx=5, omega=mach_omega(0.7), shear=0.0, operator_weights=collisionless)
        na, nt, nz, nx, ns = Yancc.state_size(op)
        eq, fd = op.rotation, op.field
        F, EF, g1, g2 = (zeros(na, nt, 1, nx, ns) for _ in 1:4)
        scale = zeros(1, 1, 1, 1, ns)   # size of the streaming term v_th F/(q R)
        for (s, sp) in enumerate(op.species), i in 1:nx, t in 1:nt, a in 1:na
            x, xi = op.speedgrid.x[i], op.pitchgrid.xi[a]
            v = x * sp.v_thermal
            B, I, R = fd.Bmag[t, 1], fd.B_sub_z[t, 1], fd.R[t, 1]
            F[a, t, 1, i, s] = sp(v) * eq.density[t, s]
            EF[a, t, 1, i, s] = (x^2 + eq.lambda[t, s]) * F[a, t, 1, i, s]
            g1[a, t, 1, i, s] = I * v * xi / B + eq.omega * R^2
            g2[a, t, 1, i, s] = g1[a, t, 1, i, s]^2 + v^2 * (1 - xi^2) * (R^2 * B^2 - I^2) / (2B^2)
            scale[1, 1, 1, 1, s] = sp.v_thermal * sp(0.0) / (2 * 3.0)
        end
        L(f) = Yancc.apply!(similar(f), op, f)
        err(a, b) = maximum(abs.(a .- b) ./ scale)
        psi1 = fd.sqrtg[1, 1] * fd.B_sup_t[1, 1]   # dψ/dρ
        drift, momentum = Yancc.radial_drift(op), Yancc.angular_momentum_drift(op)
        mq = reshape([sp.species.mass / sp.species.charge for sp in op.species], 1, 1, 1, 1, ns)
        m = reshape([sp.species.mass for sp in op.species], 1, 1, 1, 1, ns)
        # in units of a typical value of each right hand side
        rho_scale = maximum(abs, drift; dims=(1, 2, 3, 4))
        pi_scale = maximum(abs, momentum; dims=(1, 2, 3, 4))
        (err(L(F), 0 .* F), err(L(EF), 0 .* F), err(L(F .* g1) .* mq ./ psi1 ./ rho_scale, F .* drift ./ rho_scale),
            err(L(F .* g2) .* m .* mq ./ (2 * psi1) ./ pi_scale, F .* momentum ./ pi_scale))
    end
    @test all(errors[2] .< 2e-3)
    @test all(errors[2] .< errors[1] ./ 4)   # converging with the poloidal resolution
end

@testset "rotation: drive" begin
    # rigid rotation with flux functions N and T that have no gradient: no drive
    omega = mach_omega(0.6)
    op0 = rotating_operator(; omega)
    dlnn = Tuple(op0.rotation.dlnN)   # -dln(n*)/dρ that makes dN/dρ = 0
    op = rotating_operator(; omega, species=rotating_species(; dlnn, dlnT=(0.0, 0.0, 0.0)))
    @test maximum(abs, Yancc.dke_rhs(op)) <= 1e-12 * maximum(abs, Yancc.dke_rhs(op0))
    @test all(d -> d < 0, dlnn)   # the density at the outboard midplane rises with the major radius there
end

# transport matrix: fluxes (Γ_s, Q_s/T_s, Σ_s Π_s/T_s) per unit of each force
# (X1_s, X2_s, dω/dρ) with X1 = A1 + dlnN, X2 = A2, the forces of the drive
function transport_matrix(; omega, kw...)
    ns = 3
    base = rotating_operator(; omega, kw...)
    dlnN = base.rotation.dlnN
    nforce = 2ns + 1
    fluxes, forces = zeros(nforce, nforce), zeros(nforce, nforce)
    for j in 1:nforce
        dlnn, dlnT = zeros(ns), zeros(ns)
        j <= ns && (dlnn[j] = 1.0)
        ns < j <= 2ns && (dlnT[j-ns] = 1.0)
        shear = j == nforce ? omega : 0.0
        op = rotating_operator(; omega, shear, species=rotating_species(; dlnn=Tuple(dlnn), dlnT=Tuple(dlnT)), kw...)
        sol = solve_dke_tokamak(op; rtol=1e-11)
        TJ = [sp.temperature * E_CHARGE for sp in op.species]
        fluxes[:, j] .= [sol.particle_flux; sol.heat_flux ./ TJ; sum(sol.angular_momentum_flux ./ TJ)]
        A = Yancc.thermodynamic_forces(op)
        forces[:, j] .= [[A[s][1] + op.rotation.dlnN[s] for s in 1:ns]; [A[s][2] for s in 1:ns]; shear]
    end
    return fluxes / forces
end

@testset "rotation: Onsager symmetry and ambipolarity" begin
    # Sugama & Horton 1997, eqs 24-25: on an up-down symmetric surface the transport matrix
    # is symmetric; the matrix of the gradients and the viscosity are even in ω, the coupling
    # of the gradients to the rotation shear is odd in ω.
    omega = mach_omega(0.5)
    Z = [1.0, 6.0, -1.0]
    # The discrete matrix approaches this with the number of speeds: the particle flux of
    # deuterium is the small remainder left by momentum conservation.
    results = map((4, 6)) do nx
        L = transport_matrix(; omega, ntheta=13, nalpha=17, nx)
        scale = sqrt.(abs.(diag(L)))
        (maximum(abs.(L .- L') ./ (scale .* scale')), maximum(abs, Z' * L[1:3, :]) / maximum(abs, Z .* L[1:3, :]), L)
    end
    @test results[2][1] < 0.03 && results[2][1] < results[1][1] / 3   # symmetric
    @test results[2][2] < 0.01 && results[2][2] < results[1][2] / 3   # ambipolar for every force
    L = results[2][3]
    @test all(diag(L) .< 0)   # fluxes down the gradients: positive entropy production
    @test maximum(abs, L[1:6, 7]) > 0.05 * sqrt(abs(L[1, 1] * L[7, 7]))   # the coupling is not small
    Lm = transport_matrix(; omega=-omega, ntheta=13, nalpha=17, nx=6)
    @test Lm[1:6, 1:6] ≈ L[1:6, 1:6] rtol = 1e-6
    @test Lm[1:6, 7] ≈ -L[1:6, 7] rtol = 1e-6
    @test Lm[7, 1:6] ≈ -L[7, 1:6] rtol = 1e-6
    @test Lm[7, 7] ≈ L[7, 7] rtol = 1e-6
end

@testset "rotation: field directions" begin
    # the fluxes of particles and energy do not depend on the directions of the field and of
    # the rotation; the flux of angular momentum follows the rotation and its shear
    omega, shear = mach_omega(0.5), -2 * mach_omega(0.5)
    function fluxes(ipccw, btccw, sign)
        op = DKEOperator(rotating_field(13; ipccw, btccw), UniformPitchAngleGrid(17), MaxwellSpeedGrid(4), rotating_species(), 0.0;
            lnlambda=16.0, p1="4z", rotation=Rotation(sign * omega, sign * shear))
        sol = solve_dke_tokamak(op; rtol=1e-10)
        return sol.particle_flux, sol.heat_flux, sol.angular_momentum_flux
    end
    ref = fluxes(-1, -1, 1)
    for (ipccw, btccw, sign) in ((1, -1, 1), (-1, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 1, -1))
        f = fluxes(ipccw, btccw, sign)
        @test f[1] ≈ ref[1] rtol = 1e-6
        @test f[2] ≈ ref[2] rtol = 1e-6
        @test f[3] ≈ sign .* ref[3] rtol = 1e-6
    end
end

@testset "rotation: tokamak solver" begin
    # against the dense solve, with the exchange of energy between species removed and kept
    for energy_exchange in (false, true), (omega, shear) in ((mach_omega(0.5), -mach_omega(1.0)), (mach_omega(1.5), 0.0))
        op = rotating_operator(; ntheta=9, nalpha=11, nx=4, omega, shear, energy_exchange)
        ref = solve_dke_reference(op)
        sol = solve_dke_tokamak(op; rtol=1e-10)
        @test sol.residual <= 1e-10
        @test sol.iterations < 60
        @test sol.particle_flux ≈ ref.particle_flux rtol = 1e-6
        @test sol.heat_flux ≈ ref.heat_flux rtol = 1e-6
        @test sol.angular_momentum_flux ≈ ref.angular_momentum_flux rtol = 1e-6
        @test sol.JparB ≈ ref.JparB rtol = 1e-6
    end
    # the coarse matrix is the projection of the bordered operator on the moments
    op = rotating_operator(; ntheta=9, nalpha=11, nx=4, omega=mach_omega(1.0), shear=0.0)
    border = Yancc.DKEBorder(op)
    na, nt, nz, nx, ns = Yancc.state_size(op)
    n, N, nl = nx * ns, na * nt * nx * ns, 3
    E, P, W = Yancc.moment_coarse_matrix(op, border, nl)
    rot = op.collisions.rotating
    nm, nk = nl * n * nt, size(rot.V, 2)
    c = cos.(1:size(E, 1))
    cm = reshape(view(c, 1:nm), n, nl, nt)
    sw = vec(op.surface_weights)
    c[(nm+1):(nm+nk)] .= sum(sw[t] .* (rot.M[:, :, t] * cm[:, 1, t]) for t in 1:nt)
    c[(nm+nk+1):(nm+2nk)] .= sum(sw[t] .* (rot.MK[:, :, t] * cm[:, 1, t]) for t in 1:nt)
    v = zeros(N + 2ns)
    V = reshape(view(v, 1:N), na, nt, n)
    for j in 1:n, t in 1:nt
        V[:, t, j] .= P * cm[j, :, t]
    end
    v[(N+1):end] .= c[(nm+2nk+1):end]
    y = Yancc.apply_bordered!(zeros(N + 2ns), op, border, v)
    Y = reshape(view(y, 1:N), na, nt, n)
    ref = zeros(size(E, 1))
    rm = reshape(view(ref, 1:nm), n, nl, nt)
    for j in 1:n, t in 1:nt
        rm[j, :, t] .= W * Y[:, t, j]
    end
    ref[(nm+2nk+1):end] .= y[(N+1):end]
    out = E * c
    @test out[1:nm] ≈ ref[1:nm] rtol = 1e-10
    @test maximum(abs, out[(nm+1):(nm+2nk)]) < 1e-10 * maximum(abs, c)
    @test out[(nm+2nk+1):end] ≈ ref[(nm+2nk+1):end] rtol = 1e-10
end

@testset "rotation: derivatives" begin
    # ForwardDiff in ω, dω/dρ and a temperature: the dense solve against finite differences,
    # the tokamak solver against the dense solve
    p0 = [mach_omega(0.6), -mach_omega(0.9), 2e3]
    function operator(p)
        sp = rotating_species()
        sp = [sp[1], from_scale_lengths(Carbon, p[3], sp[2].density, 3.0, 1.0), sp[3]]
        return DKEOperator(rotating_field(9), UniformPitchAngleGrid(11), MaxwellSpeedGrid(3), sp, 0.0; lnlambda=16.0, p1="4z",
            rotation=Rotation(p[1], p[2]))
    end
    outputs(sol) = [sol.particle_flux; sol.heat_flux; sol.angular_momentum_flux; sol.JparB]
    dense(p) = outputs(solve_dke_reference(operator(p)))
    tokamak(p) = outputs(solve_dke_tokamak(operator(p); rtol=1e-10, tangent_rtol=1e-9))
    J = ForwardDiff.jacobian(dense, p0)
    Jfd = similar(J)
    for k in eachindex(p0)
        e = zeros(length(p0))
        e[k] = 1e-5 * abs(p0[k])
        Jfd[:, k] .= (dense(p0 + e) .- dense(p0 - e)) ./ 2e[k]
    end
    @test column_error(J, Jfd) < 1e-4
    @test column_error(ForwardDiff.jacobian(tokamak, p0), J) < 2e-5
end

# surface without up-down symmetry: an ellipse with triangularity whose centre rises with r
tilted_field(ntheta; s_zmag=0.3, btccw=-1) = miller_field(; r=0.5, a=1.0, R0=3.0, q=2.0, Bunit=2.0, kappa=1.5, delta=0.2, s_zmag,
    ipccw=-1, btccw, ntheta)

@testset "rotation: nondissipative fluxes" begin
    omega = mach_omega(0.5)
    tilted_operator(ntheta, nalpha, nx; kw...) = DKEOperator(tilted_field(ntheta; kw...), UniformPitchAngleGrid(nalpha), MaxwellSpeedGrid(nx),
        rotating_species(), 0.0; lnlambda=16.0, p1="4z", rotation=Rotation(omega, -2omega))
    op = tilted_operator(21, 33, 6)
    H = nondissipative_fluxes(op)
    sol = solve_dke_tokamak(op; rtol=1e-10)
    charge = [1, 6, -1]
    ambipolarity(G) = abs(sum(charge .* G)) / sum(abs, charge .* G)

    # they vanish with up-down symmetry and without rotation
    symmetric = nondissipative_fluxes(rotating_operator(; shear=-2omega))
    @test maximum(abs, symmetric.particle_flux) < 1e-10 * maximum(abs, H.particle_flux)
    @test maximum(abs, symmetric.heat_flux) < 1e-10 * maximum(abs, H.heat_flux)
    @test maximum(abs, symmetric.angular_momentum_flux) < 1e-10 * maximum(abs, H.angular_momentum_flux)
    norot = DKEOperator(tilted_field(13), UniformPitchAngleGrid(9), MaxwellSpeedGrid(3), rotating_species(), 0.0; lnlambda=16.0)
    @test all(iszero, nondissipative_fluxes(norot).particle_flux) && all(iszero, nondissipative_fluxes(norot).angular_momentum_flux)

    # moments of the Maxwellian: no dependence on the velocity grids, spectral convergence in θ
    coarse = nondissipative_fluxes(tilted_operator(15, 9, 3))
    @test coarse.particle_flux ≈ H.particle_flux rtol = 1e-4
    @test coarse.heat_flux ≈ H.heat_flux rtol = 1e-4
    @test coarse.angular_momentum_flux ≈ H.angular_momentum_flux rtol = 1e-4

    # Sugama & Horton 1997, eq 28: the particle fluxes are ambipolar only with them. The fluxes of the
    # kinetic equation alone are far from ambipolar here; the rest is the discretization error
    # (0.13, 0.07, 0.012 at 13×17×4, 15×25×5, 21×33×6)
    @test ambipolarity(sol.particle_flux) > 0.5
    @test ambipolarity(sol.particle_flux .+ H.particle_flux) < 0.02
    low = tilted_operator(13, 17, 4)
    @test ambipolarity(solve_dke_tokamak(low; rtol=1e-10).particle_flux .+ nondissipative_fluxes(low).particle_flux) >
          3 * ambipolarity(sol.particle_flux .+ H.particle_flux)

    # no rotation shear: no particle and energy fluxes; no gradients and no rotation: no momentum flux
    rigid = nondissipative_fluxes(DKEOperator(tilted_field(21), UniformPitchAngleGrid(9), MaxwellSpeedGrid(3), rotating_species(), 0.0;
        lnlambda=16.0, rotation=Rotation(omega, 0.0)))
    @test all(iszero, rigid.particle_flux) && all(iszero, rigid.heat_flux)
    @test maximum(abs, rigid.angular_momentum_flux) > 0.1 * maximum(abs, H.angular_momentum_flux)
    flat = rotating_species(; dlnn=(0.0, 0.0, 0.0), dlnT=(0.0, 0.0, 0.0))
    still = nondissipative_fluxes(DKEOperator(tilted_field(21), UniformPitchAngleGrid(9), MaxwellSpeedGrid(3), flat, 0.0; lnlambda=16.0,
        rotation=Rotation(0.0, 0.0)))
    @test maximum(abs, still.angular_momentum_flux) < 1e-10 * maximum(abs, H.angular_momentum_flux)

    # the surface turned upside down is the mirror image: the toroidal field reverses, the current
    # and the rotation do not, and no flux changes
    mirror = tilted_operator(21, 33, 6; s_zmag=-0.3, btccw=1)
    Hm = nondissipative_fluxes(mirror)
    @test Hm.particle_flux ≈ H.particle_flux rtol = 1e-9
    @test Hm.heat_flux ≈ H.heat_flux rtol = 1e-9
    @test Hm.angular_momentum_flux ≈ H.angular_momentum_flux rtol = 1e-9
    solm = solve_dke_tokamak(mirror; rtol=1e-10)
    @test solm.particle_flux ≈ sol.particle_flux rtol = 1e-6
    @test solm.angular_momentum_flux ≈ sol.angular_momentum_flux rtol = 1e-6
end

# NEO (E. A. Belli and J. Candy) run as a black box with its full Fokker-Planck operator and sonic
# rotation. A case of golden/rotation.npz is the field (with the major radius of its nodes), the
# species, the Coulomb logarithm and the rotation under a prefix, and NEO's result in SI: fluxes
# per unit ρ as `[species (D, C, e), (particle, energy, angular momentum)]`, with the energy flux
# of `energy_flux` and the angular momentum flux along +ζ of the field.
rotation_field(d, prefix) = golden_field(d, prefix * "field_"; derivatives=false)
rotation_of(d, prefix) = Rotation(d[prefix*"omega"], d[prefix*"domega_drho"])
rotation_cases(d, set) = sort!([k[(length(set)+2):(end-6)] for k in keys(d) if startswith(k, set * "_") && endswith(k, "_omega")])

@testset "rotation: equilibrium against NEO" begin
    d = golden("rotation")
    # potential and densities on the surface at NEO's poloidal nodes (NaN at the nodes between)
    function equilibrium_error(case)
        p = "equilibrium_" * case * "_"
        sp = golden_species(d, p)
        eq = RotatingEquilibrium(rotation_field(d, p), sp, LocalMaxwellian{Float64}[], rotation_of(d, p))
        nodes = findall(!isnan, d[p*"Phi"])
        Te, ne = sp[end].temperature, sp[end].density
        dphi = maximum(abs, eq.Phi[nodes] .- eq.Phi[1] .- d[p*"Phi"][nodes]) / Te
        dn = maximum(abs, eq.density[nodes, :] ./ eq.density[1:1, :] ./ d[p*"density_ratio"][nodes, :] .- 1)
        defect = abs(sum(s.species.charge * s.density for s in sp)) / (E_CHARGE * ne)
        return dphi, dn, defect, maximum(abs, d[p*"Phi"][nodes]) / Te
    end
    # quasineutral inputs: agreement to the digits NEO prints, up to a carbon Mach number of 2.4
    for case in ("circular_rigid1", "circular_rigid3", "circular_shear")
        dphi, dn, defect, scale = equilibrium_error(case)
        @test defect < 1e-12
        @test dphi < 2e-6 * scale
        @test dn < 2e-6
    end
    # The MAST-U inputs are not neutral (Σ Z n = -0.6e-4 … -3e-4 n_e). NEO restores neutrality with a potential
    # at the reference point too; here the densities given there are kept and the charge density of
    # the reference point is kept on the surface. The two differ by the order of the defect.
    for case in ("mastu_r045_rot", "mastu_r045_double", "mastu_r085_rot", "mastu_r025_rot")
        dphi, dn, defect, scale = equilibrium_error(case)
        @test 1e-5 < defect < 1e-3
        @test dphi < 5 * defect
        @test dn < 15 * defect
    end
end

@testset "rotation: nondissipative fluxes against NEO" begin
    # surfaces without up-down symmetry: moments of the Maxwellian, so no resolution is involved;
    # NEO prints five digits. Differences: 2e-4 (circular), 8e-4 (MAST-U, whose input is not neutral)
    d = golden("rotation")
    cases = rotation_cases(d, "nondissipative")
    @test length(cases) == 7
    for case in cases
        p = "nondissipative_" * case * "_"
        sp = golden_species(d, p)
        op = DKEOperator(rotation_field(d, p), UniformPitchAngleGrid(9), MaxwellSpeedGrid(3), sp, 0.0; lnlambda=d[p*"lnlambda"],
            rotation=rotation_of(d, p))
        H = nondissipative_fluxes(op)
        E = energy_flux(op, H.particle_flux, H.heat_flux, H.angular_momentum_flux; frame=:laboratory)
        fluxes, ref = hcat(H.particle_flux, E, H.angular_momentum_flux), d[p*"fluxes"]
        @test maximum(abs, ref[:, 3]) > 0
        for k in axes(ref, 1), j in 1:3
            # NaN: not printed by NEO (particle and energy fluxes without rotation)
            isnan(ref[k, j]) && continue
            if ref[k, j] != 0
                @test fluxes[k, j] ≈ ref[k, j] rtol = 2e-3
            else
                # no particle flux without rotation shear
                @test abs(fluxes[k, j]) * sp[k].temperature * E_CHARGE < 1e-9 * maximum(abs, ref[:, 2])
            end
        end
    end
end

@testset "rotation: fluxes against NEO" begin
    d = golden("rotation")
    # Yancc/NEO - 1, `[species (D, C, e), (Γ, Q, Π)]`: the fluxes of the kinetic equation with the
    # energy flux of the laboratory frame, or (`total`) with the nondissipative fluxes and the
    # energy flux of the rotating frame. The field of the case sets the poloidal resolution.
    function deviation(case; nx=6, nalpha=65, total=false)
        p = "fluxes_" * case * "_"
        op = DKEOperator(rotation_field(d, p), UniformPitchAngleGrid(nalpha), MaxwellSpeedGrid(nx), golden_species(d, p), 0.0;
            lnlambda=d[p*"lnlambda"], rotation=rotation_of(d, p))
        sol = solve_dke_tokamak(op; rtol=1e-8)
        G, Q, P = sol.particle_flux, sol.heat_flux, sol.angular_momentum_flux
        if total
            H = nondissipative_fluxes(op)
            G, Q, P = G .+ H.particle_flux, Q .+ H.heat_flux, P .+ H.angular_momentum_flux
        end
        E = energy_flux(op, G, Q, P; frame=total ? :rotating : :laboratory)
        return hcat(G, E, P) ./ d[p*(total ? "total_fluxes" : "fluxes")] .- 1
    end
    within(d, limits) = all(abs.(d) .< limits)

    # The tolerances are about twice the differences found at this resolution (6 speeds, 65 pitch
    # angles, 41 poloidal nodes). The electron particle flux, the small remainder of the ion
    # fluxes, is not asserted, nor are fluxes near a zero crossing.
    # MAST-U H-mode case (carbon Mach number above 1), mid radius
    dev = deviation("mastu_r045_rot")
    @test within(dev[1, :], [0.07, 0.03, 0.03])
    @test within(dev[2, :], [0.08, 0.05, 0.06])
    @test abs(dev[3, 2]) < 0.06
    # with the energy flux of the rotating frame
    t = deviation("mastu_r045_rot"; total=true)
    @test within(t[1, :], [0.07, 0.03, 0.03]) && abs(t[3, 2]) < 0.06
    # edge, where the shape matters
    dev = deviation("mastu_r085_rot")
    @test within(dev[1, :], [0.02, 0.01, 0.02])
    @test within(dev[2, :], [0.03, 0.02, 0.03])
    @test abs(dev[3, 2]) < 0.01
    dev = deviation("mastu_r065_rot")
    @test within(dev[1, :], [0.02, 0.01, 0.05])
    @test within(dev[2, [1, 3]], [0.01, 0.01])
    @test within(dev[3, 1:2], [0.015, 0.01])
    # signs: rotation reversed; field and current reversed; shear without rotation
    for case in ("mastu_r045_reversed", "mastu_r045_bothflip")
        dev = deviation(case)
        @test within(dev[1, :], [0.08, 0.05, 0.04]) && within(dev[2, :], [0.09, 0.06, 0.07])
    end
    dev = deviation("mastu_r045_shearonly")
    @test within(dev[1:2, 3], [0.04, 0.05])
    # strong rotation (carbon Mach number 2.4) with and without shear at high collisionality (33
    # poloidal nodes); the particle fluxes need 10 speeds there
    for case in ("collisional_rigid3", "collisional_shear")
        dev = deviation(case; nx=10, nalpha=49)
        @test within(dev[1, :], [0.03, 0.01, 0.02])
        @test within(dev[2, :], [0.05, 0.08, 0.05])
        @test within(dev[3, 1:2], [0.03, 0.01])
    end
    # surface without up-down symmetry: the fluxes of the kinetic equation, which are far from
    # ambipolar, and the total
    dev = deviation("updown_mastu_r045_rot")
    @test within(dev[1, :], [0.04, 0.03, 0.01])
    t = deviation("updown_mastu_r045_rot"; total=true)
    @test within(t[1, :], [0.09, 0.04, 0.05]) && within(t[2, :], [0.10, 0.05, 0.08])
end
