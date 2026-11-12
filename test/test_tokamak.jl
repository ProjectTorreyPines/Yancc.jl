# Solver for axisymmetric fields: against the dense solve, the Python code at working
# resolution, SFINCS, and the symmetries of an axisymmetric field.

rel(a, b) = maximum(abs.(a .- b) ./ abs.(b))

@testset "GMRES" begin
    n = 40
    A = [1 / (1 + abs(i - j)) + (i == j ? 3.0 : 0.0) + 0.3 * sin(i * j) for i in 1:n, j in 1:n]
    b = [cos(0.3 * i) for i in 1:n]
    Dinv = 1 ./ diag(A)
    A!(y, v) = mul!(y, A, v)
    M!(z, v) = (z .= Dinv .* v)
    for restart in (60, 7)
        x, iterations, residual = Yancc.gmres!(zeros(n), A!, b, M!; rtol=1e-12, restart, maxiter=400)
        @test x ≈ A \ b rtol = 1e-10
        @test residual ≈ norm(b - A * x) / norm(b) atol = 1e-13
        @test iterations <= (restart == 60 ? n : 400)
    end
    # initial guess at the solution
    x, iterations, _ = Yancc.gmres!(A \ b, A!, b, M!; rtol=1e-10)
    @test iterations == 0
end

@testset "plane matrices" begin
    d = golden("tokamak_4z")
    op, _ = golden_operator(d; p1="4z")
    na, nt, nz, nx, ns = Yancc.state_size(op)
    # the plane problem is the operator without the terms coupling speeds and species,
    # up to the diagonal of those terms
    w = (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    opl = DKEOperator(op.field, op.pitchgrid, op.speedgrid, op.species, op.Erho; background=op.background, lnlambda=17.0,
        p1="4z", operator_weights=w)
    f = pystate(test_vector(prod(Yancc.state_size(op))), Yancc.state_size(op))
    out = similar(f)
    Yancc.apply!(out, opl, f)
    for s in 1:ns, i in (1, nx)
        j = i + nx * (s - 1)
        P = Yancc.plane_matrix(op, i, s)
        fp = vec(f[:, :, 1, i, s])
        @test P * fp - op.collisions.Klocal[j, j] .* fp ≈ vec(out[:, :, 1, i, s]) rtol = 1e-10
    end
end

# circular flux surface of a large aspect ratio tokamak, B = B0 / (1 + ε cos θ)
function circular_field(ntheta, eps; R0=3.0, q=2.0, B0=1.0, rho=0.5)
    a = eps * R0 / rho
    theta = range(0, 2π; length=ntheta + 1)[1:ntheta]
    Bmag = reshape(B0 ./ (1 .+ eps .* cos.(theta)), ntheta, 1)
    return field_from_boozer(rho, Bmag, 0.0, B0 * R0, 1 / q, π * a^2 * B0; R_major=R0, a_minor=a)
end

@testset "moment coarse space: $name" for name in ("tokamak", "tokamak_er", "tokamak_4z"), nl in (2, 3)
    # the assembled matrix is the projection of the bordered operator
    d = golden(name)
    op, _ = golden_operator(d; golden_stencil(name)...)
    border = Yancc.DKEBorder(op)
    na, nt, nz, nx, ns = Yancc.state_size(op)
    n, N = nx * ns, na * nt * nx * ns
    E, P, W = Yancc.moment_coarse_matrix(op, border, nl)
    nm = nl * n * nt
    c = cos.(1:size(E, 1))
    cm = reshape(view(c, 1:nm), n, nl, nt)
    c[(nm+1):(nm+n)] .= cm[:, 1, :] * vec(op.surface_weights)
    v = zeros(N + 2ns)
    V = reshape(view(v, 1:N), na, nt, n)
    for j in 1:n, t in 1:nt
        V[:, t, j] .= P * cm[j, :, t]
    end
    v[(N+1):end] .= c[(nm+n+1):end]
    y = Yancc.apply_bordered!(zeros(N + 2ns), op, border, v)
    Y = reshape(view(y, 1:N), na, nt, n)
    ref = zeros(size(E, 1))
    rm = reshape(view(ref, 1:nm), n, nl, nt)
    for j in 1:n, t in 1:nt
        rm[j, :, t] .= W * Y[:, t, j]
    end
    ref[(nm+n+1):end] .= y[(N+1):end]
    out = E * c
    @test out[1:nm] ≈ ref[1:nm] rtol = 1e-12
    @test maximum(abs, out[(nm+1):(nm+n)]) < 1e-12 * maximum(abs, c)
    @test out[(nm+n+1):end] ≈ ref[(nm+n+1):end] rtol = 1e-12
end

@testset "collisionality" begin
    # deuterium, carbon and electrons from the banana regime (ν*ₑ ≈ 0.1) deep into the
    # Pfirsch-Schlüter regime (ν*ₑ ≈ 10³), where rounding limits the residual and the
    # dense solve alike
    field = circular_field(11, 0.3)
    cache = TokamakCache()
    for (density, tol, tolflow) in ((1e19, 1e-6, 1e-6), (1e21, 1e-6, 1e-6), (1e23, 1e-4, 1e-2))
        species = [from_scale_lengths(Deuterium, 1e3, 0.7 * density, 3.0, 1.0),
            from_scale_lengths(Species(12, 6), 1e3, 0.05 * density, 3.0, 1.0),
            from_scale_lengths(Electron, 1e3, density, 3.0, 1.0)]
        op = DKEOperator(field, UniformPitchAngleGrid(17), MaxwellSpeedGrid(4), species; lnlambda=17.0, p1="4z")
        ref = solve_dke_reference(op)
        sol = solve_dke_tokamak(op; rtol=1e-10)
        @test sol.iterations < 40
        @test sol.residual < (density < 1e22 ? 1e-10 : 1e-6)
        @test sol.heat_flux ≈ ref.heat_flux rtol = tol
        @test sol.particle_flux ≈ ref.particle_flux rtol = tol
        @test sol.VparB ≈ ref.VparB rtol = tolflow
        # a cache from a very different plasma: the preconditioner is rebuilt when needed
        cached = solve_dke_tokamak!(cache, op; rtol=1e-8)
        @test cached.heat_flux ≈ ref.heat_flux rtol = tol
        if density < 1e22
            other = solve_dke_tokamak(op; rtol=1e-10, preconditioner=TokamakPreconditioner(op; coarse=:constants))
            @test other.heat_flux ≈ ref.heat_flux rtol = 1e-6
            @test other.iterations > sol.iterations || density < 1e20
            @test_logs (:warn, r"relative residual") solve_dke_tokamak(op; rtol=1e-10, maxiter=2)
        end
    end
    @test cache.solves == 3
    @test 1 <= cache.rebuilds <= 3
    @test cache.iterations >= cache.fresh_iterations > 0
end

@testset "analytic limits of a single species, ν* = $nustar" for nustar in (1e-2, 1e0, 1e2)
    # Large aspect ratio circular tokamak, ε = 0.1. A density gradient drives no
    # fluxes and the diamagnetic parallel flow; the particle flux vanishes; the heat
    # flux follows Chang & Hinton, Phys. Fluids 25, 1493 (1982). The speed grid has to
    # grow with collisionality for the first three to hold.
    eps, R0, q, T, lnlambda = 0.1, 3.0, 2.0, 1e3, 17.0
    e, m = Yancc.ELEMENTARY_CHARGE, Deuterium.mass
    TJ = T * e
    field = circular_field(41, eps; R0, q)
    density = nustar * T^2 * eps^1.5 / (q * R0 * 4.90e-18 * lnlambda)
    vth = sqrt(2TJ / m)
    tau = 3 * sqrt(2π) * vth^3 / (4 * e^4 * lnlambda / (4π * Yancc.EPSILON_0^2 * m^2) * density)
    K = field.G / (field.iota * field.psi_rho)
    result = map(((1.0, 0.0), (0.0, 1.0))) do (aLT, aLn)
        op = DKEOperator(field, UniformPitchAngleGrid(65), MaxwellSpeedGrid(nustar > 10 ? 10 : 6), [from_scale_lengths(Deuterium, T, density, aLT, aLn)];
            lnlambda, p1="4z")
        sol = solve_dke_tokamak(op; rtol=1e-9)
        (flux=sol.particle_flux[1] * TJ, heat=conductive_heat_flux(op, sol)[1], flow=sol.VparB[1] / (K * TJ / e), sol.iterations)
    end
    dT, dn = result
    @test dT.iterations < 40 && dn.iterations < 40
    @test abs(dT.flux) < 1e-3 * abs(dT.heat)
    @test abs(dn.flux) < 1e-3 * abs(dT.heat)
    @test abs(dn.heat) < 1e-3 * abs(dT.heat)
    @test dn.flow ≈ 1 rtol = 1e-3

    theta = range(0, 2π; length=4097)[1:4096]
    B = 1 ./ (1 .+ eps .* cos.(theta))
    average(g) = sum(g.(B) ./ B .^ 2) / sum(1 ./ B .^ 2)
    B2, Bm2 = average(b -> b^2), average(b -> b^-2)
    mu = q * R0 / (vth * tau * eps^1.5)
    banana = (0.66 + 1.88sqrt(eps) - 1.54eps) / (1 + 1.03sqrt(mu) + 0.31mu) * Bm2
    pfirsch = 25 / (4 * 3.906) * 0.74 * mu * eps^1.5 / (1 + 0.74 * mu * eps^1.5) * (Bm2 - 1 / B2) / (2sqrt(eps))
    chang_hinton = 2sqrt(eps) * (banana + pfirsch) * density * TJ^2 * K^2 * m / (e^2 * tau)
    @test dT.heat ≈ chang_hinton rtol = 0.08
end

@testset "tokamak solver against the dense solve: $name" for name in ("tokamak", "tokamak_er", "tokamak_4z")
    d = golden(name)
    op, EparB = golden_operator(d; golden_stencil(name)...)
    ref = solve_dke_reference(op; EparB)
    sol = solve_dke_tokamak(op; EparB, rtol=1e-11)
    @test sol.residual < 1e-11
    @test 0 < sol.iterations < 60
    for k in (:particle_flux, :heat_flux, :VparB)
        @test getfield(sol, k) ≈ getfield(ref, k) rtol = 1e-7
    end
    @test sol.JparB ≈ ref.JparB rtol = 1e-7
    @test all(relerr(sol.f1[:, :, :, :, s], ref.f1[:, :, :, :, s]) < 1e-6 for s in eachindex(op.species))

    # restart from the solution and with a preconditioner built for a nearby plasma
    again = solve_dke_tokamak(op; EparB, rtol=1e-8, guess=sol)
    @test again.iterations == 0
    near = [LocalMaxwellian(sp.species, 1.05 * sp.temperature, 0.97 * sp.density, sp.dTdrho, sp.dndrho) for sp in op.species]
    opn = DKEOperator(op.field, op.pitchgrid, op.speedgrid, near, op.Erho; background=op.background, lnlambda=17.0, golden_stencil(name)...)
    lagged = solve_dke_tokamak(op; EparB, rtol=1e-11, preconditioner=TokamakPreconditioner(opn))
    @test lagged.heat_flux ≈ ref.heat_flux rtol = 1e-7
    @test lagged.iterations < 2 * sol.iterations + 10

    # cache: the second solve starts from the first, a changed plasma reuses the factors
    cache = TokamakCache()
    first = solve_dke_tokamak!(cache, op; EparB, rtol=1e-9)
    @test first.iterations == cache.fresh_iterations > 0
    @test solve_dke_tokamak!(cache, op; EparB, rtol=1e-8).iterations == 0
    pre = cache.preconditioner
    moved = solve_dke_tokamak!(cache, opn; EparB, rtol=1e-9)
    @test cache.preconditioner === pre
    @test moved.heat_flux ≈ solve_dke_reference(opn; EparB).heat_flux rtol = 1e-6
    @test moved.iterations <= first.iterations + 5
end

@testset "axisymmetric field" begin
    d = golden("tokamak")
    op, _ = golden_operator(d)
    f = op.field
    ref = solve_dke_reference(op)

    # a toroidally resolved grid gives the same fluxes
    rep(a) = repeat(a, 1, 5)
    f5 = Field(f.rho, rep(f.B_sup_t), rep(f.B_sup_z), rep(f.B_sub_t), rep(f.B_sub_z), rep(f.Bmag), rep(f.sqrtg);
        Psi=f.Psi, iota=f.iota, R_major=f.R_major, a_minor=f.a_minor, B0=f.B0, dBdt=rep(f.dBdt), dBdz=zeros(f.ntheta, 5))
    op5 = DKEOperator(f5, op.pitchgrid, op.speedgrid, op.species[1:1], 200.0; background=op.species[2:end], lnlambda=17.0)
    op1 = DKEOperator(f, op.pitchgrid, op.speedgrid, op.species[1:1], 200.0; background=op.species[2:end], lnlambda=17.0)
    s5, s1 = solve_dke_reference(op5), solve_dke_reference(op1)
    @test s5.heat_flux ≈ s1.heat_flux rtol = 1e-8
    @test s5.particle_flux ≈ s1.particle_flux rtol = 1e-8
    @test s5.VparB ≈ s1.VparB rtol = 1e-8

    # reversing the toroidal field or the plasma current leaves the radial fluxes
    # unchanged and reverses the flow along B; reversing both restores it
    for (st, sz) in ((1, -1), (-1, 1), (-1, -1))
        fr = Field(f.rho, st .* f.B_sup_t, sz .* f.B_sup_z, st .* f.B_sub_t, sz .* f.B_sub_z, f.Bmag, f.sqrtg;
            Psi=sz * f.Psi, iota=st * sz * f.iota, R_major=f.R_major, a_minor=f.a_minor, B0=f.B0, dBdt=f.dBdt)
        opr = DKEOperator(fr, op.pitchgrid, op.speedgrid, op.species, 0.0; background=op.background, lnlambda=17.0)
        sr = solve_dke_reference(opr)
        @test sr.heat_flux ≈ ref.heat_flux rtol = 1e-7
        @test sr.particle_flux ≈ ref.particle_flux rtol = 1e-6
        @test sr.VparB ≈ (st * sz) .* ref.VparB rtol = 1e-7
    end
end

@testset "Python at working resolution" begin
    d = golden("production")
    op, _ = golden_operator(d)
    sol = solve_dke_tokamak(op; rtol=1e-10)
    @test sol.particle_flux ≈ d["particle_flux"] rtol = 1e-5
    @test sol.heat_flux ≈ d["heat_flux"] rtol = 1e-6
    @test sol.VparB ≈ d["VparB"] rtol = 1e-6
    @test sol.JparB ≈ d["JparB"] rtol = 1e-6
end

# Radial scan of the DSHAPE tokamak, electrons and hydrogen, with a radial electric
# field (yancc's test_solve_dke_dshape_2species; SFINCS with ntheta=71, nxi=160, nx=24).
@testset "DSHAPE tokamak against SFINCS, rho = $(d["rho"][ir+1])" for d in (golden("sfincs_dshape"),), ir in (15, 17)
    p = "ir$(ir)_"
    field = golden_field(d, p * "field_")
    sp = golden_species(d, p)
    Erho = d[p*"Erho"]

    # resolution and stencil of the Python test
    op = DKEOperator(field, UniformPitchAngleGrid(81), MaxwellSpeedGrid(12), sp, Erho; lnlambda=17.0)
    sol = solve_dke_tokamak(op; rtol=1e-9)
    for k in (:particle_flux, :heat_flux, :VparB)
        @test getfield(sol, k) ≈ d[p*"yancc_"*string(k)] rtol = 1e-5
        @test getfield(sol, k) ≈ d[p*"sfincs_"*string(k)] rtol = 2e-2
    end
    @test sol.JparB ≈ d[p*"yancc_JparB"] rtol = 1e-5
    @test sol.JparB ≈ d[p*"sfincs_JparB"] rtol = 2e-2

    # centred stencils on a coarser velocity grid
    op = DKEOperator(field, UniformPitchAngleGrid(65), MaxwellSpeedGrid(8), sp, Erho; lnlambda=17.0, p1="4z")
    sol = solve_dke_tokamak(op; rtol=1e-8)
    for k in (:particle_flux, :heat_flux, :VparB)
        @test getfield(sol, k) ≈ d[p*"sfincs_"*string(k)] rtol = 2e-2
    end
    @test sol.JparB ≈ d[p*"sfincs_JparB"] rtol = 2e-2
end

@testset "README example" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    code = match(r"```julia\n(using Yancc\n.*?)```"s, readme).captures[1]
    m = Module()
    Base.include_string(m, code)
    sol = m.sol
    @test sol.residual < 1e-6
    @test all(sol.heat_flux .> 0)
    @test sol.JparB > 0
    # electron flux against the ion fluxes: ambipolar to discretization error
    Z = [1, 6, -1]
    @test abs(sum(Z .* sol.particle_flux)) < 0.2 * sum(abs, Z .* sol.particle_flux)
end

@testset "finite differences through the cache" begin
    # the fluxes returned for small changes of the inputs are a smooth function of them
    field = circular_field(15, 0.3)
    plasma(aLT, T) = [from_scale_lengths(Deuterium, T, 5e19, aLT, 1.0), from_scale_lengths(Electron, T, 5e19, aLT, 1.0)]
    operator(aLT, T) = DKEOperator(field, UniformPitchAngleGrid(25), MaxwellSpeedGrid(4), plasma(aLT, T); lnlambda=17.0, p1="4z")
    exact(aLT, T) = solve_dke_reference(operator(aLT, T)).heat_flux[1]
    cache = TokamakCache()
    base = solve_dke_tokamak!(cache, operator(3.0, 2e3); rtol=1e-6)
    @test solve_dke_tokamak!(cache, operator(3.0, 2e3); rtol=1e-6).heat_flux == base.heat_flux
    @test cache.solves == 2 && cache.rebuilds == 1 && cache.iterations == base.iterations
    for (derivative, h, shifted) in (
        ((exact(3.0 + 1e-4, 2e3) - exact(3.0 - 1e-4, 2e3)) / 2e-4, 3e-8, h -> operator(3.0 + h, 2e3)),
        ((exact(3.0, 2e3 + 1e-1) - exact(3.0, 2e3 - 1e-1)) / 2e-1, 2e-5, h -> operator(3.0, 2e3 + h)))
        plus = solve_dke_tokamak!(cache, shifted(h); rtol=1e-6)
        @test (plus.heat_flux[1] - base.heat_flux[1]) / h ≈ derivative rtol = 2e-3
    end
    @test cache.rebuilds == 1
    @test abs(base.heat_flux[1] - exact(3.0, 2e3)) < 1e-4 * abs(base.heat_flux[1])
    # a larger change is a full solve again
    far = solve_dke_tokamak!(cache, operator(3.3, 2e3); rtol=1e-6)
    @test far.heat_flux[1] ≈ exact(3.3, 2e3) rtol = 1e-4
    @test far.iterations > 0
end
