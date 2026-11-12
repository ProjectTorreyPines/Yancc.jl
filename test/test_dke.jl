# Operator, drive and dense solution against the Python code, term by term.

# stencil of each reference case
golden_stencil(name) = name == "tokamak_4z" ? (; p1="4z") : (;)

function golden_operator(d; kw...)
    nx, na, nL = Int.(d["params"][1:3])
    ll = d["params"][4] < 0 ? nothing : d["params"][4]
    Erho, EparB = d["params"][5], d["params"][6]
    field = golden_field(d)
    sp = golden_species(d)
    bg = haskey(d, "bg_mass") ? golden_species(d, "bg_") : LocalMaxwellian{Float64}[]
    op = DKEOperator(field, UniformPitchAngleGrid(na), MaxwellSpeedGrid(nx), sp, Erho; background=bg, nL, lnlambda=ll, kw...)
    return op, EparB
end

@testset "field $name" for name in ("dke3d", "tokamak")
    d = golden(name)
    f = golden_field(d; derivatives=false)
    scale = maximum(abs.(d["field_Bmag"]))
    @test f.dBdt ≈ d["field_dBdt"] atol = 1e-11 * scale
    @test f.dBdz ≈ d["field_dBdz"] atol = 1e-11 * scale * f.NFP
    for k in (:bdotgradB, :BxgradrhodotgradB)
        @test getfield(f, k) ≈ d["field_"*string(k)] atol = 1e-10 * maximum(abs.(d["field_"*string(k)]))
    end
    for k in (:Bmag_fsa, :B2mag_fsa, :I, :G, :psi_rho, :psi_r)
        @test getfield(f, k) ≈ d["field_"*string(k)] rtol = 1e-12
    end
    # Boozer constructor reproduces the stored arrays
    fb = field_from_boozer(f.rho, f.Bmag, f.I, f.G, f.iota, f.Psi; NFP=f.NFP)
    for k in (:B_sup_t, :B_sup_z, :B_sub_t, :B_sub_z, :sqrtg)
        @test getfield(fb, k) ≈ getfield(f, k) rtol = 1e-12
    end
    # iota and Psi recovered from the field when not given
    fr = Field(f.rho, f.B_sup_t, f.B_sup_z, f.B_sub_t, f.B_sub_z, f.Bmag, f.sqrtg; NFP=f.NFP)
    @test fr.iota ≈ f.iota rtol = 1e-10
    @test fr.Psi ≈ f.Psi rtol = 1e-10
end

@testset "operator terms $name" for name in ("dke3d", "dke3d_er", "tokamak", "tokamak_er", "tokamak_4z")
    d = golden(name)
    op, EparB = golden_operator(d; golden_stencil(name)...)
    dims = Yancc.state_size(op)
    N = prod(dims)
    f = pystate(test_vector(N), dims)
    ref(k) = pystate(d["mv_"*k], dims)

    function term(weights)
        out = zeros(dims)
        Yancc.apply_trajectories!(out, op.trajectories, f, weights)
        return out
    end
    @test relerr(term((0, 1, 0, 0)), ref("opa")) < 1e-11
    @test relerr(term((0, 0, 1, 0)), ref("opt")) < 1e-11
    if dims[3] > 1
        @test relerr(term((0, 0, 0, 1)), ref("opz")) < 1e-11
    else
        @test all(iszero, d["mv_opz"])
    end
    if op.Erho != 0
        @test relerr(term((1, 0, 0, 0)), ref("opx")) < 1e-10
    else
        @test all(iszero, d["mv_opx"])
    end

    bg = op.background
    nL = op.collisions.nL
    ll = d["params"][4] < 0 ? nothing : d["params"][4]
    for (k, w) in (("CL", (1, 0, 0)), ("CE", (0, 1, 0)), ("CF", (0, 0, 1)))
        C = Yancc.CollisionOperator(op.pitchgrid, op.speedgrid, op.species; background=bg, nL, lnlambda=ll, weights=w,
            energy_exchange=true)
        out = zeros(dims)
        Yancc.apply_collisions!(out, C, f, op.surface_weights)
        r = ref(k)
        # species differ by orders of magnitude; compare each on its own scale
        @test all(relerr(out[:, :, :, :, s], r[:, :, :, :, s]) < 1e-8 for s in 1:dims[5])
    end
    out = zeros(dims)
    Yancc.apply_collisions!(out, op.collisions, f, op.surface_weights)
    @test all(relerr(out[:, :, :, :, s], ref("C")[:, :, :, :, s]) < 1e-8 for s in 1:dims[5])

    Yancc.apply!(out, op, f)
    @test all(relerr(out[:, :, :, :, s], ref("dke")[:, :, :, :, s]) < 1e-8 for s in 1:dims[5])

    rhs = Yancc.dke_rhs(op, EparB)
    @test all(relerr(rhs[:, :, :, :, s], pystate(d["rhs"], dims)[:, :, :, :, s]) < 1e-11 for s in 1:dims[5])
end

@testset "border and dense solve $name" for name in ("dke3d", "tokamak")
    d = golden(name)
    op, EparB = golden_operator(d)
    dims = Yancc.state_size(op)
    ns = dims[5]
    N = prod(dims)
    border = Yancc.DKEBorder(op)
    n = Yancc.system_size(op)

    # sources: columns of B; constraints: rows of C applied to the test state
    v = zeros(n)
    y = zeros(n)
    for j in 1:(2ns)
        fill!(v, 0)
        v[N+j] = 1
        Yancc.apply_bordered!(y, op, border, v)
        r = pystate(d["B"][:, j], dims)
        @test relerr(reshape(y[1:N], dims), r) < 1e-12
    end
    tv = test_vector(N)
    fill!(v, 0)
    v[1:N] .= vec(pystate(tv, dims))
    Yancc.apply_bordered!(y, op, border, v)
    @test y[(N+1):end] ≈ d["C"] * tv rtol = 1e-9

    # the Python solution satisfies the Julia system
    K = Yancc.bordered_matrix(op, border)
    b = zeros(n)
    b[1:N] .= vec(Yancc.dke_rhs(op, EparB))
    ypy = vcat(vec(pystate(d["f1"], dims)), d["particle_source"], d["heat_source"])
    ypy[(N+1):end] .= vec(permutedims(hcat(d["particle_source"], d["heat_source"])))
    @test norm(K * ypy - b) < 1e-6 * norm(b)

    sol = solve_dke_reference(op; EparB)
    for (k, val) in (("particle_flux", sol.particle_flux), ("heat_flux", sol.heat_flux), ("VparB", sol.VparB))
        @test val ≈ d[k] rtol = 1e-8
    end
    # sources and the momentum flux vanish to rounding without a radial electric field;
    # compare on the scale of the drive and of the particle flux
    rhs = Yancc.dke_rhs(op, EparB)
    for (s, sp) in enumerate(op.species)
        drive = maximum(abs.(rhs[:, :, :, :, s])) / sp(sp.v_thermal)
        @test sol.particle_source[s] ≈ d["particle_source"][s] atol = 1e-8 * drive
        @test sol.heat_source[s] ≈ d["heat_source"][s] atol = 1e-8 * drive
        mom = sp.species.mass * sp.v_thermal * op.field.B0 * abs(d["particle_flux"][s])
        @test sol.momentum_flux[s] ≈ d["momentum_flux"][s] atol = 1e-8 * mom
    end
    @test sol.JparB ≈ d["JparB"] rtol = 1e-8
    @test abs(sol.J_rho - d["J_rho"]) < 1e-6 * sum(abs.(d["particle_flux"])) * Yancc.ELEMENTARY_CHARGE
    f1py = pystate(d["f1"], dims)
    @test all(relerr(sol.f1[:, :, :, :, s], f1py[:, :, :, :, s]) < 1e-5 for s in 1:ns)
end

@testset "fluxes with Erho, EparB: $name" for name in ("dke3d_er", "tokamak_er", "tokamak_4z")
    d = golden(name)
    op, EparB = golden_operator(d; golden_stencil(name)...)
    sol = solve_dke_reference(op; EparB)
    for (k, val) in (("particle_flux", sol.particle_flux), ("heat_flux", sol.heat_flux), ("VparB", sol.VparB),
        ("momentum_flux", sol.momentum_flux), ("particle_source", sol.particle_source), ("heat_source", sol.heat_source))
        @test val ≈ d[k] rtol = 1e-6
    end
    @test sol.JparB ≈ d["JparB"] rtol = 1e-8
end

@testset "DSHAPE tokamak" begin
    d = golden("dshape")
    field = golden_field(d, "rho0.5_field_")
    # the VMEC derivative of |B| and the spectral one agree on this smooth surface
    @test golden_field(d, "rho0.5_field_"; derivatives=false).dBdt ≈ field.dBdt rtol = 1e-3 atol = 1e-4
    sp = golden_species(d, "axi_")
    nx, na, nL = Int.(d["axi_params"][1:3])
    op = DKEOperator(field, UniformPitchAngleGrid(na), MaxwellSpeedGrid(nx), sp, 0.0; nL, lnlambda=d["axi_params"][4])
    sol = solve_dke_reference(op)
    @test sol.particle_flux ≈ d["axi_particle_flux"] rtol = 1e-6 atol = 1e-8 * abs(d["axi_heat_flux"][1]) / (sp[1].temperature * Yancc.ELEMENTARY_CHARGE)
    @test sol.heat_flux ≈ d["axi_heat_flux"] rtol = 1e-6
    @test sol.VparB ≈ d["axi_VparB"] rtol = 1e-6
    @test sol.heat_flux[1] > 0
end
