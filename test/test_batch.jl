# Batched solver on the host: against the single case operator and solver.

function batch_cases(; ntheta=11, nalpha=17, nx=4, Erho=(0.0, 0.0, 0.0, 0.0), energy_exchange=false, p1="4z")
    cases = ((0.30, 1e19, 1e3, 3.0), (0.20, 1e20, 2e3, 2.0), (0.10, 1e21, 5e2, 1.0), (0.25, 3e19, 1.5e3, 4.0))
    return [begin
        species = [from_scale_lengths(Deuterium, T, 0.7 * density, aLT, 1.0),
            from_scale_lengths(Species(12, 6), T, 0.05 * density, aLT, 1.0),
            from_scale_lengths(Electron, 1.1 * T, density, aLT, 1.0)]
        DKEOperator(circular_field(ntheta, eps; q=1.5 + k / 2), UniformPitchAngleGrid(nalpha), MaxwellSpeedGrid(nx), species, Erho[k];
            lnlambda=17.0, p1, energy_exchange)
    end for (k, (eps, density, T, aLT)) in enumerate(cases)]
end

@testset "batched GMRES" begin
    n, nb = 40, 5
    As = [[1 / (1 + abs(i - j)) + (i == j ? 3.0 + k : 0.0) + 0.3 * sin(i * j * k) for i in 1:n, j in 1:n] for k in 1:nb]
    B = [cos(0.3 * i * k) for i in 1:n, k in 1:nb]
    B[:, 4] .= 0
    A!(Y, V) = (for k in 1:nb; mul!(view(Y, :, k), As[k], view(V, :, k)); end; Y)
    M!(Z, V) = (for k in 1:nb; view(Z, :, k) .= view(V, :, k) ./ diag(As[k]); end; Z)
    for restart in (60, 7)
        X, iterations, residuals, status = Yancc.gmres_batched!(zeros(n, nb), A!, copy(B), M!; rtol=1e-12, restart, maxiter=400)
        for k in 1:nb
            @test X[:, k] ≈ As[k] \ B[:, k] rtol = 1e-10 atol = 1e-14
            @test status[k] == :converged
            # the same iteration as the single system solver
            _, single, _ = Yancc.gmres!(zeros(n), (y, v) -> mul!(y, As[k], v), B[:, k], (z, v) -> (z .= v ./ diag(As[k])); rtol=1e-12, restart, maxiter=400)
            @test iterations[k] == single
        end
        @test iterations[4] == 0 && residuals[4] == 0
    end
    # systems stop independently: a tight budget leaves the others unconverged only
    X, iterations, residuals, status = Yancc.gmres_batched!(zeros(n, nb), A!, copy(B), M!; rtol=1e-12, maxiter=3)
    @test all(iterations[k] == 3 && status[k] == :maxiter for k in (1, 2, 3, 5))
    @test status[4] == :converged
    # initial guess at the solutions
    X0 = reduce(hcat, [As[k] \ B[:, k] for k in 1:nb])
    _, iterations, _, _ = Yancc.gmres_batched!(X0, A!, copy(B), M!; rtol=1e-10)
    @test all(iterations .== 0)
    # a system whose preconditioner fails stops at once and leaves the others alone
    _, reference, _, _ = Yancc.gmres_batched!(zeros(n, nb), A!, copy(B), M!; rtol=1e-12)
    Mbad!(Z, V) = (M!(Z, V); Z[3, 2] = NaN; Z)
    X, iterations, residuals, status = Yancc.gmres_batched!(zeros(n, nb), A!, copy(B), Mbad!; rtol=1e-12)
    @test status == [:converged, :failed, :converged, :converged, :converged]
    @test iterations[2] <= 1 && isnan(residuals[2])
    @test iterations[[1, 3, 4, 5]] == reference[[1, 3, 4, 5]]
    @test all(X[:, k] ≈ As[k] \ B[:, k] for k in (1, 3, 5))
end

@testset "batched operator: Erho = $(Erho[2]), exchange = $energy_exchange" for (Erho, energy_exchange) in
                                                                               (((0.0, 0.0, 0.0, 0.0), false), ((0.0, 800.0, -1500.0, 300.0), false), ((0.0, 800.0, 0.0, 0.0), true))
    ops = batch_cases(; Erho, energy_exchange)
    borders = [Yancc.DKEBorder(op) for op in ops]
    bop = Yancc.BatchedDKE(ops, borders, Val(:cpu))
    M = Yancc.system_size(ops[1])
    X = reduce(hcat, [test_vector(M) .* (1 + 0.1 * k) .+ 0.01 * k for k in eachindex(ops)])
    Y = Yancc.apply_bordered!(similar(X), bop, X)
    for (k, op) in enumerate(ops)
        y = Yancc.apply_bordered!(zeros(M), op, borders[k], X[:, k])
        @test Y[:, k] ≈ y rtol = 1e-12
    end
    # the matrices of the preconditioner, assembled together from the batched operator
    maps = Yancc.PreconditionerMaps(ops[1], 4, bop.has_speed, Val(:cpu))
    planes, coarse = Yancc._assemble_preconditioner(maps, bop)
    na, nt, nz, nx, ns = Yancc.state_size(ops[1])
    close(A, B) = norm(A - B, Inf) <= 1e-12 * norm(B, Inf)
    for (k, op) in enumerate(ops)
        @test close(Yancc._matrix(coarse, k), first(Yancc.moment_coarse_matrix(op, borders[k], 4)))
        @test all(close(Yancc._matrix(planes, i + nx * (s - 1) + nx * ns * (k - 1)), Yancc.plane_matrix(op, i, s)) for s in 1:ns, i in 1:nx)
    end
    # the same pattern as the uniform assembly of the single case code
    @test Yancc._matrix(planes.m, planes.rowptr, planes.colval, ones(length(planes.colval))) ==
          (x -> (x.nzval .= 1; x))(Yancc.plane_matrix(ops[1], 1, 1; uniform=true))
    @test Yancc._matrix(coarse.m, coarse.rowptr, coarse.colval, ones(length(coarse.colval))) ==
          (x -> (x.nzval .= 1; x))(first(Yancc.moment_coarse_matrix(ops[1], borders[1], 4; uniform=true, speed=bop.has_speed)))
end

@testset "batched solve: Erho = $(Erho[2]), exchange = $energy_exchange" for (Erho, energy_exchange) in
                                                                            (((0.0, 0.0, 0.0, 0.0), false), ((0.0, 800.0, -1500.0, 300.0), false), ((0.0, 800.0, 0.0, 0.0), true))
    ops = batch_cases(; Erho, energy_exchange)
    EparB = [0.0, 0.02, -0.01, 0.0]
    sols = solve_dke_tokamak_batch(ops; EparB, rtol=1e-10)
    for (k, op) in enumerate(ops)
        ref = solve_dke_tokamak(op; EparB=EparB[k], rtol=1e-10)
        @test sols[k].residual < 1e-10
        # the same iteration as the single case solver, up to rounding
        @test abs(sols[k].iterations - ref.iterations) <= 1
        for f in (:particle_flux, :heat_flux, :VparB)
            @test getfield(sols[k], f) ≈ getfield(ref, f) rtol = 1e-7
        end
        # the sources vanish with the residual unless there is an electric field
        for f in (:particle_source, :heat_source)
            @test getfield(sols[k], f) ≈ getfield(ref, f) atol = 1e-7 * (maximum(abs, getfield(ref, f)) + maximum(abs, ref.heat_flux))
        end
        @test sols[k].JparB ≈ ref.JparB rtol = 1e-7
        @test relerr(sols[k].f1, ref.f1) < 1e-7
    end
    # in pieces, and one case alone
    pieces = solve_dke_tokamak_batch(ops; EparB, rtol=1e-10, batchsize=3)
    @test all(pieces[k].heat_flux ≈ sols[k].heat_flux for k in eachindex(ops))
    alone = solve_dke_tokamak_batch(ops[2:2]; EparB=EparB[2], rtol=1e-10)
    @test alone[1].heat_flux ≈ sols[2].heat_flux rtol = 1e-8
end

@testset "coarse matrices by dense blocks: $p1, Erho = $(Erho[2]), exchange = $energy_exchange" for p1 in ("4z", "4d"),
    (Erho, energy_exchange) in (((0.0, 0.0, 0.0, 0.0), false), ((0.0, 800.0, -1500.0, 300.0), false), ((0.0, 800.0, 0.0, 0.0), true))

    ops = batch_cases(; Erho, energy_exchange, p1, ntheta=13)
    borders = [Yancc.DKEBorder(op) for op in ops]
    bop = Yancc.BatchedDKE(ops, borders, Val(:cpu))
    sparse_maps = Yancc.PreconditionerMaps(ops[1], 4, bop.has_speed, Val(:cpu); coarse_solver=:sparse)
    block_maps = Yancc.PreconditionerMaps(ops[1], 4, bop.has_speed, Val(:cpu); coarse_solver=:block)
    @test block_maps.coarse isa Yancc.BlockAssemblyMap
    @test block_maps.coarse.sb == (p1 == "4z" ? 2 : 4) * 4 * 12
    coarse = Yancc._assemble_preconditioner(sparse_maps, bop)[2]
    blocks = Yancc._assemble_preconditioner(block_maps, bop)[2]
    R = reduce(hcat, [test_vector(coarse.m) .* k .+ 0.3 for k in eachindex(ops)])
    X = Yancc._solve_batch!(similar(R), Yancc._factor_batch(Val(:cpu), coarse), R)
    Xb = Yancc._solve_batch!(similar(R), Yancc._factor_blocks(blocks), R)
    # the matrices are badly scaled (condition numbers beyond 1e17), so the two direct
    # solvers agree to much less than rounding
    @test maximum(norm(Xb[:, k] - X[:, k]) / norm(X[:, k]) for k in eachindex(ops)) < 1e-5
    # the whole solve with either coarse solver
    a = solve_dke_tokamak_batch(ops; rtol=1e-10)
    b = first(Yancc._solve_dke_tokamak_batch(ops; rtol=1e-10, coarse_solver=:block))
    @test all(b[k].residual < 1e-10 && b[k].iterations <= a[k].iterations + 2 for k in eachindex(ops))
    @test all(isapprox(a[k].heat_flux, b[k].heat_flux; rtol=1e-7) for k in eachindex(ops))
end

@testset "coarse matrices by dense blocks: smallest poloidal grids" begin
    # stencils that wrap around the torus more than once still give neighbouring blocks
    for ntheta in (3, 4, 5)
        ops = batch_cases(; ntheta)
        layout = Yancc._coarse_block_layout(ops[1], 4)
        @test layout.K >= 1 && layout.sb * layout.K + layout.nbd == 4 * ntheta * 12 + 12 + 6
        a = solve_dke_tokamak_batch(ops; rtol=1e-10)
        b = first(Yancc._solve_dke_tokamak_batch(ops; rtol=1e-10, coarse_solver=:block))
        @test all(isapprox(a[k].heat_flux, b[k].heat_flux; rtol=1e-7) for k in eachindex(ops))
    end
end

@testset "plane matrices by dense blocks: $p1, nalpha = $nalpha, $points pitch angles a block" for p1 in ("4z", "4d"),
    nalpha in (17, 16, 5), points in (1, 2)

    ops = batch_cases(; p1, nalpha, Erho=(0.0, 800.0, -1500.0, 300.0))
    borders = [Yancc.DKEBorder(op) for op in ops]
    bop = Yancc.BatchedDKE(ops, borders, Val(:cpu))
    na, nt = nalpha, 11
    h = Yancc._plane_block_layout(ops[1]; points=100).sb ÷ nt   # the reach of the α stencils
    @test h == (p1 == "4z" ? 2 : h) && h >= 2
    layout = Yancc._plane_block_layout(ops[1]; points)
    @test layout.sb == points * nt && layout.w == cld(h, points)
    @test layout.K == nalpha ÷ points && layout.nbd == (nalpha % points) * nt
    default = Yancc.PLANE_BLOCK_POINTS[]
    Yancc.PLANE_BLOCK_POINTS[] = points
    try
        sparse_maps = Yancc.PreconditionerMaps(ops[1], 4, bop.has_speed, Val(:cpu))
        block_maps = Yancc.PreconditionerMaps(ops[1], 4, bop.has_speed, Val(:cpu); plane_solver=:block)
        @test block_maps.planes isa Yancc.PlaneBlockMap && block_maps.planes.blocks.w == layout.w
        planes = Yancc._assemble_preconditioner(sparse_maps, bop)[1]
        blocks = Yancc._assemble_preconditioner(block_maps, bop)[1]
        R = reduce(hcat, [test_vector(planes.m) .* k .+ 0.3 for k in 1:length(planes)])
        X = Yancc._solve_batch!(similar(R), Yancc._factor_batch(Val(:cpu), planes), R)
        Xb = Yancc._solve_batch!(similar(R), Yancc._factor_planes(Val(:cpu), blocks), R)
        @test maximum(norm(Xb[:, k] - X[:, k]) / norm(X[:, k]) for k in axes(R, 2)) < 1e-10
        # the whole solve with either plane solver (and the coarse solver of a GPU)
        a = first(Yancc._solve_dke_tokamak_batch(ops; rtol=1e-10, coarse_solver=:block))
        b, status, _ = Yancc._solve_dke_tokamak_batch(ops; rtol=1e-10, plane_solver=:block, coarse_solver=:block)
        # (the residual stops anywhere up to rtol, so rounding can move a case across it)
        @test all(==(:converged), status)
        @test all(abs(b[k].iterations - a[k].iterations) <= 2 for k in eachindex(ops))
        @test all(isapprox(a[k].heat_flux, b[k].heat_flux; rtol=1e-7) for k in eachindex(ops))
    finally
        Yancc.PLANE_BLOCK_POINTS[] = default
    end
end

@testset "batched solve: contract" begin
    ops = batch_cases()
    other = batch_cases(; nalpha=19)
    @test_throws ArgumentError solve_dke_tokamak_batch([ops[1], other[2]])
    @test_throws ArgumentError solve_dke_tokamak_batch(ops; EparB=[0.0, 1.0])
    @test_throws ErrorException solve_dke_tokamak_batch(ops; device=:cuda)
    @test_logs (:warn, r"stopped above") solve_dke_tokamak_batch(ops; rtol=1e-10, maxiter=2)
    @test length(solve_dke_tokamak_batch(ops; device=:auto)) == length(ops)
    # cases that failed in the batch are solved on their own
    sols = solve_dke_tokamak_batch(ops)
    again, status = similar(sols), fill(:failed, length(ops))
    Yancc._solve_failed!(again, status, ops, zeros(length(ops)), [2, 4]; rtol=1e-6, restart=80, maxiter=400)
    @test status == [:failed, :converged, :failed, :converged]
    @test all(isapprox(again[k].heat_flux, sols[k].heat_flux; rtol=1e-5) for k in (2, 4))
end
