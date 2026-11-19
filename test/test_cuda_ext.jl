# NVIDIA GPU backend of the batched solver, against the host. Not part of the default
# test run (no GPU in CI): run it with
#
#   YANCC_TEST_GPU=1 julia --project=<environment with Yancc, CUDA, CUDSS, NPZ> test/runtests.jl
#
# Every device result is compared with the host path; scalar indexing is forbidden, so
# a host fallback on device arrays fails instead of passing slowly.

import CUDA, CUDSS

if !CUDA.functional()
    @info "test_cuda_ext.jl: CUDA is not functional here, skipping"
else
    CUDA.allowscalar(false)
    cuda = Val(:cuda)

    @testset "CUDA: seams" begin
        A, B = randn(7, 5, 6), randn(5, 3, 6)
        C = Yancc._batched_gemm!('N', 'N', 1.0, A, B, 0.0, zeros(7, 3, 6))
        Cd = Yancc._batched_gemm!('N', 'N', 1.0, CUDA.CuArray(A), CUDA.CuArray(B), 0.0, CUDA.zeros(Float64, 7, 3, 6))
        @test Array(Cd) ≈ C
        # views, as for the Krylov basis
        Ad = CUDA.CuArray(A)
        h = Yancc._batched_gemm!('T', 'N', 1.0, view(A, :, 1:3, :), C[:, 1:1, :], 0.0, zeros(3, 1, 6))
        hd = Yancc._batched_gemm!('T', 'N', 1.0, view(Ad, :, 1:3, :), Cd[:, 1:1, :], 0.0, view(CUDA.zeros(Float64, 5, 1, 6), 1:3, :, :))
        @test Array(hd) ≈ h

        ops = batch_cases(; Erho=(0.0, 800.0, -1500.0, 300.0))
        borders = [Yancc.DKEBorder(op) for op in ops]
        host, dev = Yancc.BatchedDKE(ops, Val(:cpu)), Yancc.BatchedDKE(ops, cuda)
        hostbatches = Yancc._assemble_preconditioner(Yancc.PreconditionerMaps(ops[1], 4, true, Val(:cpu)), host)
        devbatches = Yancc._assemble_preconditioner(Yancc.PreconditionerMaps(ops[1], 4, true, cuda; coarse_solver=:sparse, plane_solver=:sparse), dev)
        # cuDSS factors without pivoting: close to the host LU, not equal to rounding,
        # and far from it for the badly scaled coarse matrices
        for (A, Ad, tol) in ((hostbatches[1], devbatches[1], 1e-8), (hostbatches[2], devbatches[2], 1e-2))
            @test Array(Ad.nzval) ≈ A.nzval rtol = 1e-12
            R = randn(A.m, length(A))
            X = Yancc._solve_batch!(similar(R), Yancc._factor_batch(Val(:cpu), A), R)
            Xd = Yancc._solve_batch!(CUDA.CuArray(similar(R)), Yancc._factor_batch(cuda, Ad), CUDA.CuArray(R))
            @test all(isfinite, Array(Xd))
            @test maximum(norm(Array(Xd)[:, k] - X[:, k]) / norm(X[:, k]) for k in 1:length(A)) < tol
        end
    end

    @testset "CUDA: coarse matrices by dense blocks, $p1" for p1 in ("4z", "4d")
        ops = batch_cases(; Erho=(0.0, 800.0, -1500.0, 300.0), p1, ntheta=13)
        borders = [Yancc.DKEBorder(op) for op in ops]
        host, dev = Yancc.BatchedDKE(ops, Val(:cpu)), Yancc.BatchedDKE(ops, cuda)
        maps = Yancc.PreconditionerMaps(ops[1], 4, true, cuda)
        @test maps.coarse isa Yancc.BlockAssemblyMap   # the default on the device
        blocks = Yancc._assemble_preconditioner(Yancc.PreconditionerMaps(ops[1], 4, true, Val(:cpu); coarse_solver=:block), host)[2]
        devblocks = Yancc._assemble_preconditioner(maps, dev)[2]
        @test all(Array(devblocks[name]) ≈ blocks[name] for name in (:D, :L, :U, :Ub, :Vb, :S))
        R = reduce(hcat, [test_vector(size(blocks.S, 1) + blocks.sb * blocks.K) .* k .+ 0.3 for k in eachindex(ops)])
        X = Yancc._solve_batch!(similar(R), Yancc._factor_blocks(blocks), R)
        Xd = Yancc._solve_batch!(CUDA.CuArray(similar(R)), Yancc._factor_blocks(devblocks), CUDA.CuArray(R))
        @test maximum(norm(Array(Xd)[:, k] - X[:, k]) / norm(X[:, k]) for k in eachindex(ops)) < 1e-5
    end

    @testset "CUDA: plane matrices by dense blocks, $p1, nalpha = $nalpha" for p1 in ("4z", "4d"), nalpha in (17, 16)
        ops = batch_cases(; Erho=(0.0, 800.0, -1500.0, 300.0), p1, nalpha)
        borders = [Yancc.DKEBorder(op) for op in ops]
        host, dev = Yancc.BatchedDKE(ops, Val(:cpu)), Yancc.BatchedDKE(ops, cuda)
        maps = Yancc.PreconditionerMaps(ops[1], 4, true, cuda; plane_solver=:block)
        @test maps.planes isa Yancc.PlaneBlockMap
        # by default only large planes go by blocks on the device
        @test Yancc._default_plane_solver(cuda, ops[1]) == :sparse
        @test Yancc._default_plane_solver(cuda, batch_cases(; nalpha=129, ntheta=61)[1]) == :block
        planes = Yancc._assemble_preconditioner(Yancc.PreconditionerMaps(ops[1], 4, true, Val(:cpu)), host)[1]
        devblocks = Yancc._assemble_preconditioner(maps, dev)[1]
        R = reduce(hcat, [test_vector(planes.m) .* k .+ 0.3 for k in 1:length(planes)])
        X = Yancc._solve_batch!(similar(R), Yancc._factor_batch(Val(:cpu), planes), R)
        Xd = Yancc._solve_batch!(CUDA.CuArray(similar(R)), Yancc._factor_planes(cuda, devblocks), CUDA.CuArray(R))
        @test maximum(norm(Array(Xd)[:, k] - X[:, k]) / norm(X[:, k]) for k in axes(R, 2)) < 1e-10
        # the whole solve on the device with them
        a = solve_dke_tokamak_batch(ops; rtol=1e-10)
        b = first(Yancc._solve_dke_tokamak_batch(ops; rtol=1e-10, device=:cuda, plane_solver=:block))
        @test all(b[k].residual < 1e-10 && isapprox(a[k].heat_flux, b[k].heat_flux; rtol=1e-7) for k in eachindex(ops))
    end

    @testset "CUDA: batched operator and preconditioner, Erho = $(Erho[2]), exchange = $energy_exchange" for (Erho, energy_exchange) in
                                                                                                           (((0.0, 0.0, 0.0, 0.0), false), ((0.0, 800.0, -1500.0, 300.0), false), ((0.0, 800.0, 0.0, 0.0), true))
        ops = batch_cases(; Erho, energy_exchange)
        borders = [Yancc.DKEBorder(op) for op in ops]
        host, dev = Yancc.BatchedDKE(ops, Val(:cpu)), Yancc.BatchedDKE(ops, cuda)
        M = Yancc.system_size(ops[1])
        X = reduce(hcat, [test_vector(M) .* (1 + 0.1 * k) .+ 0.01 * k for k in eachindex(ops)])
        Y = Yancc.apply_bordered!(similar(X), host, X)
        Yd = Yancc.apply_bordered!(CUDA.CuArray(similar(X)), dev, CUDA.CuArray(X))
        @test Array(Yd) ≈ Y rtol = 1e-12
        Z = Yancc._precondition!(similar(X), Yancc.BatchedTokamakPreconditioner(ops, host, Val(:cpu)), copy(X), host)
        Zd = Yancc._precondition!(CUDA.CuArray(similar(X)), Yancc.BatchedTokamakPreconditioner(ops, dev, cuda), CUDA.CuArray(X), dev)
        @test maximum(norm(Array(Zd)[:, k] - Z[:, k]) / norm(Z[:, k]) for k in eachindex(ops)) < 1e-4
    end

    @testset "CUDA: batched solve, Erho = $(Erho[2]), exchange = $energy_exchange" for (Erho, energy_exchange) in
                                                                                    (((0.0, 0.0, 0.0, 0.0), false), ((0.0, 800.0, -1500.0, 300.0), false), ((0.0, 800.0, 0.0, 0.0), true))
        ops = batch_cases(; Erho, energy_exchange)
        EparB = [0.0, 0.02, -0.01, 0.0]
        host = solve_dke_tokamak_batch(ops; EparB, rtol=1e-10)
        dev = solve_dke_tokamak_batch(ops; EparB, rtol=1e-10, device=:cuda)
        for k in eachindex(ops)
            @test dev[k].residual < 1e-10
            # the device solves the coarse problems by blocks, the host by a sparse LU
            @test dev[k].iterations <= host[k].iterations + 2
            for f in (:particle_flux, :heat_flux, :VparB)
                @test getfield(dev[k], f) ≈ getfield(host[k], f) rtol = 1e-7
            end
            @test dev[k].JparB ≈ host[k].JparB rtol = 1e-7
        end
        @test length(solve_dke_tokamak_batch(ops; device=:auto, batchsize=3)) == length(ops)
    end

    @testset "CUDA: rotation, exchange = $energy_exchange" for energy_exchange in (false, true)
        ops = rotating_batch_cases(; energy_exchange)
        host, dev = Yancc.BatchedDKE(ops, Val(:cpu)), Yancc.BatchedDKE(ops, cuda)
        M = Yancc.system_size(ops[1])
        X = reduce(hcat, [test_vector(M) .* (1 + 0.1 * k) .+ 0.01 * k for k in eachindex(ops)])
        Y = Yancc.apply_bordered!(similar(X), host, X)
        Yd = Yancc.apply_bordered!(CUDA.CuArray(similar(X)), dev, CUDA.CuArray(X))
        @test Array(Yd) ≈ Y rtol = 1e-12
        @test Array(dev.data.scaling) ≈ host.data.scaling rtol = 1e-13
        EparB = [0.0, 0.02, -0.01, 0.0]
        @test Array(Yancc._batched_rhs(ops, EparB, dev)) ≈ Yancc._batched_rhs(ops, EparB, host) rtol = 1e-13
        Z = Yancc._precondition!(similar(X), Yancc.BatchedTokamakPreconditioner(ops, host, Val(:cpu)), copy(X), host)
        Zd = Yancc._precondition!(CUDA.CuArray(similar(X)), Yancc.BatchedTokamakPreconditioner(ops, dev, cuda), CUDA.CuArray(X), dev)
        @test maximum(norm(Array(Zd)[:, k] - Z[:, k]) / norm(Z[:, k]) for k in eachindex(ops)) < 1e-4
        sols = solve_dke_tokamak_batch(ops; EparB, rtol=1e-10, device=:cuda)
        for (k, op) in enumerate(ops)
            ref = solve_dke_tokamak(op; EparB=EparB[k], rtol=1e-10)
            @test sols[k].residual < 1e-10
            @test sols[k].iterations <= ref.iterations + 2
            for f in (:particle_flux, :heat_flux, :VparB)
                @test getfield(sols[k], f) ≈ getfield(ref, f) rtol = 1e-7
            end
            @test sols[k].angular_momentum_flux ≈ ref.angular_momentum_flux rtol = 1e-7 atol = 1e-7 * maximum(abs, sols[2].angular_momentum_flux)
            @test sols[k].JparB ≈ ref.JparB rtol = 1e-7
        end
    end

    @testset "CUDA: rotation at working resolution, the cases checked against NEO" begin
        # one batch of the MAST-U cases of golden/rotation.npz (41 poloidal nodes), with the plane
        # matrices by dense blocks, against the single case solver
        d = golden("rotation")
        ops = [begin
            p = "fluxes_" * case * "_"
            DKEOperator(rotation_field(d, p), UniformPitchAngleGrid(65), MaxwellSpeedGrid(6), golden_species(d, p), 0.0;
                lnlambda=d[p*"lnlambda"], rotation=rotation_of(d, p))
        end for case in ("mastu_r045_rot", "mastu_r085_rot", "mastu_r065_rot", "mastu_r045_reversed", "mastu_r045_bothflip",
            "mastu_r045_shearonly", "updown_mastu_r045_rot")]
        dev = solve_dke_tokamak_batch(ops; device=:cuda, rtol=1e-8)
        for (k, op) in enumerate(ops)
            ref = solve_dke_tokamak(op; rtol=1e-8)
            @test dev[k].residual < 1e-8
            @test dev[k].iterations <= ref.iterations + 3
            for f in (:particle_flux, :heat_flux, :angular_momentum_flux)
                @test getfield(dev[k], f) ≈ getfield(ref, f) rtol = 1e-5
            end
            @test dev[k].JparB ≈ ref.JparB rtol = 1e-5
        end
    end

    @testset "CUDA: working resolution" begin
        # sizes at which every code path is taken (restriction, prolongation and plane
        # solves of thousands of unknowns), against the single case solver
        ops = batch_cases(; ntheta=41, nalpha=65, nx=6, Erho=(0.0, 800.0, -1500.0, 300.0))
        dev = solve_dke_tokamak_batch(ops; device=:cuda)
        for (k, op) in enumerate(ops)
            ref = solve_dke_tokamak(op)
            @test dev[k].residual < 1e-6
            @test dev[k].iterations <= ref.iterations + 3
            @test dev[k].heat_flux ≈ ref.heat_flux rtol = 1e-4
            @test dev[k].JparB ≈ ref.JparB rtol = 1e-4
        end
    end
end
