# Python tensors are indexed [a, b, ...] with species first; see collisions.jl for the
# Julia index order.

@testset "collisions case $tag" for tag in ("a", "b", "c")
    d = golden("collisions")
    p = tag * "_"
    sp = golden_species(d, p)
    bg = haskey(d, p * "bg_mass") ? golden_species(d, p * "bg_") : LocalMaxwellian{Float64}[]
    nx, na, nL = Int.(d[p*"params"][1:3])
    ll = d[p*"params"][4] < 0 ? nothing : d[p*"params"][4]
    ns = length(sp)
    sg, pg = MaxwellSpeedGrid(nx), UniformPitchAngleGrid(na)

    pot = Yancc.RosenbluthPotentials(sg, sp, nL)
    # reference [a, b, x, l, k] -> [x, l, k, a, b]
    ref(k) = permutedims(d[p*k], (3, 4, 5, 1, 2))
    @testset "Rosenbluth potentials" begin
        for (name, arr) in (("Hxlk", pot.H), ("dHxlk", pot.dH), ("ddGxlk", pot.ddG))
            r = ref(name)
            for a in 1:ns, b in 1:ns, l in 1:nL
                @test relerr(arr[:, l, :, a, b], r[:, l, :, a, b]) < 1e-8
            end
        end
    end

    C = Yancc.CollisionOperator(pg, sg, sp; background=bg, nL, lnlambda=ll, potentials=pot)
    @testset "operator tensors" begin
        @test C.Txi ≈ d[p*"Txi"] atol = 1e-12
        @test relerr(C.Txi_inv, d[p*"Txi_inv"]) < 1e-9
        @test relerr(C.D, d[p*"CL_D"]) < 1e-12
        @test C.nus ≈ permutedims(d[p*"CL_nus"]) rtol = 1e-10
        @test relerr(C.CE, permutedims(d[p*"CE_M"], (2, 3, 1))) < 1e-8
        cd = permutedims(d[p*"CF_C"], (3, 4, 1, 2))
        gh = permutedims(d[p*"CF_GHhat"], (3, 4, 5, 1, 2))
        for a in 1:ns, b in 1:ns
            @test relerr(C.CD[:, :, a, b], cd[:, :, a, b]) < 1e-8
            for l in 1:nL
                @test relerr(C.GH[:, l, :, a, b], gh[:, l, :, a, b]) < 1e-7
            end
        end
        @test relerr(C.exchange, permutedims(d[p*"exchange"], (2, 4, 1, 3))) < 1e-6
    end

    @testset "invariants" begin
        # pitch angle scattering annihilates isotropic functions
        @test maximum(abs.(C.D * ones(na))) < 1e-9 * maximum(abs.(C.D))
        # the corrected isotropic block conserves density and energy of each species
        # and annihilates each species' Maxwellian and temperature perturbation
        K = C.Klocal .+ C.KGH[:, :, 1] .+ C.Kexchange
        wv = sg.x .^ 2 .* sg.wx
        scale = maximum(abs.(K))
        for s in 1:ns
            rng = (1:nx) .+ nx * (s - 1)
            F = sp[s].(sg.x .* sp[s].v_thermal)
            dens = zeros(nx * ns)
            ener = zeros(nx * ns)
            dens[rng] .= wv
            ener[rng] .= wv .* sg.x .^ 2
            fm = zeros(nx * ns)
            ft = zeros(nx * ns)
            fm[rng] .= F
            ft[rng] .= sg.x .^ 2 .* F
            @test maximum(abs.(dens' * K)) < 1e-8 * scale * sum(wv)
            @test maximum(abs.(ener' * K)) < 1e-8 * scale * sum(wv)
            @test maximum(abs.(K * fm)) < 1e-8 * scale * maximum(F)
            @test maximum(abs.(K * ft)) < 1e-8 * scale * maximum(F)
        end
    end

    if haskey(d, p * "block")
        @testset "velocity block" begin
            # local operator on a single surface point, one column per unit vector
            Cl = Yancc.CollisionOperator(pg, sg, sp; background=bg, nL, lnlambda=ll, potentials=pot, energy_exchange=true)
            block = d[p*"block"]   # [p, y, a, s, x, b], output indices first
            w = ones(1, 1)
            f = zeros(na, 1, 1, nx, ns)
            out = similar(f)
            err = 0.0
            for s in 1:ns, i in 1:nx, b in 1:na
                fill!(f, 0)
                fill!(out, 0)
                f[b, 1, 1, i, s] = 1
                Yancc.apply_collisions!(out, Cl, f, w)
                err = max(err, maximum(abs.(out[:, 1, 1, :, :] .- permutedims(block[:, :, :, s, i, b], (3, 2, 1)))))
            end
            @test err < 1e-8 * maximum(abs.(block))
        end
    end
end
