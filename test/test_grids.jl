@testset "grids" begin
    d = golden("grids")

    @testset "speed grid nx=$nx" for nx in (3, 5, 6, 8)
        g = MaxwellSpeedGrid(nx)
        p = "speed$(nx)_"
        @test g.x ≈ d[p*"x"] rtol = 1e-13
        @test g.wx ≈ d[p*"wx"] rtol = 1e-12
        @test relerr(g.xvander, d[p*"xvander"]) < 1e-12
        @test relerr(g.Dx, d[p*"Dx_pseudospectral"]) < 1e-10
        @test relerr(g.D2x, d[p*"D2x_pseudospectral"]) < 1e-9
        # quadrature is exact for exp(-x^2) times polynomials of degree < 2 nx
        @test sum(g.wx .* exp.(-g.x .^ 2) .* g.x .^ 2) ≈ sqrt(π) / 4 rtol = 1e-13
        @test g.xvander * g.xvander_inv ≈ I(nx) atol = 1e-10
    end

    @testset "pitch grid na=$na" for na in (9, 15, 16)
        g = UniformPitchAngleGrid(na)
        p = "pitch$(na)_"
        @test g.alpha ≈ d[p*"alpha"] rtol = 1e-14
        @test g.xi ≈ d[p*"xi"] rtol = 1e-13 atol = 1e-15
        @test g.wxi ≈ d[p*"wxi"] rtol = 1e-12
        @test sum(g.wxi) ≈ 2 rtol = 1e-14
    end

    @testset "finite differences n=$n $bc" for n in (5, 7, 9, 16), bc in (:periodic, :symmetric)
        h = d["fd_h"]
        for p in ("4d", "2d", "4z", "2z")
            @test Yancc.fdfwd_matrix(p, n, h, bc) ≈ d["fdfwd_$(p)_$(bc)_$(n)"] atol = 1e-12
            @test Yancc.fdbwd_matrix(p, n, h, bc) ≈ d["fdbwd_$(p)_$(bc)_$(n)"] atol = 1e-12
        end
        for p in (2, 4)
            @test Yancc.fd2_matrix(p, n, h, bc) ≈ d["fd2_$(p)_$(bc)_$(n)"] atol = 1e-11
        end
    end

    @testset "stencil accuracy" begin
        n = 64
        h = 2π / n
        t = (0:(n-1)) .* h
        for p in ("2d", "4d")
            order = parse(Int, p[1:1])
            @test maximum(abs.(Yancc.fdfwd_matrix(p, n, h, :periodic) * sin.(t) .- cos.(t))) < 5 * h^order
            @test maximum(abs.(Yancc.fdbwd_matrix(p, n, h, :periodic) * sin.(t) .- cos.(t))) < 5 * h^order
        end
        @test maximum(abs.(Yancc.fd2_matrix(4, n, h, :periodic) * sin.(t) .+ sin.(t))) < h^4
    end
end

@testset "species" begin
    d = golden("species")
    sp = golden_species(d)
    x = d["x"]
    n = length(sp)
    @test [s.v_thermal for s in sp] ≈ d["v_thermal"] rtol = 1e-13
    @test [s(xi * s.v_thermal) for s in sp, xi in x] ≈ d["F"] rtol = 1e-12
    @test [Yancc.coulomb_logarithm(a, b) for a in sp, b in sp] ≈ d["lnlambda"] rtol = 1e-12
    for (tag, ll) in (("", nothing), ("_ll17", 17.0))
        @test [Yancc.gamma_ab(a, b, ll) for a in sp, b in sp] ≈ d["gamma"*tag] rtol = 1e-12
        @test [Yancc.nuD_ab(sp[a], sp[b], xi * sp[a].v_thermal, ll) for a in 1:n, b in 1:n, xi in x] ≈ d["nuD"*tag] rtol = 1e-10
        @test [Yancc.nupar_ab(sp[a], sp[b], xi * sp[a].v_thermal, ll) for a in 1:n, b in 1:n, xi in x] ≈ d["nupar"*tag] rtol = 1e-9
    end
    # NRL formulary: electron-ion Coulomb logarithm of a 1 keV, 1e20 m^-3 plasma ~ 15-17
    e = LocalMaxwellian(Electron, 1e3, 1e20, 0.0, 0.0)
    i = LocalMaxwellian(Deuterium, 1e3, 1e20, 0.0, 0.0)
    @test 14 < Yancc.coulomb_logarithm(e, i) < 18
end
