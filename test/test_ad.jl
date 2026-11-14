# Derivatives with ForwardDiff: the dense solve against finite differences, and the
# tokamak solver (implicit differentiation, ext/YanccForwardDiffExt.jl) against the
# dense solve.

using ForwardDiff

# D + e with a carbon background; p = (T, n, dT/dρ, dn/dρ) of D and of e, Erho, EparB, T and n of C.
# The charges of the ions depend on their temperatures, as an average charge state does.
function ad_operator(p; p1="4z", nalpha=11, nx=3)
    species = [LocalMaxwellian(Species(2, 1 + 1e-5 * p[1]), p[1], p[2], p[3], p[4]), LocalMaxwellian(Electron, p[5], p[6], p[7], p[8])]
    background = [LocalMaxwellian(Species(12, 6 * p[11] / (p[11] + 100)), p[11], p[12], 0.0, 0.0)]
    return DKEOperator(circular_field(9, 0.15), UniformPitchAngleGrid(nalpha), MaxwellSpeedGrid(nx), species, p[9]; background, p1)
end
ad_outputs(sol) = [sol.particle_flux; sol.heat_flux; sol.JparB; sol.VparB]
const AD_P0 = [2e3, 5e19, -6e3, -5e19, 1.5e3, 5e19, -4e3, -5e19, 300.0, 0.02, 1.8e3, 2e18]
# one tag and one chunk size for all functions, so that the code is compiled once
ad_jacobian(f, x) = ForwardDiff.jacobian(f, x, ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(AD_P0)}()), Val(false))
column_error(J, Jref) = maximum(maximum(abs, J[:, k] - Jref[:, k]) / maximum(abs, Jref[:, k]) for k in axes(Jref, 2))

@testset "derivatives of the dense solve" begin
    f(p) = ad_outputs(solve_dke_reference(ad_operator(p; nalpha=9); EparB=p[10]))
    J = ad_jacobian(f, AD_P0)
    Jfd = similar(J)
    for k in eachindex(AD_P0)
        e = zeros(length(AD_P0))
        e[k] = 1e-5 * abs(AD_P0[k])
        Jfd[:, k] .= (f(AD_P0 + e) .- f(AD_P0 - e)) ./ 2e[k]
    end
    @test column_error(J, Jfd) < 1e-4
end

@testset "derivatives of the tokamak solver, $p1" for p1 in ("4z", "4d")
    dense(p) = ad_outputs(solve_dke_reference(ad_operator(p; p1); EparB=p[10]))
    tokamak(p) = ad_outputs(solve_dke_tokamak(ad_operator(p; p1); EparB=p[10], rtol=1e-10, tangent_rtol=1e-9))
    Jref = ad_jacobian(dense, AD_P0)
    @test tokamak(AD_P0) ≈ dense(AD_P0) rtol = 1e-6
    @test column_error(ad_jacobian(tokamak, AD_P0), Jref) < 2e-5
    if p1 == "4z"
        n = length(AD_P0)
        # dual EparB with an operator of plain numbers
        op = ad_operator(AD_P0)
        d = ad_jacobian(u -> ad_outputs(solve_dke_tokamak(op; EparB=AD_P0[10] + u[1], rtol=1e-10, tangent_rtol=1e-9)), zeros(n))
        @test d[:, 1] ≈ Jref[:, 10] rtol = 1e-6

        # one cache: the derivative solves are reused by later calls at the same point
        cache = TokamakCache()
        cached(p) = ad_outputs(solve_dke_tokamak!(cache, ad_operator(p); EparB=p[10], rtol=1e-10, tangent_rtol=1e-9))
        along(M) = u -> cached(AD_P0 .+ M * u)
        primal = cached(AD_P0)
        seeds = zeros(n, n)   # three directions
        seeds[:, 1:3] .= [sin(i * k) * abs(AD_P0[i]) for i in 1:n, k in 1:3]
        @test ad_jacobian(along(seeds), zeros(n)) ≈ Jref * seeds rtol = 1e-5
        @test cache.rebuilds == 1 && cache.tangent_solves == 3
        iterations = cache.iterations
        mixed = seeds * [cos(i * k) for i in 1:n, k in 1:n]   # twelve directions in the same space
        @test ad_jacobian(along(mixed), zeros(n)) ≈ Jref * mixed rtol = 1e-5
        @test cache.tangent_solves == 3 && cache.rebuilds == 1 && cache.iterations == iterations
        unit = [ForwardDiff.Dual{Nothing}(0.0, ntuple(j -> Float64(j == k), n)) for k in 1:n]
        @test ForwardDiff.value.(along(seeds)(unit)) ≈ primal rtol = 1e-12
        # zero partials need no solve
        @test all(iszero, ad_jacobian(along(zeros(n, n)), zeros(n)))
        @test cache.tangent_solves == 3
        # another plasma: the solved directions are dropped
        moved = AD_P0 .* [1.02; ones(n - 1)]
        Jmoved = ad_jacobian(p -> ad_outputs(solve_dke_reference(ad_operator(p); EparB=p[10])), moved)
        @test ad_jacobian(u -> cached(moved .+ seeds * u), zeros(n)) ≈ Jmoved * seeds rtol = 1e-5
        @test size(cache.tangent_rhs, 2) == 3 && cache.tangent_solves == 6
    end
end
