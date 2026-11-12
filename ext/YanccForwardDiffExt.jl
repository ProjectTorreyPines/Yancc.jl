# Derivatives of the tokamak solver with ForwardDiff.
#
# For A(p) y = b(p) the derivative of the solution in a direction k solves
# A δ_k = ∂_k b - (∂_k A) y. The right hand sides are the partials of b(p) - A(p) y
# evaluated with the operator of dual numbers on the solution of the value problem, and
# the solves use the operator and the preconditioner of the value problem.
module YanccForwardDiffExt

using Yancc
using Yancc: DKEOperator, DKEBorder, DKESolution, TokamakCache, TokamakPreconditioner
using ForwardDiff: ForwardDiff, Dual, Partials, value, partials

_number_type(::DKEOperator{T}) where {T} = T

function Yancc._solve_dual(cache::Union{Nothing,TokamakCache}, op::DKEOperator, EparB::Real;
    rtol::Real=1e-6, restart::Integer=80, maxiter::Integer=400, tangent_rtol::Real=1e-6,
    preconditioner::Union{Nothing,TokamakPreconditioner}=nothing, guess::Union{Nothing,DKESolution}=nothing, kw...)

    D = promote_type(_number_type(op), typeof(EparB))
    D <: Dual && ForwardDiff.valtype(D) === Float64 ||
        error("solve_dke_tokamak: derivatives are implemented for ForwardDiff.Dual numbers of Float64 values, got $D")
    np = ForwardDiff.npartials(D)
    opd = op isa DKEOperator{D} ? op : Yancc.map_parameters(x -> convert(D, x), D, op)
    op0 = Yancc.map_parameters(value, Float64, opd)
    EparB0 = Float64(value(EparB))

    # value problem
    if cache === nothing
        P = preconditioner === nothing ? TokamakPreconditioner(op0) : preconditioner
        sol0 = solve_dke_tokamak(op0; EparB=EparB0, rtol, restart, maxiter, preconditioner=P, guess)
        Q = Z = zeros(Yancc.system_size(op0), 0)
    else
        sol0 = solve_dke_tokamak!(cache, op0; EparB=EparB0, rtol, restart, maxiter, kw...)
        P = cache.preconditioner
        # solved directions belong to the operator, which the temperatures, densities and Erho determine
        if Yancc._plasma_change(op0, cache.tangent_state) > Yancc.SAME_INPUT || size(cache.tangent_rhs, 1) != Yancc.system_size(op0)
            cache.tangent_state = Yancc._plasma_state(op0)
            cache.tangent_rhs = cache.tangent_sol = zeros(Yancc.system_size(op0), 0)
        end
        Q, Z = cache.tangent_rhs, cache.tangent_sol
    end
    y0 = Yancc.solution_vector(sol0)

    # residual in dual numbers; its partials are the right hand sides
    n, N = Yancc.system_size(op0), prod(Yancc.state_size(op0))
    r = zeros(D, n)
    Yancc.apply_bordered!(r, opd, DKEBorder(opd), y0)
    r .= .-r
    view(r, 1:N) .+= vec(Yancc.dke_rhs(opd, EparB))
    R = [partials(r[i], k) for i in 1:n, k in 1:np]

    Δ, Q, Z, solves, iterations = Yancc._solve_tangents(R, op0, DKEBorder(op0), P, Q, Z; rtol=tangent_rtol, restart, maxiter)
    if cache !== nothing
        cache.tangent_rhs, cache.tangent_sol = Q, Z
        cache.tangent_solves += solves
        cache.iterations += iterations
    end

    y = [D(y0[i], Partials{np,Float64}(ntuple(k -> Δ[i, k], np))) for i in 1:n]
    return DKESolution(opd, y; iterations=sol0.iterations + iterations, residual=sol0.residual)
end

end
