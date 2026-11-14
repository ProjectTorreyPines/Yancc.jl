# Many flux surfaces or plasmas solved together, on the host or on a GPU.

"""
    solve_dke_tokamak_batch(ops; EparB=0, device=:cpu, rtol=1e-6, restart=80, maxiter=400,
                            moments=4, batchsize=nothing)

[`solve_dke_tokamak`](@ref) for every operator of `ops` at once; returns their
[`DKESolution`](@ref)s. The operators must share the grids, stencils and options
(`nalpha`, `ntheta`, `nx`, number of species, `p1`, `p2`, `nL`, `operator_weights`,
`energy_exchange`); the field, the plasma and the electric field may differ, as between
the flux surfaces of a profile or the points of a scan. `EparB` is one value or one
per case.

`device = :cuda` runs the Krylov iteration and the preconditioner on an NVIDIA GPU and
needs `using CUDA, CUDSS` first; `:auto` uses the GPU when there is one. The operators
themselves, and the fluxes of the solutions, are computed on the host, on all Julia
threads. `batchsize` limits the number of cases solved together (default: all on the
host, as many as fit in the free memory on a device).

A case whose batched factorization fails (the sparse solver of the GPU does not pivot,
and has been seen to return NaN for a few matrices of a large batch) is solved again
on its own by `solve_dke_tokamak`.

A solve on its own does not gain from a GPU; many together do.
"""
function solve_dke_tokamak_batch(ops::AbstractVector{<:DKEOperator{Float64}}; kw...)
    sols, status, _ = _solve_dke_tokamak_batch(ops; kw...)
    bad = count(k -> status[k] in (:maxiter, :failed) || (status[k] == :stagnated && sols[k].residual > 1e-6), eachindex(sols))
    if bad > 0
        worst = any(==(:failed), status) ? NaN : maximum(sol.residual for sol in sols)
        @warn "solve_dke_tokamak_batch: $bad of $(length(sols)) cases stopped above the requested residual, the largest at $worst"
    end
    return sols
end

# solutions, status of each case, and the seconds spent in each stage
function _solve_dke_tokamak_batch(ops::AbstractVector{<:DKEOperator{Float64}}; EparB=0.0, device::Symbol=:cpu,
    rtol::Real=1e-6, restart::Integer=80, maxiter::Integer=400, moments::Integer=4,
    batchsize::Union{Nothing,Integer}=nothing, coarse_solver::Union{Nothing,Symbol}=nothing,
    plane_solver::Union{Nothing,Symbol}=nothing)

    _check_batch(ops)
    ncases = length(ops)
    epar = EparB isa Real ? fill(Float64(EparB), ncases) : collect(Float64, EparB)
    length(epar) == ncases || throw(ArgumentError("EparB needs one value, or one per case"))
    dev = _pick_device(device)
    plane_solver = something(plane_solver, _default_plane_solver(dev, first(ops)))
    coarse_solver = something(coarse_solver, _default_coarse_solver(dev))
    planes = plane_solver == :block ? _plane_block_layout(first(ops)) : nothing
    bsize = batchsize === nothing ? _default_batchsize(dev, ncases, system_size(first(ops)), restart, planes) : Int(batchsize)
    bsize >= 1 || throw(ArgumentError("batchsize must be positive"))

    sols = Vector{DKESolution{Float64}}(undef, ncases)
    status = Vector{Symbol}(undef, ncases)
    seconds = Dict(:maps => 0.0, :assemble => 0.0, :factor => 0.0, :iterate => 0.0, :moments => 0.0)
    maps = Ref{Any}(nothing)   # built with the first chunk, the same for all
    for chunk in Iterators.partition(1:ncases, bsize)
        _solve_chunk!(sols, status, seconds, maps, ops, epar, chunk, dev; rtol, restart, maxiter, moments, coarse_solver, plane_solver)
        _release(dev)   # the factors of a chunk are large; the next one needs the room
    end
    return sols, status, seconds
end

function _solve_chunk!(sols, status, seconds, maps, ops, epar, chunk, dev::Val; rtol, restart, maxiter, moments, coarse_solver, plane_solver)
    cases = view(ops, chunk)
    B = length(cases)
    N, M = prod(state_size(first(cases))), system_size(first(cases))

    # operators, right hand sides and preconditioner matrices
    local bop, planes_coarse, rhs
    seconds[:assemble] += @elapsed begin
        # the same structure for every chunk, whichever cases have an electric field
        speed = any(op.trajectories.has_speed for op in ops) && first(ops).weights[1] != 0
        bop = BatchedDKE(cases, dev; speed)
        rhs = _batched_rhs(cases, view(epar, chunk), bop)
        if maps[] === nothing
            t = @elapsed (maps[] = PreconditionerMaps(first(ops), min(Int(moments), bop.nL), speed, dev;
                coarse_solver, plane_solver))
            seconds[:maps] += t
            seconds[:assemble] -= t
        end
        planes_coarse = _assemble_preconditioner(maps[], bop)
        _synchronize(dev)
    end
    local pre
    seconds[:factor] += @elapsed begin
        pre = BatchedTokamakPreconditioner(maps[], bop, planes_coarse..., dev)
        planes_coarse = nothing   # the factors replace the assembled blocks
        _synchronize(dev)
    end

    # GMRES in the scaled norm of the single case solver
    local Xh, iterations, residuals, stat
    seconds[:iterate] += @elapsed begin
        S = bop.data.scaling
        Bd = rhs
        Bd .*= S
        X = fill!(similar(Bd), 0)
        tmp = similar(Bd)
        A!(out, v) = (apply_bordered!(out, bop, v); out .*= S; out)
        M!(z, v) = (tmp .= v ./ S; _precondition!(z, pre, tmp, bop))
        X, iterations, residuals, stat = gmres_batched!(X, A!, Bd, M!; rtol, restart, maxiter)
        Xh = Array(X)
        foreach(_free!, (X, Bd, tmp))
        _free!(pre)   # the device memory of the chunk, now rather than at a garbage collection
    end

    seconds[:moments] += @elapsed begin
        Threads.@threads for b in 1:B
            stat[b] == :failed && continue
            sols[chunk[b]] = DKESolution(cases[b], view(Xh, :, b); iterations=iterations[b], residual=residuals[b])
            status[chunk[b]] = stat[b]
        end
    end
    _free!(bop)
    seconds[:iterate] += @elapsed _solve_failed!(sols, status, ops, epar, chunk[stat.==:failed]; rtol, restart, maxiter)
    return nothing
end

# the cases whose batched solve met a number that is not finite, one by one on the host
function _solve_failed!(sols, status, ops, epar, failed; rtol, restart, maxiter)
    isempty(failed) && return nothing
    @debug "solve_dke_tokamak_batch: $(length(failed)) cases solved again on their own"
    Threads.@threads for k in failed
        sol, stat = _solve_dke_tokamak(ops[k], TokamakPreconditioner(ops[k]), nothing; EparB=epar[k], rtol, restart, maxiter)
        sols[k], status[k] = sol, isfinite(sol.residual) ? stat : :failed
    end
    return nothing
end
