# Restarted GMRES with right preconditioning for a batch of systems, iterated together.

"""
    gmres_batched!(X, A!, B, M!; rtol=1e-8, restart=60, maxiter=300)

[`gmres!`](@ref) for the systems `A_k x_k = b_k` whose unknowns and right hand sides
are the columns of `X` and `B`. `A!(Y, V)` and `M!(Z, V)` act on all columns at once
and must map a zero column to a zero column.

All systems take their Arnoldi steps together. One that has converged, or used its
`maxiter` iterations, is left alone from then on: its part of the Krylov basis is zero,
so the steps of the others do not change it. The small Hessenberg problems are kept on
the host, one per system.

Returns `(X, iterations, relative_residuals, status)`, the last three with one entry
per system; `status` is `:converged`, `:stagnated` or `:maxiter` as for `gmres!`, or
`:failed` for a system whose iteration met a number that is not finite (a failed
factorization in `M!`, say). Such a system is stopped at once, its column of `X` is not
meaningful, and the other systems are not affected.
"""
function gmres_batched!(X::AbstractMatrix{T}, A!, B::AbstractMatrix{T}, M!;
    rtol::Real=1e-8, restart::Integer=60, maxiter::Integer=300) where {T<:AbstractFloat}

    n, nb = size(B)
    m = min(restart, maxiter)
    V = similar(B, n, min(m, 12) + 1, nb)   # Krylov bases [:, vector, system], grown as needed
    H = zeros(T, m + 1, m, nb)
    cs, sn, g = zeros(T, m, nb), zeros(T, m, nb), zeros(T, m + 1, nb)
    coefficients = similar(B, m, 1, nb)
    W, Z, R, Vj = similar(B), similar(B), similar(B), similar(B)
    W3 = reshape(W, n, 1, nb)
    row(v::Vector) = _like(B, reshape(v, 1, nb))

    bnorm = _column_norms(B)
    tol = rtol .* bnorm
    iterations = zeros(Int, nb)
    status = fill(:maxiter, nb)
    done = bnorm .== 0
    status[done] .= :converged
    any(done) && (X .*= row(T.(.!done)))

    if any(!iszero, X)
        A!(R, X)
        R .= B .- R
    else
        copyto!(R, B)
    end
    beta = _column_norms(R)
    for k in 1:nb
        done[k] && continue
        if !isfinite(beta[k])
            done[k], status[k] = true, :failed
        elseif beta[k] <= tol[k]
            done[k], status[k] = true, :converged
        elseif maxiter <= 0
            done[k] = true
        end
    end

    while !all(done)
        active = .!done
        view(V, :, 1, :) .= R .* row([active[k] ? 1 / beta[k] : zero(T) for k in 1:nb])
        fill!(g, zero(T))
        g[1, :] .= ifelse.(active, beta, zero(T))
        steps = zeros(Int, nb)   # Arnoldi steps of each system in this cycle
        last = 0
        for j in 1:m
            if j + 1 > size(V, 2)
                grown = similar(V, n, min(2 * size(V, 2), m + 1), nb)
                view(grown, :, 1:size(V, 2), :) .= V
                V = grown
            end
            Vj .= view(V, :, j, :)
            M!(Z, Vj)
            A!(W, Z)
            # classical Gram-Schmidt, applied twice
            basis = view(V, :, 1:j, :)
            hj = view(coefficients, 1:j, :, :)
            for pass in 1:2
                _batched_gemm!('T', 'N', one(T), basis, W3, zero(T), hj)
                _batched_gemm!('N', 'N', -one(T), basis, hj, one(T), W3)
                hhost = reshape(Array(hj), j, nb)
                pass == 1 ? (@views H[1:j, j, :] .= hhost) : (@views H[1:j, j, :] .+= hhost)
            end
            wnorm = _column_norms(W)

            # Givens rotations bringing each Hessenberg matrix to triangular form
            scale = zeros(T, nb)
            for k in 1:nb
                active[k] || continue
                H[j+1, j, k] = wnorm[k]
                for i in 1:(j-1)
                    t = cs[i, k] * H[i, j, k] + sn[i, k] * H[i+1, j, k]
                    H[i+1, j, k] = -sn[i, k] * H[i, j, k] + cs[i, k] * H[i+1, j, k]
                    H[i, j, k] = t
                end
                d = hypot(H[j, j, k], H[j+1, j, k])
                if !isfinite(d)   # (0 NaN = NaN: its basis stays NaN, in its own columns only)
                    active[k], done[k], status[k] = false, true, :failed
                    continue
                elseif d == 0   # the Krylov space is exhausted without a new direction
                    active[k] = false
                    continue
                end
                cs[j, k], sn[j, k] = H[j, j, k] / d, H[j+1, j, k] / d
                H[j, j, k] = d
                H[j+1, j, k] = zero(T)
                g[j+1, k] = -sn[j, k] * g[j, k]
                g[j, k] = cs[j, k] * g[j, k]
                steps[k] = j
                iterations[k] += 1
                if abs(g[j+1, k]) <= tol[k] || iterations[k] >= maxiter || wnorm[k] == 0
                    active[k] = false
                else
                    scale[k] = 1 / wnorm[k]
                end
            end
            last = j
            any(active) || break
            view(V, :, j + 1, :) .= W .* row(scale)
        end

        y = zeros(T, last, 1, nb)
        for k in 1:nb
            s = steps[k]
            s > 0 && status[k] != :failed && (y[1:s, 1, k] .= UpperTriangular(view(H, 1:s, 1:s, k)) \ view(g, 1:s, k))
        end
        _batched_gemm!('N', 'N', one(T), view(V, :, 1:last, :), _like(B, y), zero(T), W3)
        M!(Z, W)
        X .+= Z
        A!(R, X)
        R .= B .- R
        previous = beta
        beta = _column_norms(R)
        for k in 1:nb
            done[k] && continue
            s = steps[k]
            if !isfinite(beta[k])
                done[k], status[k] = true, :failed
            elseif s > 0 && abs(g[s+1, k]) <= tol[k] && beta[k] > max(tol[k], previous[k] / 2)
                # the recurrence reports convergence, the true residual does not improve
                done[k], status[k] = true, :stagnated
            elseif beta[k] <= tol[k]
                done[k], status[k] = true, :converged
            elseif iterations[k] >= maxiter || s == 0
                done[k] = true
            end
        end
    end
    residuals = [bnorm[k] == 0 ? 0.0 : status[k] == :failed ? NaN : Float64(beta[k] / bnorm[k]) for k in 1:nb]
    return (X, iterations, residuals, status)
end
