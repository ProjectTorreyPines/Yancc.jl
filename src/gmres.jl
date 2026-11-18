# Restarted GMRES with right preconditioning (Saad & Schultz 1986) in the flexible form
# (Y. Saad, SIAM J. Sci. Comput. 14, 461 (1993)).

"""
    gmres!(x, A!, b, M!; rtol=1e-8, restart=60, maxiter=300)

Solve `A x = b` starting from the initial guess in `x`. `A!(y, v)` stores `A v` in `y`
and `M!(z, v)` stores the action of the preconditioner `M⁻¹ v` in `z`; the iteration
is on `A M⁻¹ u = b` with `x = M⁻¹ u`, so the residual is that of the original system.
The preconditioned basis vectors `M⁻¹ v_j` are kept and the solution is their combination
(flexible GMRES): `M⁻¹` is then never applied to a combination of basis vectors, which
needs it to be linear to rounding. The preconditioner of `solve_dke_tokamak` is not: its
factorizations of the planes are nearly singular, so its action is linear only to their
condition number times the rounding error, and on fine grids `M⁻¹ (V y)` missed
`Σ y_j M⁻¹ v_j` by more than the tolerance.

Stops when `‖b - A x‖ ≤ rtol ‖b‖`, after `maxiter` iterations, or when the residual
recomputed from `x` no longer follows the one of the Arnoldi recurrence, which is the
accuracy that rounding allows. Returns `(x, iterations, relative_residual, status)`
with the residual recomputed from `x` and `status` one of `:converged`, `:stagnated`
(rounding limit reached) or `:maxiter`.
"""
function gmres!(x::AbstractVector{T}, A!, b::AbstractVector{T}, M!;
    rtol::Real=1e-8, restart::Integer=60, maxiter::Integer=300) where {T<:AbstractFloat}

    n = length(b)
    m = min(restart, maxiter)
    V = Matrix{T}(undef, n, min(m, 16) + 1)   # Krylov basis, grown as needed
    Zp = Matrix{T}(undef, n, min(m, 16))      # preconditioned basis M⁻¹ v_j
    H = zeros(T, m + 1, m)
    cs, sn, g = zeros(T, m), zeros(T, m), zeros(T, m + 1)
    h = zeros(T, m + 1)
    w, z, r = similar(b), similar(b), similar(b)

    bnorm = norm(b)
    bnorm == 0 && return (fill!(x, zero(T)), 0, 0.0, :converged)
    tol = rtol * bnorm
    iterations = 0

    if any(!iszero, x)
        A!(r, x)
        r .= b .- r
    else
        copyto!(r, b)
    end
    beta = norm(r)

    while beta > tol && iterations < maxiter
        @views V[:, 1] .= r ./ beta
        fill!(g, zero(T))
        g[1] = beta
        k = 0
        for j in 1:m
            j + 1 > size(V, 2) && (V = hcat(V, Matrix{T}(undef, n, min(size(V, 2), m + 1 - size(V, 2)))))
            j > size(Zp, 2) && (Zp = hcat(Zp, Matrix{T}(undef, n, min(size(Zp, 2), m - size(Zp, 2)))))
            M!(z, view(V, :, j))
            @views Zp[:, j] .= z
            A!(w, z)
            # classical Gram-Schmidt, applied twice
            Vj = view(V, :, 1:j)
            hj = view(h, 1:j)
            mul!(hj, Vj', w)
            mul!(w, Vj, hj, -one(T), one(T))
            @views H[1:j, j] .= hj
            mul!(hj, Vj', w)
            mul!(w, Vj, hj, -one(T), one(T))
            @views H[1:j, j] .+= hj
            H[j+1, j] = norm(w)
            @views V[:, j+1] .= w ./ H[j+1, j]

            # Givens rotations bringing the Hessenberg matrix to triangular form
            for i in 1:(j-1)
                t = cs[i] * H[i, j] + sn[i] * H[i+1, j]
                H[i+1, j] = -sn[i] * H[i, j] + cs[i] * H[i+1, j]
                H[i, j] = t
            end
            d = hypot(H[j, j], H[j+1, j])
            cs[j], sn[j] = H[j, j] / d, H[j+1, j] / d
            H[j, j] = d
            H[j+1, j] = zero(T)
            g[j+1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            k = j
            iterations += 1
            (abs(g[j+1]) <= tol || iterations >= maxiter) && break
        end

        y = UpperTriangular(view(H, 1:k, 1:k)) \ view(g, 1:k)
        mul!(x, view(Zp, :, 1:k), y, one(T), one(T))
        A!(r, x)
        r .= b .- r
        previous = beta
        beta = norm(r)
        # the recurrence reports convergence, the true residual does not improve
        if abs(g[k+1]) <= tol && beta > max(tol, previous / 2)
            return (x, iterations, beta / bnorm, :stagnated)
        end
    end
    return (x, iterations, beta / bnorm, beta <= tol ? :converged : :maxiter)
end
