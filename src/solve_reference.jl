# Reference solver: dense factorization of the bordered kinetic system.

"""
    bordered_matrix(op, border=DKEBorder(op))

Dense matrix of the bordered system `[A B; C 0]`, built column by column from the
matrix-vector product. For verification and small problems only.
"""
function bordered_matrix(op::DKEOperator{T}, border::DKEBorder=DKEBorder(op)) where {T}
    n = system_size(op)
    K = zeros(T, n, n)
    e = zeros(T, n)
    for j in 1:n
        e[j] = one(T)
        apply_bordered!(view(K, :, j), op, border, e)
        e[j] = zero(T)
    end
    return K
end

"""
    solve_dke_reference(op; EparB=0)

Solve the drift kinetic equation by dense LU of the bordered system. The cost grows
with the cube of the number of unknowns; intended for verification.
"""
function solve_dke_reference(op::DKEOperator{T}; EparB::Real=0.0) where {T}
    border = DKEBorder(op)
    K = bordered_matrix(op, border)
    b = zeros(T, system_size(op))
    b[1:prod(state_size(op))] .= vec(dke_rhs(op, EparB))
    return DKESolution(op, K \ b)
end
