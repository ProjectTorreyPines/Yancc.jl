# Finite difference stencils on uniform grids (yancc/_finite_diff.py).

const DEFAULT_P1 = "4d"
const DEFAULT_P2 = 4

# First derivative, forward-biased, reference point at the centre of the stencil.
# The backward version is the mirrored stencil with opposite sign.
# "2d" and "4d" are the widened upwind stencils of Conlin & Landreman (2026), Table 1;
# "2z" and "4z" are centred.
const FD_COEFFS_1 = Dict{String,Vector{Float64}}(
    "1a" => [0, -1, 1],
    "2a" => [0, 0, -3 / 2, 2, -1 / 2],
    "2b" => [0, -1 / 4, -3 / 4, 5 / 4, -1 / 4],
    "2c" => [0, -3 / 8, -3 / 8, 7 / 8, -1 / 8],
    "2d" => [0, 0, 0, 0, -5 / 4, 4 / 3, 0, 0, -1 / 12],
    "2z" => [-1 / 2, 0, 1 / 2],
    "3a" => [0, 0, 0, -11 / 6, 3, -3 / 2, 1 / 3],
    "3b" => [0, -1 / 3, -1 / 2, 1, -1 / 6],
    "3d" => [0, -1 / 10, 0, -5 / 6, 1, 0, -1 / 15],
    "4a" => [0, 0, 0, 0, -25 / 12, 4, -3, 4 / 3, -1 / 4],
    "4b" => [0, 0, -1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12],
    "4d" => [0, 0, -1 / 15, 0, -13 / 12, 4 / 3, 0, -4 / 15, 1 / 12],
    "4z" => [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12],
    "6z" => [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]
)

# Second derivative, centred.
const FD_COEFFS_2 = Dict{Int,Vector{Float64}}(
    2 => [1, -2, 1],
    4 => [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
    6 => [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    8 => [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]
)

"""
    fd_matrix(stencil, n, bc)

`n × n` matrix applying `stencil` (centred on its middle entry) to a function on `n`
equally spaced points. `bc` is `:periodic` or `:symmetric`; the latter mirrors the
function about half a cell beyond each end point (`f[0] = f[1]`, `f[n+1] = f[n]`).
"""
function fd_matrix(stencil::AbstractVector{<:Real}, n::Integer, bc::Symbol)
    @assert bc in (:periodic, :symmetric)
    m = length(stencil)
    npad = min(m, n)
    # index into f of each entry of the padded function
    left = bc == :symmetric ? (npad:-1:1) : ((n-npad+1):n)
    right = bc == :symmetric ? (n:-1:(n-npad+1)) : (1:npad)
    padidx = vcat(collect(left), collect(1:n), collect(right))
    nvalid = length(padidx) - m + 1
    offset = (nvalid - n) ÷ 2
    M = zeros(Float64, n, n)
    for i in 1:n, k in 1:m
        M[i, padidx[i+offset+k-1]] += stencil[k]
    end
    return M
end

"Forward-biased first derivative matrix for stencil `p` and grid spacing `h`."
fdfwd_matrix(p::AbstractString, n::Integer, h::Real, bc::Symbol) = fd_matrix(FD_COEFFS_1[p] ./ h, n, bc)

"Backward-biased first derivative matrix for stencil `p` and grid spacing `h`."
fdbwd_matrix(p::AbstractString, n::Integer, h::Real, bc::Symbol) = fd_matrix(-reverse(FD_COEFFS_1[p]) ./ h, n, bc)

"Centred second derivative matrix of order `p` for grid spacing `h`."
fd2_matrix(p::Integer, n::Integer, h::Real, bc::Symbol) = fd_matrix(FD_COEFFS_2[p] ./ h^2, n, bc)
