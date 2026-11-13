# The two level preconditioner of solve_tokamak.jl for a batch of cases: the coarse
# solve on the low Legendre moments, then the plane problem at each speed and species.
# The matrices of all cases have one sparsity pattern and are assembled together from
# the batched operator (src/batch/assembly_maps.jl), so that a device can build, factor
# and solve them as uniform batches.

"""
    BatchedTokamakPreconditioner(ops, bop, device; moments=4)

[`TokamakPreconditioner`](@ref) with the moment coarse space (at least three moments)
for every case of the batched operator `bop`.
"""
struct BatchedTokamakPreconditioner{PL,CO,D<:NamedTuple,W<:NamedTuple}
    nl::Int
    planes::PL    # factors of the plane matrices, index j + n (b - 1)
    coarse::CO    # factors of the coarse matrices, one per case
    data::D
    work::W
end

"""
    PreconditionerMaps(op, nl, speed, device; coarse_solver, plane_solver)

What the preconditioners of all batches of cases like `op` share: the structure of the
plane and coarse matrices and the maps from the batched operator to their values
(src/batch/assembly_maps.jl), with `nl` coarse moments. `speed` says whether a case has
a radial electric field. The coarse matrices are stored sparse (`coarse_solver =
:sparse`) or as the dense blocks of src/batch/block_coarse.jl (`:block`), and so are
the plane matrices (`plane_solver`).
"""
struct PreconditionerMaps{PL,C,D<:NamedTuple}
    nl::Int
    planes::PL  # AssemblyMap or PlaneBlockMap
    coarse::C   # AssemblyMap or BlockAssemblyMap
    layout::CoarseLayout
    data::D     # projections of the coarse ingredients; prolongation P and restriction W
end

function PreconditionerMaps(op::DKEOperator{Float64}, nl::Integer, speed::Bool, device::Val;
    coarse_solver::Symbol=_default_coarse_solver(device), plane_solver::Symbol=_default_plane_solver(device, op))
    @assert nl >= 3 "the moment coarse space needs at least three moments"
    @assert coarse_solver in (:sparse, :block) && plane_solver in (:sparse, :block)
    na, nt = state_size(op)[1:2]
    plane_blocks = plane_solver == :block ? _plane_block_layout(op) : nothing   # nothing: grid too small
    planes = plane_blocks === nothing ? AssemblyMap(_plane_entries(op)..., device) :
             PlaneBlockMap(_plane_entries(op)[1], na, nt, 4na * nt + 2, plane_blocks, device)
    entries, m, ningredients, layout, projections, P, W = _coarse_entries(op, nl, speed)
    data = map(A -> _to_device(device, A), (; projections..., P, W))
    blocks = coarse_solver == :block ? _coarse_block_layout(op, nl) : nothing   # nothing: grid too small
    coarse = blocks === nothing ? AssemblyMap(entries, m, ningredients, device) :
             BlockAssemblyMap(entries, m, ningredients, blocks, device)
    return PreconditionerMaps(Int(nl), planes, coarse, layout, data)
end

_assemble(map::AssemblyMap, Θ::AbstractMatrix) = UniformSparseBatch(map.m, map.rowptr, map.colval, map.G * Θ)
_assemble(map::BlockAssemblyMap, Θ::AbstractMatrix) = _assemble_blocks(map, Θ)

# the plane matrices (index j + n (b - 1)) and the coarse matrices (one per case) of `bop`
function _assemble_preconditioner(maps::PreconditionerMaps, bop::BatchedDKE)
    planes = _assemble(maps.planes, _plane_ingredients(bop))
    coarse = _assemble(maps.coarse, _coarse_ingredients(bop, maps.layout, maps.data))
    return planes, coarse
end

_factor_coarse(device::Val, coarse::UniformSparseBatch) = _factor_batch(device, coarse)
_factor_coarse(::Val, coarse::NamedTuple) = _factor_blocks(coarse)

function BatchedTokamakPreconditioner(maps::PreconditionerMaps, bop::BatchedDKE, planes, coarse, device::Val)
    na, nt, nx, ns = bop.dims
    B, np, n = bop.nbatch, na * nt, nx * ns
    nl = maps.nl
    nc = nl * nt * n + n + 2ns
    like(dims...) = similar(bop.data.wa, dims...)
    work = (Rf=like(np, n * B), Zf=like(np, n * B), WR=like(nl, nt, n, B), cm=like(n, nl, nt, B),
        rc=like(nc, B), c=like(nc, B), AZ=like(np * n + 2ns, B))
    return BatchedTokamakPreconditioner(nl, _factor_planes(device, planes), _factor_coarse(device, coarse),
        (P=maps.data.P, W=maps.data.W), work)
end

function BatchedTokamakPreconditioner(ops::AbstractVector{<:DKEOperator{Float64}}, bop::BatchedDKE, device::Val; moments::Integer=4,
    coarse_solver::Symbol=_default_coarse_solver(device), plane_solver::Symbol=_default_plane_solver(device, first(ops)))
    maps = PreconditionerMaps(first(ops), min(Int(moments), bop.nL), bop.has_speed, device; coarse_solver, plane_solver)
    return BatchedTokamakPreconditioner(maps, bop, _assemble_preconditioner(maps, bop)..., device)
end

# Z = M⁻¹ R for the bordered operators of `bop`; R is overwritten
function _precondition!(Z::AbstractMatrix, P::BatchedTokamakPreconditioner, R::AbstractMatrix, bop::BatchedDKE)
    na, nt, nx, ns = bop.dims
    B, np, n, nl = bop.nbatch, na * nt, nx * ns, P.nl
    N, nB = np * n, n * B
    ncm = nl * n * nt          # moments; then n surface averages and 2 ns sources
    nc = ncm + n + 2ns
    w = P.work

    # coarse correction: Z = prolongation of the coarse solution, R = R - K Z
    reshape(w.Rf, N, B) .= view(R, 1:N, :)
    mul!(reshape(w.WR, nl, nt * nB), P.data.W, reshape(w.Rf, na, nt * nB))
    permutedims!(w.cm, w.WR, (3, 1, 2, 4))
    fill!(w.rc, 0)
    view(w.rc, 1:ncm, :) .= reshape(w.cm, ncm, B)
    view(w.rc, (ncm+n+1):nc, :) .= view(R, (N+1):(N+2ns), :)
    _solve_batch!(w.c, P.coarse, w.rc)
    reshape(w.cm, ncm, B) .= view(w.c, 1:ncm, :)
    permutedims!(w.WR, w.cm, (2, 3, 1, 4))
    mul!(reshape(w.Zf, na, nt * nB), P.data.P, reshape(w.WR, nl, nt * nB))
    view(Z, 1:N, :) .= reshape(w.Zf, N, B)
    view(Z, (N+1):(N+2ns), :) .= view(w.c, (ncm+n+1):nc, :)
    apply_bordered!(w.AZ, bop, Z)
    R .-= w.AZ

    # plane solves at each speed and species
    reshape(w.Rf, N, B) .= view(R, 1:N, :)
    _solve_batch!(w.Zf, P.planes, w.Rf)
    view(Z, 1:N, :) .+= reshape(w.Zf, N, B)
    return Z
end
