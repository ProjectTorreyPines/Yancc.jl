# Yancc.jl

Yancc.jl is a Julia port of [**yancc**](https://github.com/f0uriest/yancc) ("Yet Another
NeoClassical Code") by **Rory Conlin**: a radially local drift kinetic solver for
neoclassical flows and transport fluxes in tokamaks and stellarators, with the full
linearized Fokker-Planck collision operator between any number of species.

The physics model, the discretization and the verification cases are the work of Rory
Conlin and Matt Landreman. This package reimplements them in Julia, checks the result
against the original term by term, and adds a solver specialized to axisymmetric fields
so that the model can be used inside the transport loop of
[FUSE](https://github.com/ProjectTorreyPines/FUSE.jl). The original code
([repository](https://github.com/f0uriest/yancc),
[documentation](https://yancc.readthedocs.io/)) remains the reference implementation:
it is the one to use for stellarators, for differentiation, and for anything not listed as ported
below.

## Citing

If you use Yancc.jl, please cite the paper that introduced the method:

> R. Conlin and M. Landreman, "yancc: a GPU-accelerated, differentiable solver for
> neoclassical transport in tokamaks and stellarators",
> *Plasma Phys. Control. Fusion* **68**, 115001 (2026).
> [doi:10.1088/1361-6587/aeb628](https://doi.org/10.1088/1361-6587/aeb628)

```bibtex
@article{Conlin2026yancc,
  author  = {Conlin, Rory and Landreman, Matt},
  title   = {yancc: a {GPU}-accelerated, differentiable solver for neoclassical
             transport in tokamaks and stellarators},
  journal = {Plasma Physics and Controlled Fusion},
  volume  = {68},
  number  = {11},
  pages   = {115001},
  year    = {2026},
  doi     = {10.1088/1361-6587/aeb628}
}
```

## What is ported

| Part of yancc | Yancc.jl |
|---|---|
| Speed grid (Maxwell polynomials), pitch angle grid, finite difference stencils | yes |
| Species, Maxwellians, collision frequencies, Coulomb logarithms | yes |
| Field on a flux surface from arrays or Boozer spectra | yes |
| Field loaders for VMEC, DESC, BOOZ_XFORM | no (pass the arrays) |
| Drift kinetic operator: streaming, mirror force, E×B and speed terms | yes, 2-D and 3-D |
| Fokker-Planck operator: pitch angle and energy scattering, field particle part with Rosenbluth potentials, background species | yes |
| Sources, constraints, drive, fluxes, flows and bootstrap current | yes |
| Dense reference solver | yes (`solve_dke_reference`), for small problems in 2-D and 3-D |
| Solver for axisymmetric fields | yes (`solve_dke_tokamak`), specific to Yancc.jl, see below |
| GCROT Krylov solver with geometric multigrid, for stellarators | not yet |
| Monoenergetic equation and transport coefficients | not yet |
| GPU | batches of tokamak solves (`solve_dke_tokamak_batch`), specific to Yancc.jl, see below |
| Ambipolar electric field, automatic differentiation | not yet |

## Example

One flux surface of a large aspect ratio tokamak with deuterium, carbon and electrons:

```julia
using Yancc

R0, a, B0, q, rho = 1.7, 1.0, 2.0, 2.0, 0.5
ntheta = 41
theta = range(0, 2π; length=ntheta + 1)[1:ntheta]
Bmag = reshape(B0 ./ (1 .+ rho * a / R0 .* cos.(theta)), ntheta, 1)
field = field_from_boozer(rho, Bmag, 0.0, B0 * R0, 1 / q, π * a^2 * B0; R_major=R0, a_minor=a)

# temperature [eV], density [m⁻³], -dln(T)/dρ, -dln(n)/dρ
Carbon = Species(12, 6)
species = [
    from_scale_lengths(Deuterium, 2e3, 4.0e19, 3.0, 1.0),
    from_scale_lengths(Carbon, 2e3, 0.17e19, 3.0, 1.0),
    from_scale_lengths(Electron, 2e3, 5.0e19, 3.0, 1.0)
]

op = DKEOperator(field, UniformPitchAngleGrid(65), MaxwellSpeedGrid(6), species; p1="4z")
sol = solve_dke_tokamak(op)

sol.particle_flux   # ⟨Γ·∇ρ⟩ of each species [m⁻³ s⁻¹]
sol.heat_flux       # energy flux ⟨Q·∇ρ⟩ [W m⁻³]
sol.JparB           # bootstrap current ⟨J∥B⟩ [A T m⁻²]
```

All fluxes are per unit of the radial label `ρ` of the field, and the profile gradients
are derivatives in the same `ρ`; multiply by `dV/dρ` for the flow through the surface.
`heat_flux` is the total energy flux, `conductive_heat_flux(op, sol)` subtracts the
convective part.

## Differences from the Python code

**Solver for axisymmetric fields.** The Python code solves every case with a GCROT
Krylov method preconditioned by geometric multigrid. `solve_dke_tokamak` instead uses
GMRES with a two level preconditioner: a sparse coarse solve for the first four Legendre
moments in pitch angle at every poloidal point, speed and species, which carry the
conserved quantities and the coupling between speeds and species that dominates at
high collisionality, followed by an exact sparse LU of the two dimensional (pitch
angle, poloidal angle) problem at each speed and species. The iteration count is then
nearly independent of collisionality (6 to 9 for `rtol = 1e-6` from ν* = 10⁻³ to 10²).

**Centred differences.** The direct solve of the plane problems does not rely on the
diagonal dominance that multigrid smoothers need, which makes the centred stencil
`p1 = "4z"` (an option of yancc as well, though its multigrid solver does not converge
with it) practical. It reaches a given accuracy on a much coarser grid than the default
upwind stencil `"4d"`, and is recommended with `solve_dke_tokamak`. The defaults of
`DKEOperator` are those of yancc.

**Residual norm.** `rtol` of `solve_dke_tokamak` is measured in a norm that weighs all
species alike, so its values are not those of the Python `rtol`. At low collisionality
the fluxes are more sensitive to the residual than its size suggests, in either code;
the comparisons below use tolerances of 10⁻⁹ to 10⁻¹⁰ in both, tighter than the 10⁻⁸
of the benchmarks in the paper.

## Verification

Every stage of the port is tested against reference data written by the Python code
(`test/golden/generate.py`, yancc v0.0.3):

| Test | Agreement |
|---|---|
| Grids, stencils, collision frequencies | 1e-12 |
| Each term of the operator applied to a test state, 3-D field with and without electric field, tokamak with three species and a background species | 1e-8 to 1e-11 for each species |
| Fluxes, flows and bootstrap current of the dense solve, same cases | 1e-8 |
| `solve_dke_tokamak` against the dense solve | 1e-7 |
| `solve_dke_tokamak` against yancc at working resolution (D + C + e, 48 000 unknowns) | 1e-6 |
| DSHAPE tokamak, electrons and hydrogen with a radial electric field, against yancc (138 000 unknowns) | 1e-5 |
| The same case against SFINCS, at the two radii of the yancc test | within 2 % |
| Reversal of the toroidal field and of the plasma current; toroidally resolved grid for an axisymmetric field | 1e-7 |
| `solve_dke_tokamak` against the dense solve from the banana to the Pfirsch-Schlüter regime (ν*ₑ ≈ 0.1 to 10³) | 1e-6 (1e-4 at 10³, where rounding limits both) |
| Single species in a circular tokamak, ν* = 10⁻² to 10²: no particle flux, no fluxes from a density gradient, diamagnetic parallel flow | 1e-3 |
| The same case, ion heat flux against the Chang-Hinton formula | within 8 % |

## Resolution and cost for tokamaks

Fluxes relative to a converged solution for a circular surface with D + C + e
(`Zeff = 2`), with `p1 = "4z"` and `nx = 6` speeds; time for one cold solve to
`rtol = 1e-6` on one core (operator and preconditioner included):

| `nalpha` × `ntheta` | ν*ₑ = 10⁻¹ | ν*ₑ = 10⁻² | ν*ₑ = 10⁻³ | time |
|---|---|---|---|---|
| 65 × 25 | 3 % | 6 % | 10 % | 0.3 s |
| 65 × 41 | 1 % | 3 % | 4 % | 0.7 s |
| 97 × 41 | 1 % | 3 % | 4 % | 0.9 s |
| 257 × 61 | < 1 % | 1 % | 1 % | 4 s |

(inverse aspect ratio 0.3; at 0.1 the pitch angle grid matters more and `nalpha = 97`
is needed for the same accuracy). The upwind stencil `"4d"` needs about `ntheta = 81`
and `nalpha = 129` for the accuracy of the second row.

The cost grows with the number of species (about 1.2 s for four and 2 s for five at
65 × 41, mostly the coarse factorization). Between the iterations of a transport
solver, `solve_dke_tokamak!` with a `TokamakCache` reuses the preconditioner and starts
from the previous solution. For inputs within 10⁻⁵ of the last full solve it solves only
for the change, so that the fluxes vary smoothly and a transport solver can
differentiate them by finite differences without tightening `rtol`.

**High collisionality.** The speed grid has to grow with collisionality: the parallel
flow, which collisions determine only through small differences, is 4 % off at
ν* = 10² with `nx = 6` and 0.1 % off with `nx = 10`. Beyond ν* ≈ 10³ the system is too
ill conditioned for double precision; `solve_dke_tokamak` warns when rounding stops
the residual above 10⁻⁶. Both are properties of the discretization, not of the solver.

## Many solves at once, on a GPU

`solve_dke_tokamak_batch` solves many flux surfaces or scan points together, as needed
for profiles and for databases:

```julia
using Yancc, CUDA, CUDSS          # without CUDA and CUDSS: device = :cpu

ops = [DKEOperator(field, UniformPitchAngleGrid(65), MaxwellSpeedGrid(6), species; p1="4z")
       for (field, species) in cases]
sols = solve_dke_tokamak_batch(ops; device=:cuda)    # one DKESolution per case
```

The cases must share the grids, the stencils and the options of the operator; the
field, the plasma and the electric field may differ. The method is that of
`solve_dke_tokamak`, with all cases iterated together and the matrices of the
preconditioner built, factored and solved as batches on the device: the coarse
matrices, and the plane matrices from 2500 unknowns (`nalpha × ntheta`) up, by a dense
block elimination in batched cuBLAS kernels, smaller plane matrices by cuDSS, which is
as fast for them. The operators and the fluxes are computed on the host, on all Julia
threads. Cases are solved in pieces that fit in the memory of the device (4 to 6 kB per
unknown and case; `batchsize` sets the size), and a case whose batched factorization
fails is solved again on the host.

CUDA.jl and CUDSS.jl are weak dependencies: they are loaded only when the user loads
them, and only NVIDIA GPUs are supported. A solve on its own does not gain from a GPU,
and on the host one `solve_dke_tokamak` per process and core is faster than
`device = :cpu`, which is there for testing.

Solves per second for D + C + e with `nx = 6`, to `rtol = 1e-6`, on a Perlmutter GPU
node (four A100 with 40 GB, one process per GPU) and on a CPU node (two EPYC 7763, one
single threaded process per core, 128 in all):

| `nalpha` × `ntheta` | unknowns | one A100 | four A100 | CPU node |
|---|---|---|---|---|
| 65 × 41 | 48 000 | 163 | 644 | 69 |
| 257 × 61 | 282 000 | 18.0 | 72 | 17 |

(1024 and 128 cases per GPU; the cases differ in aspect ratio, safety factor, density,
temperature and radial electric field and take 6 to 7 iterations each.) The GPU numbers
leave out the construction of the operators on the host, which takes 1.4 s for 1024
cases on 32 threads and lowers the first row to about 133 per GPU. With cuDSS for the
plane matrices instead of blocks the second row is about half as fast.
`benchmark/batch_throughput.jl` and `benchmark/single_throughput.jl` reproduce the
table; the GPU tests run with `YANCC_TEST_GPU=1`.

Two processes per GPU under CUDA MPS raise the node throughput to 743 and 80 solves
per second for the two rows. Start `nvidia-cuda-mps-control -d` before the processes,
set `JULIA_CUDA_USE_COMPAT=false`, and give each process a `batchsize` of about 0.8
times the single-process default.

## Use in FUSE

`NeoclassicalTransport.yancc` builds the Miller flux surface and the species
from an IMAS data dictionary and returns the fluxes in gyro-Bohm units; in FUSE it is
selected with `act.ActorNeoclassical.model = :yancc`. At the default resolution
(6 × 65 × 41) the energy fluxes of an ITER baseline case are 2 to 3 % above the
converged values at mid radius and 11 % above in the collisionless core, where
`nalpha = 129` brings them within 4 %.

## License

MIT, see `LICENSE`, which retains the copyright notice of yancc, and `NOTICE.md`.
