# Throughput of independent tokamak solves in one process with one thread: the unit of
# the CPU baseline. Run one copy per core at the same time, e.g.
#
#   srun -n 128 -c 2 --cpu-bind=cores julia -t 1 --project=<env> benchmark/single_throughput.jl 8 65 41 6
#
# and add up the rates they print. Each copy solves `ncases` cases spread over the same
# set as benchmark/batch_throughput.jl; compilation is excluded.

using LinearAlgebra, Printf, Yancc

ncases = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
nalpha = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 65
ntheta = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 41
nx = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 6
rank, nranks = parse(Int, get(ENV, "SLURM_PROCID", "0")), parse(Int, get(ENV, "SLURM_NTASKS", "1"))

include(joinpath(@__DIR__, "cases.jl"))

BLAS.set_num_threads(1)
solve_dke_tokamak(case(1, 2))
total = ncases * nranks
started = time()
t_build = @elapsed ops = [case(rank + 1 + nranks * (i - 1), total) for i in 1:ncases]
t_solve = @elapsed sols = [solve_dke_tokamak(op) for op in ops]
@printf("rank %d of %d: %d solves of %d unknowns in %.2f s (+ %.2f s operators), %.3f solves/s, started %.1f, %d to %d iterations\n",
    rank, nranks, ncases, Yancc.system_size(ops[1]), t_solve, t_build, ncases / (t_solve + t_build), started,
    extrema(s.iterations for s in sols)...)
