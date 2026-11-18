# Throughput of many tokamak solves: one solve per thread against the batched solver.
#
#   julia -t <threads> --project=<env> benchmark/batch_throughput.jl [ncases] [nalpha] [ntheta] [nx] [device] [baseline] [batchsize] [plane_solver]
#
# `device` is `cpu` or `cuda` (needs CUDA and CUDSS in the environment); `baseline` is
# the number of cases solved one per thread for comparison (default: all, 0 for none);
# `batchsize` 0 is the default, and `plane_solver` is `sparse` or `block` (default: the
# one of the device).
# Threads share the garbage collector and the memory bandwidth, so for the throughput of
# a CPU node also run one process per core (`-t 1`, `ncases` small) and add them up. The cases are
# D + C + e on circular flux surfaces that differ in aspect ratio, safety factor,
# density, temperature and radial electric field. Times exclude compilation; the
# construction of the operators, which both paths need, is reported separately.

using LinearAlgebra, Printf, Yancc

ncases = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
nalpha = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 65
ntheta = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 41
nx = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 6
device = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :cpu
nbaseline = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : ncases
batchsize = length(ARGS) >= 7 && ARGS[7] != "0" ? parse(Int, ARGS[7]) : nothing
plane_solver = length(ARGS) >= 8 ? Symbol(ARGS[8]) : nothing
device == :cuda && @eval using CUDA, CUDSS

include(joinpath(@__DIR__, "cases.jl"))

# one solve per thread, each with a single BLAS thread
function one_per_thread(ops)
    sols = Vector{DKESolution{Float64}}(undef, length(ops))
    Threads.@threads for k in eachindex(ops)
        sols[k] = solve_dke_tokamak(ops[k])
    end
    return sols
end

BLAS.set_num_threads(1)
warm = build(min(ncases, 2))
one_per_thread(warm)
Yancc._solve_dke_tokamak_batch(warm; device, plane_solver)

t_build = @elapsed ops = build(ncases)
unknowns = Yancc.system_size(ops[1])
@printf("%d cases, %d × %d × %d × %d = %d unknowns each, %d threads, device %s\n", ncases, nalpha, ntheta, nx,
    length(ops[1].species), unknowns, Threads.nthreads(), device)
@printf("operators            %8.2f s\n", t_build)

single = one_per_thread(ops[1:nbaseline])   # also the reference of the comparison below
if nbaseline > 0
    t_single = @elapsed one_per_thread(ops[1:nbaseline])
    @printf("one solve per thread %8.2f s   %7.1f solves/s   (%d to %d iterations)\n", t_single, nbaseline / t_single,
        extrema(s.iterations for s in single)...)
end

t_batch = @elapsed ((sols, status, seconds) = Yancc._solve_dke_tokamak_batch(ops; device, batchsize, plane_solver))
planes = something(plane_solver, Yancc._default_plane_solver(Val(device), ops[1]))
@printf("planes by %s, batch size %s\n", planes, something(batchsize, Yancc._default_batchsize(Val(device), ncases, unknowns, 80,
    planes == :block ? Yancc._plane_block_layout(ops[1]) : nothing)))
@printf("batched, %-10s  %8.2f s   %7.1f solves/s   (%d to %d iterations)\n", device, t_batch, ncases / t_batch,
    extrema(s.iterations for s in sols)...)
for stage in (:maps, :assemble, :factor, :iterate, :moments)
    @printf("    %-10s %8.2f s\n", stage, seconds[stage])
end
if nbaseline > 0
    worst = maximum(maximum(abs.(sols[k].heat_flux .- single[k].heat_flux)) / maximum(abs.(single[k].heat_flux)) for k in 1:nbaseline)
    @printf("largest difference of the heat fluxes of a case between the two, relative to its largest: %.1e\n", worst)
end
@printf("status: %s\n", join(("$(count(==(st), status)) $st" for st in unique(status)), ", "))
if device == :cuda
    @printf("device memory in use after the solve: %.1f GiB\n", (CUDA.total_memory() - CUDA.free_memory()) / 2^30)
end
