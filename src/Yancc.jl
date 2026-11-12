"""
Yancc.jl: Julia port of yancc (Yet Another NeoClassical Code) by Rory Conlin,
https://github.com/f0uriest/yancc.

R. Conlin and M. Landreman, "yancc: a GPU-accelerated, differentiable solver for
neoclassical transport in tokamaks and stellarators", Plasma Phys. Control. Fusion 68
(2026) 115001, doi:10.1088/1361-6587/aeb628.
"""
module Yancc

using LinearAlgebra
using SparseArrays
using SpecialFunctions: erf

include("finite_diff.jl")
include("velocity_grids.jl")
include("species.jl")
include("field.jl")
include("rosenbluth.jl")
include("collisions.jl")
include("trajectories.jl")
include("operators.jl")
include("solution.jl")
include("solve_reference.jl")
include("gmres.jl")
include("solve_tokamak.jl")

export Species, Electron, Hydrogen, Deuterium, Tritium, Helium4
export LocalMaxwellian, from_scale_lengths
export Field, field_from_boozer, flux_surface_average
export MaxwellSpeedGrid, UniformPitchAngleGrid
export DKEOperator, DKESolution, solve_dke_reference, conductive_heat_flux
export solve_dke_tokamak, solve_dke_tokamak!, TokamakPreconditioner, TokamakCache

end
