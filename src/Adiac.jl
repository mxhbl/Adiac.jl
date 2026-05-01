module Adiac

using LinearAlgebra, StaticArrays, SparseArrays, NautyGraphs, Graphs, QuadGK, Statistics
using Convex, SCS, CDDLib, Polyhedra
using ForwardDiff, LogExpFunctions, NonlinearSolve, StaticArrays
using Roly
using StatsBase, Random, OrdinaryDiffEq
using Integrals

include("utils.jl")
include("structurecollection.jl")
include("yieldcalc.jl")
include("design.jl")
include("polyhedra.jl")
include("energy_functions.jl")
include("entropy.jl")
include("kinetics.jl")

export linear_design, convex_design, convex_multidesign
export logdensities, densities, particle_densitites, logyields, yields, chemical_potentials
export polyform_hessian, entropy
export singleton_sets, count_faces
export monoadd_kinetics
export _simulate_kinetics, stochastic_simulate

end
