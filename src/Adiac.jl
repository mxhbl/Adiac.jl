module Adiac


using LinearAlgebra, StaticArrays, Graphs, QuadGK, Statistics
using Convex, SCS, CDDLib, Polyhedra
using ForwardDiff, LogExpFunctions, NonlinearSolve
using Roly

include("utils.jl")
include("yieldcalc.jl")
include("design.jl")
include("polyhedra.jl")
include("energy_functions.jl")
include("entropy.jl")

export linear_design, convex_design, convex_multidesign
export logdensities, densities, monomer_densities, logyields, yields, μs_of_ϕs
export composition, compositions
export polyform_hessian, entropy
export singleton_sets, count_faces

end
