module SlimOptim

using Printf, LinearAlgebra, LineSearches

export spg, pqn, backtracking_linesearch
#############################################################################
# Optimization algorithms
include("SPGSlim.jl")   # minConf_SPG
include("PQNSlim.jl")   # minConf_PQN
include("utils.jl") # common functions

end # module
