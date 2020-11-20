module SlimOptim


using Printf, LinearAlgebra, LineSearches

import LineSearches: BackTracking, HagerZhang, Static, MoreThuente, StrongWolfe
export BackTracking, HagerZhang, Static, MoreThuente, StrongWolfe

export pqn, pqn_options
export spg, spg_options
export bregman, bregman_options
#############################################################################
# Optimization algorithms
include("SPGSlim.jl")   # minConf_SPG
include("PQNSlim.jl")   # minConf_PQN
include("bregman.jl")   # minConf_PQN
include("utils.jl") # common functions

end # module
