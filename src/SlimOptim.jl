module SlimOptim


using Printf, LinearAlgebra, LineSearches

import LineSearches: BackTracking, HagerZhang, Static, MoreThuente, StrongWolfe
export BackTracking, HagerZhang, Static, MoreThuente, StrongWolfe

export pqn, pqn_options
export spg, spg_options
export bregman, bregman_options
#############################################################################
# Optimization algorithms
include("linesearches.jl")   # minConf_PQN
include("SPGSlim.jl")   # minConf_SPG
include("PQNSlim.jl")   # minConf_PQN
include("bregman.jl")   # minConf_PQN
include("utils.jl") # common functions


function checkls(ls) 
    is_ls = (typeof(ls) <: Union{BackTracking, HagerZhang, Static, MoreThuente, StrongWolfe})
    !is_ls && throw(ArgumentError("Unrecognized linsearch input, only LineSearches.jl's[BackTracking, HagerZhang, Static, MoreThuente, StrongWolfe] are supported"))
end

end # module
