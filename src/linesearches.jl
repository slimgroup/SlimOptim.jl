export linesearch

"""
    linesearch(ls, sol, d, f, g!, fg!, t, funRef, gtd, gvec)

Line search interface to LineSearches.jl

# Arguments

- `ls`: Line search structure (see LineSearches.jl documentation)
- `f`: Objective function, x -> f(x)
- `g!``: Gradient in place function, x-> (g.= gradient(x))
- `fg!``: Objective and in place gradient function, x-> (f = f(x); g.= gradient(x))
- `t`: Initial steplength gess
- `funRef`: Reference objective function value
- `gtd`: Reference direction inner product dot(g0, d0)
- `gvec`: prealocated array for thhe gradient

"""
function linesearch(ls, sol::result, d::Array{T}, f::Function, g!::Function, fg!::Function,
                    t::T, funRef::T, gtd::T, gvec::AbstractArray{T}) where T
    # Univariate line search functions
    ϕ(α) = f(sol.x .+ α.*d)

    function dϕ(α)
        g!(gvec, sol.x .+ α.*d)
        return dot(gvec, d)
    end

    function ϕdϕ(α)
        phi = fg!(gvec, sol.x .+ α.*d)
        dphi = dot(gvec, d)
        return (phi, dphi)
    end

    # Line search. Prevents it to throw error.
    try
        return ls(ϕ, dϕ, ϕdϕ, t, funRef, gtd)
    catch e
        @info "Line search failed"
        return 0, funRef
    end
end
