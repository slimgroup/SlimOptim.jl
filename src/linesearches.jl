export linesearch

"""
Line search interface
"""
function linesearch(ls, sol, d, f, g!, fg!, t, funRef, gtd, gvec)
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

    return ls(ϕ, dϕ, ϕdϕ, t, funRef, gtd)
end