# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

export isLegal, lbfgsUpdate, lbfgsHvFunc2, ssbin, solveSubProblem!, subHv, result, soft_thresholding

mutable struct result{T}
    x::AbstractArray{T}
    g::AbstractArray{T}
    ϕ::T
    ϕ_trace::Vector{T}
    x_trace::Vector{<:AbstractArray{T}}
    n_project::Integer
    n_ϕeval::Integer
    n_geval::Integer
end

function update!(r::result; x=nothing, ϕ=nothing, g=nothing, iter=1, store_trace=false)
    ~isnothing(x) && copyto!(r.x, x)
    ~isnothing(ϕ) && (r.ϕ = ϕ)
    ~isnothing(g) && copyto!(r.g, g)
    (~isnothing(x) && length(r.x_trace) == iter && store_trace) && (push!(r.x_trace, deepcopy(x)))
    (~isnothing(ϕ) && length(r.ϕ_trace) == iter) && (push!(r.ϕ_trace, ϕ))
end

function result(init_x::AbstractArray{T}; ϕ0=0, ϕeval=0, δϕeval=0) where T
    return result(deepcopy(init_x), T(0)*init_x, T(ϕ0), Vector{T}(), [init_x], 0, ϕeval, δϕeval)
end

noop_callback(::result) = nothing

function isLegal(v::AbstractArray{T}) where T
    nv = norm(v)
    return !isnan(nv) && !isinf(nv)
end

isLegal(v::T) where T = (!isnan(v) && !isinf(v))

function terminate(options, optCond, t, d, ϕ, ϕ_old)
    ~isLegal(ϕ) && return true
    ~isLegal(d) && return true
    # Check optimality
    if optCond < options.optTol
        options.verbose >= 1 &&  @printf("First-Order Optimality Conditions %2.2e Below optTol %2.2e\n",
                                         optCond, options.optTol)
        return true
    end

    if norm(t*d, Inf) < options.progTol
        options.verbose >= 1 && @printf("Step size: %2.2e below progTol: %2.2e\n",
                                        norm(t*d, Inf), options.progTol)
        return true
    end

    if norm(d, Inf) < options.progTol
        options.verbose >= 1 && @printf("Update direction smaller than progTol (%2.2e < %2.2e)\n",
                                         norm(d, Inf), options.progTol)
        return true
    end
    t == 0 && return true
    return false
end


"""
    PQN lbfgs functions
"""
function lbfgsUpdate(y::AbstractArray{T}, s::AbstractArray{T}, corrections::Integer,
                     old_dirs::AbstractArray{T, 2}, old_stps::AbstractArray{T, 2},
                     Hdiag) where T
    ys = dot(y,s)
    if ys > 1e-10 || size(old_dirs,2)==0
        numCorrections = size(old_dirs, 2)
        if numCorrections < corrections
            # Full Update
            old_dirs = [old_dirs s]
            old_stps = [old_stps y]
        else
            # Limited-Memory Update
            old_dirs = [old_dirs[:, 2:corrections] s] 
            old_stps = [old_stps[:, 2:corrections] y]
        end

        # Update scale of initial Hessian approximation
        Hdiag = ys/dot(y, y)
    end
    return old_dirs, old_stps, Hdiag
end

function lbfgsHvFunc2(v::AbstractArray{T}, Hdiag, N::AbstractArray{T, 2}, M::AbstractArray{T, 2}) where T
    if cond(M)>(1/(eps(T)))
        pr = ssbin(M, 500)
        L = Diagonal(vec(pr))
        Hv = v/Hdiag - N*L*((L*M*L)\(L*(N'*v)))
    else
        Hv = v/Hdiag - N*(M\(N'*v))
    end

    return Hv
end

function ssbin(A::AbstractArray{T, N}, nmv) where {T, N}
    # Stochastic matrix-free binormalization for symmetric real A.
    # x = ssbin(A,nmv,n)
    #   A is a symmetric real matrix or function handle. If it is a
    #     function handle, then v = A(x) returns A*x.
    #   nmv is the number of matrix-vector products to perform.
    #   [n] is the size of the matrix. It is necessary to specify n
    #     only if A is a function handle.
    #   diag(x) A diag(x) is approximately binormalized.

    # Jan 2010. Algorithm and code by Andrew M. Bradley (ambrad@stanford.edu).
    # Aug 2010. Modified to record and use dp. New omega schedule after running
    #   some tests.
    # Jul 2011. New strategy to deal with reducible matrices: Use the old
    #   iteration in the early iterations; then switch to snbin-like behavior,
    #   which deals properly with oscillation.
    n = size(A, 1)
    d = ones(T, n)
    dp = deepcopy(d)
    tmp = zeros(T, n)
    for k = 1:nmv
      # Approximate matrix-vector product
      u = randn(T, n)
      s = u ./ sqrt.(dp)
      y = A*s
      # omega^k
      alpha = (k - 1)/nmv
      omega = (1 - alpha)/2 + alpha/nmv
      # Iteration
      d .= (1-omega)*d/sum(d) + omega*y.^2/sum(y.^2)
      if (k < min(32,floor(nmv/2)))
        # First focus on making d a decent approximation
        dp .= d
      else
        # This block makes ssbin behave like snbin except for omega
        tmp .= dp
        dp .= d
        d .= tmp
      end
    end

    return 1 ./(d.*dp).^(T(.25))
end

function solveSubProblem!(p::AbstractArray{T}, x::AbstractArray{T}, g::AbstractArray{T}, H,
                         funProj::Function, options, x_init::AbstractArray{T}) where T
    # Uses SPG to solve for projected quasi-Newton direction
    funObj(γ) = subHv(γ, x[1:end], g[1:end], H)
    sol = spg(funObj, x_init, funProj, options)
    p .= sol.x
end

function subHv(p::AbstractArray{T}, x::AbstractArray{T}, g::AbstractArray{T}, HvFunc::Function) where T
    d = p - x
    Hd = HvFunc(d)
    f = dot(g, d) + dot(d, Hd) / 2
    g = g + Hd
    return f, g
end

# THresholding
soft_thresholding(x::AbstractArray{Complex{T}}, λ::Union{T, Array{T}}) where {T} = exp.(angle.(x)im) .* max.(abs.(x) .- λ, T(0))
soft_thresholding(x::AbstractArray{T}, λ::Union{Array{T}, T}) where {T} = sign.(x) .* max.(abs.(x) .- λ, T(0))