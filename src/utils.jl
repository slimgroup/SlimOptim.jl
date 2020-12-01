export isLegal, lbfgsUpdate, lbfgsHvFunc2, ssbin, solveSubProblem, subHv, result, soft_thresholding

mutable struct result
    x
    g
    ϕ
    ϕ_trace
    x_trace
    n_project
    n_ϕeval
    n_geval
end

function update!(r::result; x=nothing, ϕ=nothing, g=nothing, iter=1, store_trace=false)
    ~isnothing(x) && copyto!(r.x, x)
    ~isnothing(ϕ) && (r.ϕ = ϕ)
    ~isnothing(g) && copyto!(r.g, g)
    (~isnothing(x) && length(r.x_trace) == iter-1 && store_trace) && (push!(r.x_trace, deepcopy(x)))
    (~isnothing(ϕ) && length(r.ϕ_trace) == iter-1) && (push!(r.ϕ_trace, ϕ))
end

function result(init_x; ϕ0=0, ϕeval=0, δϕeval=0)
    return result(deepcopy(init_x), 0.0f0*init_x, ϕ0, Vector{}(), Vector{}(), 0, ϕeval, δϕeval)
end

function isLegal(v)
    nv = norm(v)
    return !isnan(nv) && !isinf(nv)
end


function terminate(options, optCond, t, d, ϕ, ϕ_old)
    ~isLegal(ϕ) && return true
    # Check optimality
    if optCond < options.optTol
        options.verbose >= 1 &&  @printf("First-Order Optimality Conditions Below optTol\n")
        return true
    end

    if norm(t*d, Inf) < options.progTol
        options.verbose >= 1 && @printf("Step size below progTol\n")
        return true
    end

    if abs.(ϕ-ϕ_old) < options.progTol
        options.verbose >= 1 && @printf("Function value changing by less than progTol\n")
        return true
    end
    return false
end


"""
    PQN lbfgs functions
"""

function lbfgsUpdate(y, s, corrections, debug, old_dirs, old_stps, Hdiag)
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
    else
        if debug==3
            @printf("Skipping Update\n")
        end
    end
    return old_dirs, old_stps, Hdiag
end

function lbfgsHvFunc2(v, Hdiag, N, M)
    if cond(M)>(1/(eps(Float32)))
        pr =  Array{Float32}(ssbin(M,500))
        L = Diagonal(vec(pr))
        Hv = v/Hdiag - N*L*((L*M*L)\(L*(N'*v)))
    else
        Hv = v/Hdiag - N*(M\(N'*v))
    end
    
    return Hv
end

function ssbin(A, nmv)
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
    n = size(A,1)
    d = ones(Float32,n)
    dp = deepcopy(d)
    tmp = zeros(Float32, n)
    for k = 1:nmv
      # Approximate matrix-vector product
      u = randn(Float32, n)
      s = u ./ sqrt.(dp)
      y = A*s
      # omega^k
      alpha = (k - 1)/nmv
      omega = (1 - alpha)/2 + alpha/nmv
      # Iteration
      d = (1-omega)*d/sum(d) + omega*y.^2/sum(y.^2)
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
    return 1 ./(d.*dp).^(1/4)
end

function solveSubProblem(x, g, H, funProj, options, x_init)
# Uses SPG to solve for projected quasi-Newton direction
    funObj(p) = subHv(p, x, g, H)
    sol = spg(funObj, x_init, funProj, options)
    return sol.x 
end

function subHv(p, x, g, HvFunc)
    d = p - x
    Hd = HvFunc(d)
    f = dot(g, d) + dot(d, Hd) / 2
    g = g + Hd
    return f, g
end

# THresholding
soft_thresholding(x::Array{Complex{vDt}}, λ::vDt) where {vDt} = exp.(angle.(x)im) .* max.(abs.(x) .- convert(vDt, λ), vDt(0))
soft_thresholding(x::Array{Complex{vDt}}, λ::Array{vDt}) where {vDt} = exp.(angle.(x)im) .* max.(abs.(x) .- convert(Array{vDt}, λ), vDt(0))
soft_thresholding(x::Array{vDt}, λ::vDt) where {vDt} = sign.(x) .* max.(abs.(x) .- convert(vDt, λ), vDt(0))
soft_thresholding(x::Array{vDt}, λ::Array{vDt}) where {vDt} = sign.(x) .* max.(abs.(x) .- convert(Array{vDt}, λ), vDt(0))
