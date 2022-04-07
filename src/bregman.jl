# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

mutable struct BregmanParams
    verbose
    progTol
    maxIter
    store_trace
    antichatter
    quantile
    lambda
    alpha
    spg
end

"""
    bregman_options(;verbose=1, optTol=1e-6, progTol=1e-8, maxIter=20
                    store_trace=false, linesearch=false, alpha=.25, spg=false)

Options structure for the bregman iteration algorithm

# Arguments

- `verbose`: level of verbosity (0: no output, 1: final, 2: iter (default), 3: debug)
- `progTol`: tolerance used to check for lack of progress (default: 1e-9)
- `maxIter`: maximum number of iterations (default: 20)
- `store_trace`: Whether to store the trace/history of x (default: false)
- `antichatter`: Whether to use anti-chatter step length correction
- `quantile`: Thresholding level as quantile value, (default=.95 i.e thresholds 95% of the vector)
- `lambda`: Thresholding level, (default=nothing then quantile will be used to determine the threshold)
- `alpha`: Strong convexity modulus. (step length is ``α \\frac{||r||_2^2}{||g||_2^2}``)

"""
bregman_options(;verbose=1, progTol=1e-8, maxIter=20, store_trace=false, antichatter=true, quantile=.95, lambda=nothing, alpha=.5, spg=false) =
                BregmanParams(verbose, progTol, maxIter, store_trace, antichatter, quantile, lambda, alpha, spg)

"""
    bregman(A, TD, x, b, options)

Linearized bregman iteration for the system

``\\frac{1}{2} ||TD \\ x||_2^2 + λ ||TD \\ x||_1  \\ \\ \\ s.t Ax = b``

For example, for sparsity promoting denoising (i.e LSRTM)

# Arguments

- `TD`: sparsifying transform (e.g. curvelet), default is nothing (i.e. identity)
- `A`: Forward operator (e.g. J or preconditioned J for LSRTM)
- `b`: observed data
- `x`: Initial guess
- `options`: bregman options
"""
function bregman(A, TD, x::Array{T}, b, options) where {T}
    # residual function wrapper
    function obj(x)
        d = A*x
        fun = .5*norm(d - b)^2
        grad = A'*(d - b)
        return fun, grad
    end
    
    return bregman(obj, x, options, TD)
end

"""
    bregman(fun, TD, x, b, options)

Linearized bregman iteration for the system

``\\frac{1}{2} ||TD \\ x||_2^2 + λ ||TD \\ x||_1  \\ \\ \\ s.t Ax = b``

For example, for sparsity promoting denoising (i.e LSRTM)

# Arguments

- `TD`: transform (default is nothing, i.e. identity)
- `thresholdfunc`: function to obtain threshold λ from the dual variable z in the first iteration (default is nothing, i.e. following the options)
- `fun`: residual function, return the tuple (``f = \\frac{1}{2}||Ax - b||_2``, ``g = A^T(Ax - b)``)
- `x`: Initial guess
- `options`: bregman options
"""
function bregman(funobj::Function, x::AbstractArray{T}, options::BregmanParams; TD=nothing, thresholdfunc=nothing) where {T}
    # Output Parameter Settings
    if options.verbose > 0
        @printf("Running linearized bregman...\n");
        @printf("Progress tolerance: %.2e\n",options.progTol)
        @printf("Maximum number of iterations: %d\n",options.maxIter)
        @printf("Anti-chatter correction: %d\n",options.antichatter)
    end
    isnothing(TD) && (TD = LinearAlgebra.I)
    # Intitalize variables
    z = TD*x
    d = similar(z)
    options.spg && (gold = similar(x); xold=similar(x))
    if options.antichatter
        tk = 0 * z
    end

    # Result structure
    sol = breglog(x, z)
    # Initialize λ
    λ = abs(T(0))

    # Output Log
    if options.verbose > 0
        @printf("%10s %15s %15s %15s %5s\n","Iteration","Step Length", "L1-2    ", "||A*x - b||_2^2", "λ")
    end

    # Iterations
    for i=1:options.maxIter
        f, g = funobj(x)
        # Preconditionned ipdate direction
        d .= -TD*g
        # Step length
        t = (options.spg && i> 1) ? T(dot(x-xold, x-xold)/dot(x-xold, g-gold)) : T(options.alpha*f/norm(d)^2)
        t = abs(t)
        mul!(d, d, t)

        # Anti-chatter
        if options.antichatter
            @. tk = tk - sign(d)
            # Chatter correction
            inds_z = findall(abs.(z) .> λ)
            @views d[inds_z] .*= abs.(tk[inds_z])/i
        end
        # Update z variable
        @. z = z + d
        # Get λ at first iteration
        if i == 1
            if ~isnothing(thresholdfunc)
                λ = thresholdfunc(z)
            elseif ~isnothing(options.lambda)
                λ = options.lambda
            else
                λ = quantile(abs.(z), options.quantile)
            end
            λ = abs.(T.(λ))
            sol.λ = abs.(T.(λ))
        end
        # Save curent state
        options.spg && (gold .= g; xold .= x)
        # Update x
        x = TD'*soft_thresholding(z, λ)

        obj_fun = norm(λ .* z, 1) + .5 * norm(z, 2)^2
        if options.verbose > 0
            if length(λ) == 1
                @printf("%10d %15.5e %15.5e %15.5e %15.5e \n",i, t, obj_fun, f, λ)
            else
                @printf("%10d %15.5e %15.5e %15.5e %5s \n",i, t, obj_fun, f, "vector")
            end
        end
        norm(x - sol.x) < options.progTol && (@printf("Step size below progTol\n"); break;)
        update!(sol; iter=i, ϕ=obj_fun, residual=f, x=x, z=z, g=g, store_trace=options.store_trace)
    end
    return sol
end

# Utility functions
"""
Simplified Quantile from Statistics.jl since we only need simplified version of it.
"""
function quantile(u::AbstractVector, p::Real)
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))
    n = length(u)
    v = sort(u; alg=Base.QuickSort)

    m = 1 - p
    aleph = n*p + oftype(p, m)
    j = clamp(trunc(Int, aleph), 1, n-1)
    γ = clamp(aleph - j, 0, 1)

    n == 1 ? a = v[1] : a = v[j]
    n == 1 ? b = v[1] : b = v[j+1]

    (isfinite(a) && isfinite(b)) ? q = a + γ*(b-a) : q = (1-γ)*a + γ*b
    return q
end


"""
Bregman result structure
"""
mutable struct BregmanIterations
    x
    z
    g
    ϕ
    λ
    residual
    ϕ_trace
    r_trace
    x_trace
    z_trace
end

function update!(r::BregmanIterations; x=nothing, z=nothing, ϕ=nothing, residual=nothing, g=nothing, iter=1, store_trace=false)
    ~isnothing(x) && copyto!(r.x, x)
    ~isnothing(z) && copyto!(r.z, z)
    ~isnothing(ϕ) && (r.ϕ = ϕ)
    ~isnothing(residual) && (r.residual = residual)
    ~isnothing(g) && copyto!(r.g, g)
    (~isnothing(x) && length(r.x_trace) == iter-1 && store_trace) && (push!(r.x_trace, x))
    (~isnothing(z) && length(r.z_trace) == iter-1 && store_trace) && (push!(r.z_trace, z))
    (~isnothing(ϕ) && length(r.ϕ_trace) == iter-1) && (push!(r.ϕ_trace, ϕ))
    (~isnothing(residual) && length(r.r_trace) == iter-1) && (push!(r.r_trace, residual))
end

function breglog(init_x, init_z; lambda0=0, f0=0, obj0=0)
    return BregmanIterations(1*init_x, 1*init_z, 0*init_z, f0, lambda0, obj0, Vector{}(), Vector{}(), Vector{}(), Vector{}())
end
