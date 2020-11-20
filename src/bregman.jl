mutable struct BregmanParams
    verbose
    progTol
    maxIter
    store_trace
    antichatter
    quantile
end

"""
    bregman_options(;verbose=1, optTol=1e-6, progTol=1e-8, maxIter=20
                    store_trace=false, linesearch=false)

Options structure for the bregman iteration algorithm

    * verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3: debug)
    * optTol: tolerance used to check for optimality (default: 1e-5)
    * progTol: tolerance used to check for lack of progress (default: 1e-9)
    * maxIter: maximum number of iterations (default: 20)
    * store_trace: Whether to store the trace/history of x (default: false)
    * antichatter: Whether to use anti-chatter step length correction
    * quantile: Thresholding level as quantile value, (default=.95)
"""

bregman_options(;verbose=1, progTol=1e-8, maxIter=20, store_trace=false, antichatter=true, quantile=.95) =
                BregmanParams(verbose, progTol, maxIter, store_trace, antichatter, quantile)

"""
    bregman(A, TD, x, b, options)

Linearized bregman iteration for the system

    ||TD*x||_1 + λ ||x||_2   s.t A*x = b

For example, for sparsity promoting denoising (i.e LSRTM)
    * TD: curvelet transform
    * A: Forward operator (J or preconditioned J for LSRTM)
    * b: observed data
    * x: Initial guess
"""

function bregman(A, TD, x::AbstractArray{vDt}, b, options) where {vDt}
    # Objective function wrapper
    function obj(x)
        d = A*x
        fun = .5*norm(d - b)^2
        grad = A'*(d - b)
        return fun, grad
    end
    
    return bregman(obj, TD, x, options)
end

"""
    bregman(fun, TD, x, b, options)

Linearized bregman iteration for the system

    .5 * ||TD*x||_2^2 + λ ||x||_1   s.t A*x = b

For example, for sparsity promoting denoising (i.e LSRTM)
    * TD: curvelet transform
    * fun: objective function, return the tuple (||A*x - b||_2, A'*(A*x - b))
    * b: observed data
    * x: Initial guess
"""

function bregman(funobj::Function, x::AbstractArray{vDt}, options, TD=nothing) where {vDt}
    # Output Parameter Settings
    if options.verbose > 0
        @printf("Running linearized bregman...\n");
        @printf("Progress tolerance: %.2e\n",options.progTol)
        @printf("Maximum number of iterations: %d\n",options.maxIter)
        @printf("Anti-chatter correction: %d\n",options.antichatter)
    end
    isnothing(TD) && (TD = Matrix{vDt}(I, length(x), length(x)))
    # Intitalize variables
    z = zeros(vDt, size(TD, 1))
    d = zeros(vDt, size(TD, 1))
    if options.antichatter
        tk = zeros(vDt, size(z, 1))
    end

    # Result structure
    sol = breglog(x, z)
    # Initialize λ
    λ = vDt(0)

    # Output Log
    if options.verbose > λ
        @printf("%10s %15s %15s %15s %5s\n","Iteration","Step Length", "Bregman objective", "||A*x - b||_2^2", "λ")
    end

    # Iterations
    for i=1:options.maxIter
        f, g = funobj(x)
        # Preconditionned ipdate direction
        d .= -TD*g
        # Step length
        t = vDt(.5*f/norm(d)^2)

        # Anti-chatter
        if options.antichatter
            tk[:] .+= sign.(d)
            # Chatter correction
            inds_z = findall(abs.(z) .> λ)
            mul!(d, d, t)
            d[inds_z] .*= abs.(tk[inds_z])/i
        end
        # Update dual variable
        @. z = z + d
        # Get λ at first iteration
        i%10 == 1  && (λ = quantile(abs.(z), options.quantile))
        # Update x
        x = TD'*soft_thresholding(z, λ)

        obj_fun = λ * norm(z, 1) + .5 * norm(z, 2)^2
        if options.verbose > 0
            @printf("%10d %15.5e %15.5e %15.5e %15.5e \n",i, t, obj_fun, f, λ)
        end
        norm(x - sol.sol) < options.progTol && (@printf("Step size below progTol\n"); break;)
        update!(sol; iter=i, misfit=f, objective=obj_fun, sol=x, dual=z, gradient=g, store_trace=options.store_trace)
    end
    return sol
end

# Utility functions
"""
Quantile from Statistics.jl since nly need this one
"""
function quantile(v::AbstractVector, p::Real; alpha::Real=1.0, beta::Real=alpha)
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))
    0 <= alpha <= 1 || throw(ArgumentError("alpha parameter out of [0,1] range"))
    0 <= beta <= 1 || throw(ArgumentError("beta parameter out of [0,1] range"))

    n = length(v)
    
    m = alpha + p * (one(alpha) - alpha - beta)
    aleph = n*p + oftype(p, m)
    j = clamp(trunc(Int, aleph), 1, n-1)
    γ = clamp(aleph - j, 0, 1)

    if n == 1
        a = v[1]
        b = v[1]
    else
        a = v[j]
        b = v[j + 1]
    end
    
    if isfinite(a) && isfinite(b)
        return a + γ*(b-a)
    else
        return (1-γ)*a + γ*b
    end
end


"""
Bregman result structure
"""

mutable struct BregmanIterations
    sol
    dual
    gradient
    misfit
    objective
    f_trace
    o_trace
    x_trace
    z_trace
end

function update!(r::BregmanIterations; sol=nothing, dual=nothing, misfit=nothing, objective=nothing, gradient=nothing, iter=1, store_trace=false)
    ~isnothing(sol) && copyto!(r.sol, sol)
    ~isnothing(dual) && copyto!(r.dual, dual)
    ~isnothing(misfit) && (r.misfit = misfit)
    ~isnothing(objective) && (r.objective = objective)
    ~isnothing(gradient) && copyto!(r.gradient, gradient)
    (~isnothing(sol) && length(r.x_trace) == iter-1 && store_trace) && (push!(r.x_trace, sol))
    (~isnothing(dual) && length(r.z_trace) == iter-1 && store_trace) && (push!(r.z_trace, dual))
    (~isnothing(misfit) && length(r.f_trace) == iter-1) && (push!(r.f_trace, misfit))
    (~isnothing(objective) && length(r.o_trace) == iter-1) && (push!(r.o_trace, objective))
end

function breglog(init_x, init_z; f0=0, obj0=0)
    return BregmanIterations(init_x, init_z, 0.0f0*init_z, f0, obj0, Vector{}(), Vector{}(), Vector{}(), Vector{}())
end
