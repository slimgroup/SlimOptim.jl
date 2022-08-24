# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

mutable struct BregmanParams
    verbose
    progTol
    maxIter
    store_trace
    antichatter
    alpha
    spg
    TD
    λfunc
    reset_λ
end

"""
    bregman_options(;verbose=1, optTol=1e-6, progTol=1e-8, maxIter=20,
                    store_trace=false, quantile=.5, alpha=.25, spg=false)

Options structure for the bregman iteration algorithm

# Arguments

- `verbose`: level of verbosity (0: no output, 1: final, 2: iter (default), 3: debug)
- `progTol`: tolerance used to check for lack of progress (default: 1e-9)
- `maxIter`: maximum number of iterations (default: 20)
- `store_trace`: Whether to store the trace/history of x (default: false)
- `antichatter`: Whether to use anti-chatter step length correction
- `alpha`: Strong convexity modulus. (step length is ``α \\frac{||r||_2^2}{||g||_2^2}``)
- `spg`: whether to use spg, default is false
- `TD`: sparsifying transform (e.g. curvelet), default is identity (LinearAlgebra.I)
- `λfunc`: a function to calculate threshold value, default is nothing
- `λ`: a pre-set threshold, will only be used if `λfunc` is not defined, default is nothing
- `quantile`: a percentage to calculate the threshold by quantile of the dual variable in 1st iteration, will only be used if neither `λfunc` nor `λ` are defined, default is .95 i.e thresholds 95% of the vector
- `w`: a weight vector that is applied on the threshold element-wise according to relaxation of weighted l1, default is 1 (no weighting)
- `reset_lambda`: How often should lambda be updated. Defaults to `nothing` i.e lambda is nerver updated and estimated at the first iteration.
"""
function bregman_options(;verbose=1, progTol=1e-8, maxIter=20, store_trace=false, antichatter=true, alpha=.5,
                          spg=false, TD=LinearAlgebra.I, quantile=.95, λ=nothing, λfunc=nothing, w=1,
                          reset_lambda=nothing)
    if isnothing(λfunc)
        if ~isnothing(λ) 
            λfunc = z->λ
        else
            λfunc = z->Statistics.quantile(abs.(z), quantile)
        end
    end
    return BregmanParams(verbose, progTol, maxIter, store_trace, antichatter, alpha, spg, TD, z->w.*λfunc(z), reset_lambda)
end

"""
    bregman(A, x, b, options)

Linearized bregman iteration for the system

``\\frac{1}{2} ||TD \\ x||_2^2 + λ ||TD \\ x||_{1,w}  \\ \\ \\ s.t Ax = b``

For example, for sparsity promoting denoising (i.e LSRTM)

# Required arguments

- `A`: Forward operator (e.g. J or preconditioned J for LSRTM)
- `x`: Initial guess
- `b`: observed data

# Optional Arguments

- `callback` : Callback function. Must take as input a `result` callback(x::result)

# Non-required arguments

- `options`: bregman options, default is bregman_options(); options.TD provides the sparsifying transform (e.g. curvelet), options.w provides the weight vector for the weighted l1

"""
function bregman(A, x::AbstractVector{T1}, b::AbstractVector{T2}, options::BregmanParams=bregman_options(); callback=noop_callback) where {T1<:Number, T2<:Number}
    # residual function wrapper
    function obj(x)
        d = A*x
        fun = .5*norm(d - b)^2
        grad = A'*(d - b)
        return fun, grad
    end
    return bregman(obj, x, options; callback=callback)
end

function bregman(A, TD, x::AbstractVector{T1}, b::AbstractVector{T2}, options::BregmanParams=bregman_options(); callback=noop_callback) where {T1<:Number, T2<:Number}
    @warn "deprecation warning: please put TD in options (BregmanParams) for version > 0.1.7; now overwritting TD in BregmanParams"
    options.TD = TD
    return bregman(A, x, b, options; callback=callback)
end

"""
    bregman(funobj, x, options)

Linearized bregman iteration for the system

``\\frac{1}{2} ||TD \\ x||_2^2 + λ ||TD \\ x||_{1,w}  \\ \\ \\ s.t Ax = b``

# Required arguments

- `funobj`: a function that calculates the objective value (`0.5 * norm(Ax-b)^2`) and the gradient (`A'(Ax-b)`)
- `x`: Initial guess

# Optional Arguments

- `callback` : Callback function. Must take as input a `result` callback(x::result)

# Non-required arguments

- `options`: bregman options, default is bregman_options(); options.TD provides the sparsifying transform (e.g. curvelet), options.w provides the weight vector for the weighted l1

"""
function bregman(funobj::Function, x::AbstractVector{T}, options::BregmanParams=bregman_options(); callback=noop_callback) where {T}
    # Output Parameter Settings
    if options.verbose > 0
        @printf("Running linearized bregman...\n");
        @printf("Progress tolerance: %.2e\n",options.progTol)
        @printf("Maximum number of iterations: %d\n",options.maxIter)
        @printf("Anti-chatter correction: %d\n",options.antichatter)
    end
    # Initialize variables
    z = options.TD*x
    d = similar(z)
    options.spg && (gold = deepcopy(x); xold = deepcopy(x))
    if options.antichatter
        tk = 0 * z
    end

    # Result structure
    sol = breglog(x, z; obj0=l12(0, z))

    # Output Log
    if options.verbose > 0
        @printf("%10s %15s %15s %15s %5s\n","Iteration","Step Length", "L1-2    ", "||A*x - b||_2^2", "λ")
    end

    # Iterations
    for i=1:options.maxIter
        flush(stdout)
        # Compute gradient
        f, g = funobj(x)
        update!(sol; iter=i-1, residual=f)

        # Optional callback at init state
        (i == 1) && callback(sol)

        # Preconditionned ipdate direction
        d .= -(options.TD*g)
        # Step length
        t = (options.spg) ? T(dot(x-xold, x-xold)/dot(x-xold, g-gold)) : T(options.alpha*f/norm(d)^2)
        scale!(d, t)

        # Anti-chatter
        if options.antichatter
            @assert isreal(z) "we currently do not support anti-chatter for complex numbers"
            @. tk = tk - sign(d)
            # Chatter correction after 1st iteration
            if i > 1
                inds_z = findall(abs.(z) .> sol.λ)
                @views d[inds_z] .*= abs.(tk[inds_z])/i
            end
        end
        # Update z variable
        @. z = z + d
        # Get λ at first iteration
        set_λ!(sol, z, options, i, options.reset_λ)
        # Save curent state
        options.spg && (gold .= g; xold .= x)
        # Update x
        x = options.TD'*soft_thresholding(z, sol.λ)
        obj_fun = l12(sol.λ, sol.z)
        progress = norm(x - sol.x)

        # Save history
        update!(sol; iter=i, ϕ=obj_fun, x=x, z=z, g=g, store_trace=options.store_trace)

        # Print log
        (options.verbose > 0) && (@printf("%10d %15.5e %15.5e %15.5e %15.5e \n",i, t, obj_fun, f, maximum(sol.λ)))

        # Optional callback
        callback(sol)

        # Terminate if no progress
        progress < options.progTol && (@printf("Step size below progTol\n"); break;)
    end
    return sol
end

function bregman(funobj::Function, x::AbstractVector{T}, options::BregmanParams, TD; kw...) where {T}
    @warn "deprecation warning: please put TD in options (BregmanParams) for version > 0.1.7; now overwritting TD in BregmanParams"
    options.TD = TD
    return bregman(funobj, x, options; kw...)
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
    (~isnothing(x) && length(r.x_trace) == iter && store_trace) && (push!(r.x_trace, x))
    (~isnothing(z) && length(r.z_trace) == iter && store_trace) && (push!(r.z_trace, z))
    (~isnothing(ϕ) && length(r.ϕ_trace) == iter) && (push!(r.ϕ_trace, ϕ))
    (~isnothing(residual) && length(r.r_trace) == iter) && (push!(r.r_trace, residual))
end

function breglog(init_x, init_z; lambda0=0, f0=0, obj0=0)
    return BregmanIterations(1*init_x, 1*init_z, 0*init_z, f0, lambda0, obj0, [obj0], [], [init_x], [init_z])
end

noop_callback(::BregmanIterations) = nothing
scale!(d, t) = (t == 0 || !isLegal(t)) ? lmul!(1/norm(d)^2, d) : lmul!(abs(t), d)

set_λ!(sol::BregmanIterations, z::AbstractVector{T}, options::BregmanParams, s, ::Nothing) where {T} = (s == 1) ? set_λ!(sol, z, options, s, 1) : nothing
set_λ!(sol::BregmanIterations, z::AbstractVector{T}, options::BregmanParams, s::Integer, rs::Integer) where {T} = (s % rs == 0 || s == 1) ? (sol.λ = abs.(T.(options.λfunc(z)))) : nothing

l12(λ::AbstractVector{T1}, z::AbstractVector{T}) where {T1<:Number, T<:Number} = norm(λ .* z, 1) + .5 * norm(z, 2)^2
l12(λ::T1, z::AbstractVector{T}) where {T1<:Number, T<:Number} = abs(λ) * norm(z, 1) + .5 * norm(z, 2)^2