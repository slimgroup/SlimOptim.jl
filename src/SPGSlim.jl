# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

mutable struct SPG_params
    verbose::Integer
    optTol::Real
    progTol::Real
    maxIter::Integer
    suffDec::Real
    memory::Integer
    useSpectral::Bool
    curvilinear::Bool
    feasibleInit::Bool
    testOpt::Bool
    bbType::Bool
    testInit::Bool
    store_trace::Bool
    optNorm::Union{Real, Integer}
    iniStep::Real
    maxLinesearchIter::Integer
end
"""
    spg_options(;verbose=3,optTol=1f-5,progTol=1f-7,
                maxIter=20,suffDec=1f-4,memory=2,
                useSpectral=true,curvilinear=false,
                feasibleInit=false,testOpt=true,
                bbType=true,testInit=false, store_trace=false,
                optNorm=Inf,iniStep=1f0, maxLinesearchIter=10)

Options structure for Spectral Project Gradient algorithm.

# Arguments

- `verbose`: level of verbosity (0: no output, 1: iter (default))
- `optTol`: tolerance used to check for optimality (default: 1e-5)
- `progTol`: tolerance used to check for lack of progress (default: 1e-9)
- `maxIter`: maximum number of iterations (default: 20)
- `suffDec`: sufficient decrease parameter in Armijo condition (default: 1e-4)
- `memory`: number of steps to look back in non-monotone Armijo condition
- `useSpectral`: use spectral scaling of gradient direction (default: 1)
- `curvilinear`: backtrack along projection Arc (default: 0)
- `testOpt`: test optimality condition (default: 1)
- `feasibleInit`: if 1, then the initial point is assumed to be feasible
- `bbType`: type of Barzilai Borwein step (default: 1)
- `testInit`: Whether to test the initial estimate for optimality (default: false)
- `store_trace`: Whether to store the trace/history of x (default: false)
- `optNorm`: First-Order Optimality Conditions norm (default: Inf)
- `iniStep`: Initial step length estimate (default: 1)
- `maxLinesearchIter`: Maximum number of line search iteration (default: 20)
"""
function spg_options(;verbose=1,optTol=1f-10,progTol=1f-10,
                     maxIter=20,suffDec=1f-8,memory=2,
                     useSpectral=true,curvilinear=false,
                     feasibleInit=false,testOpt=true,
                     bbType=true,testInit=false, store_trace=false,
                     optNorm=Inf,iniStep=1, maxLinesearchIter=20)
    return SPG_params(verbose,optTol,progTol,
                      Int64(maxIter),suffDec,memory,
                      useSpectral,curvilinear,
                      feasibleInit,testOpt, bbType,testInit, store_trace,
                      optNorm,iniStep, Int64(maxLinesearchIter))
end


"""
    spg(funObj, x, funProj, options; ls=nothing, callback=nothing)

Function for using Spectral Projected Gradient to solve problems of the form
  min funObj(x) s.t. x in C

# Arguments

- `funObj(x)`:function to minimize (returns gradient as second argument)
- `funProj(x)`: function that returns projection of x onto C
- `x`: Initial guess
- `options`: spg_options structure

# Optional Arguments
- `ls` `: User provided linesearch function
- `callback` : Callback function. Must take as input a `result` callback(x::result)

# Notes:

- if the projection is expensive to compute, you can reduce the
  number of projections by setting testOpt to 0 in the options

- Adapted fromt he matlab implementation of minConf_SPG
"""
function spg(funObj::Function, x::AbstractArray{T}, funProj::Function,
             options::SPG_params=spg_options(); ls=nothing, callback=noop_callback) where {T}
    # Result structure
    sol = result(x)
    # Initialize array for gradient
    G = similar(x)

    # Setup Function to track number of evaluations
    projection(x) = (sol.n_project +=1; return funProj(x))
    grad!(g, x) = (sol.n_ϕeval +=1; sol.n_geval +=1 ; g .= funObj(x)[2])
    objgrad!(g, x) = (sol.n_ϕeval +=1; sol.n_geval +=1 ;(obj, g0) = funObj(x); g .= g0; return obj)
    obj(x) = objgrad!(G, x)

    # Solve optimization
    return _spg(obj, grad!, objgrad!, projection, x, G, sol, ls, options; callback=callback)
end

spg(funObj, x, funProj, options, ls) = spg(funObj, x, funProj, options;ls=ls)

"""
    spg(f, g!, fg!, x, funProj, options; ls=nothing, callback=nothing)

Function for using Spectral Projected Gradient to solve problems of the form
min funObj(x) s.t. x in C

# Arguments
- `f(x)`: function to minimize (returns objective only)
- `g!(g, x)`: gradient of function (in place)
- `fg!(g, x)`: objective and gradient (in place)
- `funProj(x)`: function that returns projection of x onto C
- `x`: Initial guess
- `options`: spg_options structure

# Optional Arguments
- `ls` `: User provided linesearch function
- `callback` : Callback function. Must take as input a `result` callback(x::result)

# Notes:
- if the projection is expensive to compute, you can reduce the
      number of projections by setting testOpt to 0 in the options

- Adapted fromt he matlab implementation of minConf_SPG
"""
function spg(f::Function, g!::Function, fg!::Function, x::AbstractArray{T},
             funProj::Function, options::SPG_params=spg_options();
             ls=nothing, callback=noop_callback) where {T}
    # Result structure
    sol = result(x)
    # Initialize array for gradient
    G = similar(x)

    # Setup Function to track number of evaluations
    projection(x) = (sol.n_project +=1; return funProj(x))
    obj(x) = (sol.n_ϕeval +=1 ; return f(x))
    grad!(g, x) = (sol.n_geval +=1 ; return g!(g, x))
    objgrad!(g, x) = (sol.n_ϕeval +=1;sol.n_geval +=1 ; return fg!(g, x))

    # Solve optimization
    return _spg(obj, grad!, objgrad!, projection, x, G, sol, ls, options; callback=callback)
end

spg(f, g!, fg!, x, funProj, options, ls) = spg(f, g!, fg!, x, funProj, options; ls=ls)

"""
Low level SPG solver
"""
function _spg(obj::Function, grad!::Function, objgrad!::Function, projection::Function,
              x::AbstractArray{T}, g::AbstractArray{T}, sol::result, ls, options::SPG_params;
              callback=noop_callback) where {T}
    # Initialize local variables
    nVars = length(x)
    old_ϕvals = -T(Inf)*ones(T, options.memory)
    d = similar(x)
    optCond = 0
    # Best solution
    x_best = x

    # Line search function
    isnothing(ls) && (ls = BackTracking{T}(order=3, iterations=options.maxLinesearchIter))
    checkls(ls)

    # Evaluate Initial Point and objective function
    ~options.feasibleInit && (x = projection(x))
    ϕ = objgrad!(g, x)
    ϕ_best = ϕ
    update!(sol; iter=0, ϕ=ϕ, g=g, x=x, store_trace=options.store_trace)

    # Output Log
    options.testOpt ? optCond = norm(projection(x-g)-x, options.optNorm) : optCond = 0
    init_log(ϕ, norm(projection(x-g)-x, options.optNorm), options, ls)

    # Optionally check optimality
    if options.testOpt && options.testInit
        if optCond < optTol
            verbose > 0 &&  @printf("First-Order Optimality Conditions Below optTol at Initial Point, norm g is %5.4f \n", norm(g))
            return
        end
    end

    # Start iterations
    for i = 1:options.maxIter
        flush(stdout)
        # Compute Step Directional
        if i == 1 || ~options.useSpectral
            alpha = T(.1*norm(x, Inf)/norm(g, Inf))
        else
            y = g - sol.g
            s = x - sol.x
            options.bbType ? alpha = dot(s,s)/dot(s,y) : alpha = dot(s,y)/dot(y,y)
        end
        # Make sure alpha value is valid
        (alpha <= 1e-10 || alpha > 1e10 || ~isLegal(alpha)) && (alpha = T(1))

        # Compute Step
        @. d = -T(alpha).*g

        # Compute Projected Step
        ~options.curvilinear && (d .= projection(x + d) - x)

        # Check that Progress can be made along the direction
        gtd = dot(g,d)

        # Select Initial Guess to step length
        t = T(options.iniStep)

        # Compute reference function for non-monotone condition
        old_ϕvals[i%options.memory + 1] = T(ϕ)

        # Line search
        t, ϕ = linesearch(ls, sol, d, obj, grad!, objgrad!, t, maximum(old_ϕvals), gtd, g)
        x .= projection(sol.x + t*d)
        g == sol.g && grad!(g, x)

        # Check conditioning
        options.testOpt ? optCond = norm(projection(x-g)-x, options.optNorm) : optCond = Inf

        # Check if better than best solution
        ϕ < ϕ_best && (x_best = x; ϕ_best = ϕ)

        # Update log
        update!(sol; iter=i, ϕ=ϕ, x=x, g=g, store_trace=options.store_trace)

        # Output Log
        iter_log(i, sol, t, alpha, ϕ, optCond, options)
    
        # Potential callback
        callback(sol)

        # Check if terminate
        i>1 && (terminate(options, optCond, t, d, ϕ, sol.ϕ) && break)
    end

    # Restore best iteration
    sol.x = x_best
    sol.ϕ = ϕ_best
    options.store_trace && (sol.x_trace[end] = x_bes)
    return sol
end


"""
Loging utilities
"""
function init_log(ϕ, optCond, options, ls)
    if options.verbose > 0
        @printf("Running SPG...\n");
        @printf("Number of objective function to store: %d\n",options.memory);
        @printf("Using  spectral projection : %s\n",options.useSpectral);
        @printf("Maximum number of iterations: %d\n",options.maxIter);
        @printf("SPG optimality tolerance: %.2e\n",options.optTol);
        @printf("SPG progress tolerance: %.2e\n",options.progTol);
        @printf("Line search: %s\n", typeof(ls))
     end
    if options.verbose > 0
        if options.testOpt
            @printf("%10s %10s %10s %10s %15s %15s %15s %15s\n","Iteration","FunEvals", "GradEvals","Projections","Step Length","alpha","Function Val","Opt Cond")
            @printf("%10d %10d %10d %10d %15.5e %15.5e %15.5e %15.5e\n",0,0,0,0,0,0,ϕ,optCond)
        else
            @printf("%10s %10s %10s %10s %15s %15s %15s\n","Iteration","FunEvals","GradEvals","Projections","Step Length","alpha","Function Val")
            @printf("%10d %10d %10d %10d %15.5e %15.5e %15.5e\n",0,0,0,0,0,0,ϕ)
        end
    end
end

function iter_log(i, sol, t, alpha, ϕ, optCond, options)
    if options.verbose > 0
        if options.testOpt
            @printf("%10d %10d %10d %10d %15.5e %15.5e %15.5e %15.5e\n",i,sol.n_ϕeval,sol.n_geval,sol.n_project,t,alpha,ϕ,optCond)
        else
            @printf("%10d %10d %10d %10d %15.5e %15.5e %15.5e\n",i,sol.n_ϕeval,sol.n_geval,sol.n_project,t,alpha,ϕ)
        end
    end
end