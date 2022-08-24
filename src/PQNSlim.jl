# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

mutable struct PQN_params
    verbose::Integer
    optTol::Real
    progTol::Real
    maxIter::Integer
    suffDec::Real
    corrections::Integer
    adjustStep::Bool
    bbInit::Bool
    store_trace::Bool
    SPGoptTol::Real
    SPGprogTol::Real
    SPGiters::Integer
    SPGtestOpt::Bool
    maxLinesearchIter::Integer
    memory::Integer
    iniStep::Real
end

"""
    pqn_options(;verbose=2, optTol=1f-5, progTol=1f-7,
                maxIter=20, suffDec=1f-4, corrections=10, adjustStep=false,
                bbInit=false, store_trace=false, SPGoptTol=1f-6, SPGprogTol=1f-7,
                SPGiters=10, SPGtestOpt=false, maxLinesearchIter=20)

Options structure for Spectral Project Gradient algorithm.

# Arguments

- `verbose`: level of verbosity (0: no output, 1: iter (default))
- `optTol`: tolerance used to check for optimality (default: 1e-5)
- `progTol`: tolerance used to check for progress (default: 1e-9)
- `maxIter`: maximum number of iterations (default: 20)
- `suffDec`: sufficient decrease parameter in Armijo condition (default: 1e-4)
- `corrections`: number of lbfgs corrections to store (default: 10)
- `adjustStep`: use quadratic initialization of line search (default: 0)
- `bbInit`: initialize sub-problem with Barzilai-Borwein step (default: 1)
- `store_trace`: Whether to store the trace/history of x (default: false)
- `SPGoptTol`: optimality tolerance for SPG direction finding (default: 1e-6)
- `SPGprogTol`: SPG tolerance used to check for progress (default: 1e-7)
- `SPGiters`: maximum number of iterations for SPG direction finding (default:10)
- `SPGtestOpt`: Whether to check for optimality in SPG (default: false)
- `maxLinesearchIter`: Maximum number of line search iteration (default: 20)
- `memory`: Number of steps for the non-monotone functional decrease condition.
- `iniStep`: Initial step length estimate (default: 1). Ignored with adjustStep.
"""
function pqn_options(;verbose=1, optTol=1f-5, progTol=1f-7,
                     maxIter=20, suffDec=1f-4, corrections=10, adjustStep=false,
                     bbInit=true, store_trace=false, SPGoptTol=1f-6, SPGprogTol=1f-7,
                     SPGiters=100, SPGtestOpt=false, maxLinesearchIter=20, memory=1, iniStep=1)
    return PQN_params(verbose, optTol ,progTol, Int64(maxIter), suffDec, corrections,
                      adjustStep, bbInit, store_trace, SPGoptTol,
                      SPGprogTol, SPGiters, SPGtestOpt, Int64(maxLinesearchIter),
                      memory, iniStep)
end

"""
    pqn(objective, x, projection, options; ls=nothing, callback=nothing)

Function for using a limited-memory projected quasi-Newton to solve problems of the form
  min objective(x) s.t. x in C

The projected quasi-Newton sub-problems are solved the spectral projected
gradient algorithm

# Arguments

- `funObj(x)`: function to minimize (returns gradient as second argument)
- `funProj(x)`: function that returns projection of x onto C
- `x`: Initial guess
- `options`: pqn_options structure

# Optional Arguments
- `ls` `: User provided linesearch function
- `callback` : Callback function. Must take as input a `result` callback(x::result)

# Notes:
    Adapted fromt he matlab implementation of minConf_PQN
"""
function pqn(funObj, x::AbstractArray{T}, funProj::Function, options::PQN_params=pqn_options(); ls=nothing, callback=noop_callback) where {T}
    # Result structure
    sol = result(x)
    G = similar(x)

    # Setup Function to track number of evaluations
    projection(x) = (sol.n_project +=1; return funProj(x))
    grad!(g, x) = (sol.n_ϕeval +=1; sol.n_geval +=1 ; g .= funObj(x)[2])
    objgrad!(g, x) = (sol.n_ϕeval +=1; sol.n_geval +=1 ;(obj, g0) = funObj(x); g .= g0; return obj)
    obj(x) = objgrad!(G, x)

    # Solve optimization
    return _pqn(obj, grad!, objgrad!, projection, x, G, sol, ls, options; callback=callback)
end

pqn(funObj, x, funProj, options, ls) = pqn(funObj, x, funProj, options;ls=ls)

"""
    pqn(f, g!, fg!, x, projection, options; ls=nothing, callback=nothing)

Function for using a limited-memory projected quasi-Newton to solve problems of the form
  min objective(x) s.t. x in C

The projected quasi-Newton sub-problems are solved the spectral projected
gradient algorithm.

# Arguments

- `f(x)`: function to minimize (returns objective only)
- `g!(g, x)`: gradient of function (in place)
- `fg!(g, x)`: objective and gradient (in place)
- `funProj(x)`: function that returns projection of x onto C
- `x`: Initial guess
- `options`: pqn_options structure

# Optional Arguments
- `ls` `: User provided linesearch function
- `callback` : Callback function. Must take as input a `result` callback(x::result)

# Notes:
    Adapted fromt he matlab implementation of minConf_PQN
"""
function pqn(f::Function, g!::Function, fg!::Function, x::AbstractArray{T},
             funProj::Function, options::PQN_params=pqn_options();
             ls=nothing, callback=noop_callback) where {T}
    # Result structure
    sol = result(x)
    G = similar(x)

    # Setup Function to track number of evaluations
    projection(x) = (sol.n_project +=1; return funProj(x))
    obj(x) = (sol.n_ϕeval +=1 ; return f(x))
    grad!(g, x) = (sol.n_geval +=1 ; return g!(g, x))
    objgrad!(g, x) = (sol.n_ϕeval +=1;sol.n_geval +=1 ; return fg!(g, x))

    # Solve optimization
    return _pqn(obj, grad!, objgrad!, projection, x, G, sol, ls, options; callback=callback)
end

pqn(f, g, fg!, x, funProj, options, ls) = pqn(f, g, fg!, x, funProj, options; ls=ls)

"""
Low level PQN solver
"""
function _pqn(obj::Function, grad!::Function, objgrad!::Function, projection::Function,
              x::AbstractArray{T}, g::AbstractArray{T}, sol::result, ls, options::PQN_params;
              callback=noop_callback) where {T}
    nVars = length(x)
    old_ϕvals = -T(Inf)*ones(T, options.memory)
    spg_opt = spg_options(optTol=options.SPGoptTol,progTol=options.SPGprogTol, maxIter=options.SPGiters,
                          testOpt=options.SPGtestOpt, feasibleInit=~options.bbInit, verbose=0)

    # Line search function
    isnothing(ls) && (ls = BackTracking{T}(order=3, iterations=options.maxLinesearchIter))
    checkls(ls)

    # Output Parameter Settings
    if options.verbose > 0
       @printf("Running PQN...\n");
       @printf("Number of L-BFGS Corrections to store: %d\n",options.corrections)
       @printf("Spectral initialization of SPG: %d\n",options.bbInit)
       @printf("Maximum number of SPG iterations: %d\n",options.SPGiters)
       @printf("SPG optimality tolerance: %.2e\n",options.SPGoptTol)
       @printf("SPG progress tolerance: %.2e\n",options.SPGprogTol)
       @printf("PQN optimality tolerance: %.2e\n",options.optTol)
       @printf("PQN progress tolerance: %.2e\n",options.progTol)
       @printf("Quadratic initialization of line search: %d\n",options.adjustStep)
       @printf("Maximum number of iterations: %d\n",options.maxIter)
       @printf("Line search: %s\n", typeof(ls))
    end
    # Best solution
    x_best = x

    # Project initial parameter vector
    x = projection(x)

    # Evaluate initial parameters
    ϕ = objgrad!(g, x)
    ϕ_best = ϕ
    old_ϕvals[1] = T(ϕ)
    update!(sol; iter=0, ϕ=ϕ, x=x, g=g, store_trace=options.store_trace)

    # Output Log
    if options.verbose > 0
        @printf("%10s %10s %10s %10s %15s %15s %15s\n","Iteration","FunEvals","GradEvals","Projections","Step Length","Function Val","Opt Cond")
        @printf("%10d %10d %10d %10d %15.5e %15.5e %15.5e\n",0, 0, 0, 0, 0, ϕ, norm(projection(x-g)-x, Inf))
    end
    
    # Check Optimality of Initial Point
    if maximum(abs.(projection(x-g)-x)) < options.optTol
        options.verbose > 0 && @printf("First-Order Optimality Conditions Below optTol at Initial Point\n");
        update!(sol; ϕ=ϕ, g=g, store_trace=options.store_trace)
        return sol
    end

    # Initialize variables
    S = zeros(T, nVars, 0)
    Y = zeros(T, nVars, 0)
    d = similar(x)
    p = similar(x)
    y = Vector{T}(undef, length(x))
    s = Vector{T}(undef, length(x))

    Hdiag = 1
    i = 1

    for i=1:options.maxIter
        flush(stdout)
        # Compute the new gradient unless already computed in the previous line search
        (i > 1 && sol.g == g) && grad!(g, x)
        @. y = g - sol.g

        # Compute Step Direction
        if i == 1
            p = projection(x-g)
        else
            S, Y, Hdiag = lbfgsUpdate(y, s, options.corrections, S, Y, Hdiag)

            # Make Compact Representation
            k = size(Y, 2)
            L = zeros(T, k, k)
            for j = 1:k
                L[j+1:k,j] = transpose(S[:,j+1:k])*Y[:,j]
            end
            N = [S/Hdiag Y];
            M = [S'*S/Hdiag L;transpose(L) -Diagonal(diag(S'*Y))]
            HvFunc(v) = lbfgsHvFunc2(v, Hdiag, N, M)

            if options.bbInit || i < options.corrections/2
                # Use Barzilai-Borwein step to initialize sub-problem
                alpha = dot(s,s)/dot(s,y);
                if alpha <= 1e-10 || alpha > 1e10 || ~isLegal(alpha)
                    alpha = min(1,1/norm(g, 1))
                end
                # Solve Sub-problem
                xSubInit = x-T(alpha)*g
            else
                xSubInit = x
            end
            # Solve Sub-problem
            solveSubProblem!(p, x, g, HvFunc, projection, spg_opt, xSubInit)
        end
        @. d = p - x

        # Directional derivative
        gtd = dot(g, d)

        # conditioning
        optCond = norm(projection(x-g) - x, Inf)

        # Select Initial Guess to step length
        (~options.adjustStep || gtd == 0 || i==1) ? t = T(options.iniStep) : t = T(min(1, 2*(ϕ-sol.ϕ_trace[end-1])/gtd))
        
        # save current gradient before linesearch
        update!(sol; g=g)

        # Line search
        t, ϕ = linesearch(ls, sol, d, obj, grad!, objgrad!, t, maximum(old_ϕvals), gtd, g)
        x .= projection(sol.x + t*d)
    
        # Check if better than best solution
        ϕ < ϕ_best && (x_best = x; ϕ_best = ϕ)

        # Output Log
        if options.verbose > 0
            @printf("%10d %10d %10d %10d %15.5e %15.5e %15.5e\n",i,sol.n_ϕeval, sol.n_geval, sol.n_project, t, ϕ, optCond)
        end

        # Compute reference function for non-monotone condition
        old_ϕvals[i%options.memory + 1] = T(ϕ)

        # New lbfgs vectors
        @. s = x - sol.x

        # Save history
        update!(sol; iter=i, ϕ=ϕ, x=x, store_trace=options.store_trace)

        # Optional callback
        callback(sol)

        # Check termination
        i>1 && (terminate(options, optCond, t, d, ϕ, sol.ϕ) && break)
    end

    return return sol
end
