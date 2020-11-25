mutable struct PQN_params
    verbose
    optTol
    progTol
    maxIter
    suffDec
    corrections
    adjustStep
    bbInit
    store_trace
    SPGoptTol
    SPGprogTol
    SPGiters
    SPGtestOpt
    maxLinesearchIter
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
"""
function pqn_options(;verbose=1, optTol=1f-5, progTol=1f-7,
                     maxIter=20, suffDec=1f-4, corrections=10, adjustStep=false,
                     bbInit=true, store_trace=false, SPGoptTol=1f-6, SPGprogTol=1f-7,
                     SPGiters=10, SPGtestOpt=false, maxLinesearchIter=20)
    return PQN_params(verbose, optTol ,progTol, Int64(maxIter), suffDec, corrections,
                      adjustStep, bbInit, store_trace, SPGoptTol,
                      SPGprogTol, SPGiters, SPGtestOpt, Int64(maxLinesearchIter))
end

"""
    pqn(objective, projection, x,options)

Function for using a limited-memory projected quasi-Newton to solve problems of the form
  min objective(x) s.t. x in C

The projected quasi-Newton sub-problems are solved the spectral projected
gradient algorithm

# Arguments

- `funObj(x)`: function to minimize (returns gradient as second argument)
- `funProj(x)`: function that returns projection of x onto C
- `x`: Initial guess
- `options`: pqn_options structure

# Notes:
    Adapted fromt he matlab implementation of minConf_PQN
"""
function pqn(funObj, x::AbstractArray{vDt}, funProj, options, ls=nothing) where {vDt}
    f(x) = funObj(x)[1]
    g!(g, x) = (g .= funObj(x)[2]; return)
    fg!(g, x) = ((obj, g0) = funObj(x); g .= g0; return obj)
    return pqn(f, g!, fg!, x, funProj, options, ls)
end

"""
    pqn(f, g!, fg!, x, projection,options)

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

# Notes:
    Adapted fromt he matlab implementation of minConf_PQN
"""
function pqn(f::Function, g!::Function, fg!::Function, x::AbstractArray{vDt}, funProj, options, ls=nothing) where {vDt}
    nVars = length(x)
    spg_opt = spg_options(optTol=options.SPGoptTol,progTol=options.SPGprogTol, maxIter=options.SPGiters,
                          testOpt=options.SPGtestOpt, feasibleInit=~options.bbInit, verbose=0)
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
    end

    # Result structure
    sol = result(x)
    g = similar(x)

    # Setup Function to track number of evaluations
    projection(x) = (sol.n_project +=1; return funProj(x))
    obj(x) = (sol.n_ϕeval +=1 ; return f(x))
    grad!(g, x) = (sol.n_geval +=1 ; return g!(g, x))
    objgrad!(g, x) = (sol.n_ϕeval +=1;sol.n_geval +=1 ; return fg!(g, x))

    # Line search function
    isnothing(ls) && (ls = BackTracking(order=3, iterations=options.maxLinesearchIter))
    checkls(ls)

    # Project initial parameter vector
    x = projection(x)

    # Evaluate initial parameters
    ϕ = objgrad!(g, x)
    update!(sol; iter=1, ϕ=ϕ, x=x, g=g, store_trace=options.store_trace)

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
    S = zeros(vDt, nVars, 0)
    Y = zeros(vDt, nVars, 0)
    d = zeros(vDt, nVars)
    Hdiag = 1

    for i=1:options.maxIter
        # Compute Step Direction
        if i == 1
            p = projection(x-g)
        else
            y = g-sol.g
            s = x-sol.x
            S, Y, Hdiag = lbfgsUpdate(y,s,options.corrections,options.verbose,S,Y,Hdiag)

            # Make Compact Representation
            k = size(Y,2)
            L = zeros(Float32, k, k)
            for j = 1:k
                L[j+1:k,j] = transpose(S[:,j+1:k])*Y[:,j]
            end
            N = [S/Hdiag Y];
            M = [S'*S/Hdiag L;transpose(L) -Diagonal(diag(S'*Y))]
            HvFunc(v) = lbfgsHvFunc2(v,Hdiag,N,M)

            if options.bbInit || i < options.corrections/2
                # Use Barzilai-Borwein step to initialize sub-problem
                alpha = dot(s,s)/dot(s,y);
                if alpha <= 1e-10 || alpha > 1e10 || ~isLegal(alpha)
                    alpha = min(1,1/norm(g, 1))
                end
                # Solve Sub-problem
                xSubInit = x-alpha*g
            else
                xSubInit = x
            end
            # Solve Sub-problem
            p = solveSubProblem(x,g,HvFunc,projection,spg_opt,xSubInit)
        end
        @. d = p - x

        # Check that Progress can be made along the direction
        gtd = dot(g,d)
        if gtd > -options.progTol && (i > options.corrections/2)
            options.verbose > 0 && @printf("Directional Derivative below progTol\n")
            break
        end
        # Select Initial Guess to step length
        (~options.adjustStep || gtd == 0 || i==1) ? t = 1.0 : t = vDt(min(1, 2*(ϕ-sol.ϕ)/gtd))

        # Save history
        i>1 && update!(sol; iter=i, ϕ=ϕ, x=x, g=g, store_trace=options.store_trace)
        
        # Line search
        t, ϕ = linesearch(ls, sol, d, obj, grad!, objgrad!, t, sol.ϕ, gtd, g)
        x .= projection(sol.x + t*d)
        grad!(g, x)

        # Check termination
	    optCond = norm(projection(x-g) - x, Inf)
        i>1 && (terminate(options, optCond, t, d, ϕ, sol.ϕ) && break)
        # Output Log
        if options.verbose > 0
            @printf("%10d %10d %10d %10d %15.5e %15.5e %15.5e\n",i,sol.n_ϕeval, sol.n_geval, sol.n_project, t, ϕ, optCond)
        end

    end
    isLegal(x) && update!(sol; iter=options.maxIter+1, ϕ=ϕ, x=x, g=g, store_trace=options.store_trace)
    return return sol
end
