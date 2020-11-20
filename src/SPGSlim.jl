mutable struct SPG_params
    verbose
    optTol
    progTol
    maxIter
    suffDec
    interp
    memory
    useSpectral
    curvilinear
    feasibleInit
    testOpt
    bbType
    testInit
    store_trace
    optNorm
    iniStep
    maxLinesearchIter
end
"""
    spg_options(;verbose=3,optTol=1f-5,progTol=1f-7,
                maxIter=20,suffDec=1f-4,interp=0,memory=2,
                useSpectral=true,curvilinear=false,
                feasibleInit=false,testOpt=true,
                bbType=true,testInit=false, store_trace=false,
                optNorm=Inf,iniStep=1f0, maxLinesearchIter=10)

Options structure for Spectral Project Gradient algorithm.

    * verbose: level of verbosity (0: no output, 1: iter (default))
    * optTol: tolerance used to check for optimality (default: 1e-5)
    * progTol: tolerance used to check for lack of progress (default: 1e-9)
    * maxIter: maximum number of iterations (default: 20)
    * suffDec: sufficient decrease parameter in Armijo condition (default: 1e-4)
    * interp: type of interpolation (0: step-size halving, 2: cubic)
    * memory: number of steps to look back in non-monotone Armijo condition
    * useSpectral: use spectral scaling of gradient direction (default: 1)
    * curvilinear: backtrack along projection Arc (default: 0)
    * testOpt: test optimality condition (default: 1)
    * feasibleInit: if 1, then the initial point is assumed to be feasible
    * bbType: type of Barzilai Borwein step (default: 1)
    * testInit: Whether to test the initial estimate for optimality (default: false)
    * store_trace: Whether to store the trace/history of x (default: false)
    * optNorm: First-Order Optimality Conditions norm (default: Inf)
    * iniStep: Initial step length estimate (default: 1)
    * maxLinesearchIter: Maximum number of line search iteration (default: 20)
"""
function spg_options(;verbose=1,optTol=1f-5,progTol=1f-7,
                     maxIter=20,suffDec=1f-4,interp=0,memory=2,
                     useSpectral=true,curvilinear=false,
                     feasibleInit=false,testOpt=true,
                     bbType=1,testInit=false, store_trace=false,
                     optNorm=Inf,iniStep=1f0, maxLinesearchIter=20)
    return SPG_params(verbose,optTol,progTol,
                      Int64(maxIter),suffDec,interp,memory,
                      useSpectral,curvilinear,
                      feasibleInit,testOpt, bbType,testInit, store_trace,
                      optNorm,iniStep, Int64(maxLinesearchIter))
end

"""
    minConF_SPG(funObj, x, funProj, options)

Function for using Spectral Projected Gradient to solve problems of the form
  min funObj(x) s.t. x in C

  * funObj(x): function to minimize (returns gradient as second argument)
  * funProj(x): function that returns projection of x onto C
  * x: Initial guess
  * options: spg_options structure

  Notes:
      - if the projection is expensive to compute, you can reduce the
          number of projections by setting testOpt to 0 in the options

Adapted fromt he matlab implementation of minConf_SPG
"""
function spg(funObj, x::Array{vDt}, funProj, options) where {vDt}
    if options.verbose > 0
       @printf("Running SPG...\n");
       @printf("Number of objective function to store: %d\n",options.memory);
       @printf("Using  spectral projection : %s\n",options.useSpectral);
       @printf("Maximum number of iterations: %d\n",options.maxIter);
       @printf("SPG optimality tolerance: %.2e\n",options.optTol);
       @printf("SPG progress tolerance: %.2e\n",options.progTol);
    end
    # Initialize local variables
    nVars = length(x)
    options.memory > 1 && (old_fvals = -Inf*ones(vDt, options.memory))
    d = zeros(vDt, nVars)
    # Result structure
    sol = result(x)
    x_best = x

    # Setup Function to track number of evaluations
    projection(x) = (sol.n_project +=1; return funProj(x))
    objective(x) = (sol.n_feval +=1 ; return funObj(x))
    ls = BackTracking(order=3, iterations=options.maxLinesearchIter)

    # Evaluate Initial Point and objective function
    ~options.feasibleInit && (x = projection(x))
    f, g = objective(x)
    f_best = f
    update!(sol; iter=1, misfit=f, gradient=g, sol=x, store_trace=options.store_trace)

    # Output Log
    if options.verbose > 0
        if options.testOpt
            @printf("%10s %10s %10s %15s %15s %15s %15s\n","Iteration","FunEvals","Projections","Step Length","alpha","Function Val","Opt Cond")
            @printf("%10d %10d %10d %15.5e %15.5e %15.5e %15.5e\n",0,0,0,0,0,f,norm(projection(x-g)-x, options.optNorm))
        else
            @printf("%10s %10s %10s %15s %15s %15s\n","Iteration","FunEvals","Projections","Step Length","alpha","Function Val")
            @printf("%10d %10d %10d %15.5e %15.5e %15.5e\n",0,0,0,0,0,f)
        end
    end

    # Optionally check optimality
    if options.testOpt && options.testInit
        if norm(projection(x-g)-x,options.optNorm) < optTol
            verbose > 0 &&  @printf("First-Order Optimality Conditions Below optTol at Initial Point, norm g is %5.4f \n", norm(g))
            return
        end
    end

    # Start iterations
    for i = 1:options.maxIter
        # Compute Step Directional
        if i == 1 || ~options.useSpectral
            alpha = vDt(.1*norm(x, Inf)/norm(g, Inf))
        else
            y = g - sol.gradient
            s = x - sol.sol
            if options.bbType == 1
                alpha = dot(s,s)/dot(s,y)
            else
                alpha = dot(s,y)/dot(y,y)
            end
            if alpha <= 1e-10 || alpha > 1e10 || ~isLegal(alpha)
                alpha = 1
            end
        end
        i>1 && update!(sol; iter=i, misfit=f, sol=x, gradient=g, store_trace=options.store_trace)
        @. d = -vDt(alpha).*g

        # Compute Projected Step
        ~options.curvilinear && (d .= projection(x + d) - x)

        # Check that Progress can be made along the direction
        gtd = dot(g,d)
        if gtd > -options.progTol && i>1
            options.verbose > 0 &&  @printf("Directional Derivative below progTol\n")
            break
        end

        # Select Initial Guess to step length
        t = vDt(options.iniStep)

        # Compute reference function for non-monotone condition
        if options.memory == 1
            funRef = f
        else
            i <= options.memory ? old_fvals[i] = f : old_fvals = [old_fvals[2:end];f]
            funRef = maximum(old_fvals)
        end

        # Line search
        ϕ(α) = (sol.n_feval +=1 ; return funObj(sol.sol + α * d)[1])
        t, f = ls(ϕ, t, funRef, gtd)
        x .= projection(sol.sol + t*d)
        f, g = objective(x)

        # Check conditioning
        if options.testOpt
            optCond = norm(projection(x-g)-x, options.optNorm)
        end
        # Check if terminate
        i>1 && (terminate(options, norm(projection(x-g)-x, options.optNorm), t, d, f, sol.misfit) && break)

        # Take step
        if f < f_best
            x_best = x
            f_best = f
        end

        # Output Log
        if options.verbose > 0
            if options.testOpt
                @printf("%10d %10d %10d %15.5e %15.5e %15.5e %15.5e\n",i,sol.n_feval,sol.n_project,t,alpha,f,optCond)
            else
                @printf("%10d %10d %10d %15.5e %15.5e %15.5e\n",i,sol.n_feval,sol.n_project,t,alpha,f)
            end
        end
    end

    # Restore best iteration
    update!(sol; iter=options.maxIter+1, misfit=f_best, sol=x_best, gradient=g, store_trace=options.store_trace)
    return sol
end
