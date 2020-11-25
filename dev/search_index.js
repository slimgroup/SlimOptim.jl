var documenterSearchIndex = {"docs":
[{"location":"","page":"Line searches","title":"Line searches","text":"","category":"page"},{"location":"#Line-searches","page":"Line searches","title":"Line searches","text":"","category":"section"},{"location":"","page":"Line searches","title":"Line searches","text":"Line search utuility function that calls LineSearches.jl, this function is used for the line search at each iteration in spg and pqn and can be used by itslef as weel. For convenience the linesearches are all exported and available.","category":"page"},{"location":"","page":"Line searches","title":"Line searches","text":"linesearch","category":"page"},{"location":"#SlimOptim.linesearch","page":"Line searches","title":"SlimOptim.linesearch","text":"linesearch(ls, sol, d, f, g!, fg!, t, funRef, gtd, gvec)\n\nLine search interface to LineSearches.jl\n\nArguments\n\nls: Line search structure (see LineSearches.jl documentation)\nf: Objective function, x -> f(x)\ng!`: Gradient in place function, x-> (g.= gradient(x))\nfg!`: Objective and in place gradient function, x-> (f = f(x); g.= gradient(x))\nt: Initial steplength gess\nfunRef: Reference objective function value\ngtd: Reference direction inner product dot(g0, d0)\ngvec: prealocated array for thhe gradient\n\n\n\n\n\n","category":"function"},{"location":"#SPG","page":"Line searches","title":"SPG","text":"","category":"section"},{"location":"","page":"Line searches","title":"Line searches","text":"Spectral Projected gradient algorithm adapted from min_Conf for constrained optimization.","category":"page"},{"location":"","page":"Line searches","title":"Line searches","text":"spg","category":"page"},{"location":"#SlimOptim.spg","page":"Line searches","title":"SlimOptim.spg","text":"spg(funObj, x, funProj, options)\n\nFunction for using Spectral Projected Gradient to solve problems of the form   min funObj(x) s.t. x in C\n\nArguments\n\nfunObj(x):function to minimize (returns gradient as second argument)\nfunProj(x): function that returns projection of x onto C\nx: Initial guess\noptions: spg_options structure\n\nNotes:\n\nif the projection is expensive to compute, you can reduce the number of projections by setting testOpt to 0 in the options\nAdapted fromt he matlab implementation of minConf_SPG\n\n\n\n\n\nspg(f, g!, fg!, x, funProj, options)\n\nFunction for using Spectral Projected Gradient to solve problems of the form min funObj(x) s.t. x in C\n\nArguments\n\nf(x): function to minimize (returns objective only)\ng!(g, x): gradient of function (in place)\nfg!(g, x): objective and gradient (in place)\nfunProj(x): function that returns projection of x onto C\nx: Initial guess\noptions: spg_options structure\n\nNotes:\n\nif the projection is expensive to compute, you can reduce the     number of projections by setting testOpt to 0 in the options\nAdapted fromt he matlab implementation of minConf_SPG\n\n\n\n\n\n","category":"function"},{"location":"","page":"Line searches","title":"Line searches","text":"The algorithms uses the following options:","category":"page"},{"location":"","page":"Line searches","title":"Line searches","text":"spg_options","category":"page"},{"location":"#SlimOptim.spg_options","page":"Line searches","title":"SlimOptim.spg_options","text":"spg_options(;verbose=3,optTol=1f-5,progTol=1f-7,\n            maxIter=20,suffDec=1f-4,memory=2,\n            useSpectral=true,curvilinear=false,\n            feasibleInit=false,testOpt=true,\n            bbType=true,testInit=false, store_trace=false,\n            optNorm=Inf,iniStep=1f0, maxLinesearchIter=10)\n\nOptions structure for Spectral Project Gradient algorithm.\n\nArguments\n\nverbose: level of verbosity (0: no output, 1: iter (default))\noptTol: tolerance used to check for optimality (default: 1e-5)\nprogTol: tolerance used to check for lack of progress (default: 1e-9)\nmaxIter: maximum number of iterations (default: 20)\nsuffDec: sufficient decrease parameter in Armijo condition (default: 1e-4)\nmemory: number of steps to look back in non-monotone Armijo condition\nuseSpectral: use spectral scaling of gradient direction (default: 1)\ncurvilinear: backtrack along projection Arc (default: 0)\ntestOpt: test optimality condition (default: 1)\nfeasibleInit: if 1, then the initial point is assumed to be feasible\nbbType: type of Barzilai Borwein step (default: 1)\ntestInit: Whether to test the initial estimate for optimality (default: false)\nstore_trace: Whether to store the trace/history of x (default: false)\noptNorm: First-Order Optimality Conditions norm (default: Inf)\niniStep: Initial step length estimate (default: 1)\nmaxLinesearchIter: Maximum number of line search iteration (default: 20)\n\n\n\n\n\n","category":"function"},{"location":"#PQN","page":"Line searches","title":"PQN","text":"","category":"section"},{"location":"","page":"Line searches","title":"Line searches","text":"Projected Quasi-Newton algorithm adapted from min_Conf for constrained optimization.","category":"page"},{"location":"","page":"Line searches","title":"Line searches","text":"pqn","category":"page"},{"location":"#SlimOptim.pqn","page":"Line searches","title":"SlimOptim.pqn","text":"pqn(objective, projection, x,options)\n\nFunction for using a limited-memory projected quasi-Newton to solve problems of the form   min objective(x) s.t. x in C\n\nThe projected quasi-Newton sub-problems are solved the spectral projected gradient algorithm\n\nArguments\n\nfunObj(x): function to minimize (returns gradient as second argument)\nfunProj(x): function that returns projection of x onto C\nx: Initial guess\noptions: pqn_options structure\n\nNotes:\n\nAdapted fromt he matlab implementation of minConf_PQN\n\n\n\n\n\npqn(f, g!, fg!, x, projection,options)\n\nFunction for using a limited-memory projected quasi-Newton to solve problems of the form   min objective(x) s.t. x in C\n\nThe projected quasi-Newton sub-problems are solved the spectral projected gradient algorithm.\n\nArguments\n\nf(x): function to minimize (returns objective only)\ng!(g, x): gradient of function (in place)\nfg!(g, x): objective and gradient (in place)\nfunProj(x): function that returns projection of x onto C\nx: Initial guess\noptions: pqn_options structure\n\nNotes:\n\nAdapted fromt he matlab implementation of minConf_PQN\n\n\n\n\n\n","category":"function"},{"location":"","page":"Line searches","title":"Line searches","text":"The algorithms uses the following options:","category":"page"},{"location":"","page":"Line searches","title":"Line searches","text":"pqn_options","category":"page"},{"location":"#SlimOptim.pqn_options","page":"Line searches","title":"SlimOptim.pqn_options","text":"pqn_options(;verbose=2, optTol=1f-5, progTol=1f-7,\n            maxIter=20, suffDec=1f-4, corrections=10, adjustStep=false,\n            bbInit=false, store_trace=false, SPGoptTol=1f-6, SPGprogTol=1f-7,\n            SPGiters=10, SPGtestOpt=false, maxLinesearchIter=20)\n\nOptions structure for Spectral Project Gradient algorithm.\n\nArguments\n\nverbose: level of verbosity (0: no output, 1: iter (default))\noptTol: tolerance used to check for optimality (default: 1e-5)\nprogTol: tolerance used to check for progress (default: 1e-9)\nmaxIter: maximum number of iterations (default: 20)\nsuffDec: sufficient decrease parameter in Armijo condition (default: 1e-4)\ncorrections: number of lbfgs corrections to store (default: 10)\nadjustStep: use quadratic initialization of line search (default: 0)\nbbInit: initialize sub-problem with Barzilai-Borwein step (default: 1)\nstore_trace: Whether to store the trace/history of x (default: false)\nSPGoptTol: optimality tolerance for SPG direction finding (default: 1e-6)\nSPGprogTol: SPG tolerance used to check for progress (default: 1e-7)\nSPGiters: maximum number of iterations for SPG direction finding (default:10)\nSPGtestOpt: Whether to check for optimality in SPG (default: false)\nmaxLinesearchIter: Maximum number of line search iteration (default: 20)\n\n\n\n\n\n","category":"function"},{"location":"#Linearized-bregman","page":"Line searches","title":"Linearized bregman","text":"","category":"section"},{"location":"","page":"Line searches","title":"Line searches","text":"Linearized bregman iteration for split feasability problems.","category":"page"},{"location":"","page":"Line searches","title":"Line searches","text":"bregman_options","category":"page"},{"location":"#SlimOptim.bregman_options","page":"Line searches","title":"SlimOptim.bregman_options","text":"bregman_options(;verbose=1, optTol=1e-6, progTol=1e-8, maxIter=20\n                store_trace=false, linesearch=false, alpha=.25)\n\nOptions structure for the bregman iteration algorithm\n\nArguments\n\nverbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3: debug)\nprogTol: tolerance used to check for lack of progress (default: 1e-9)\nmaxIter: maximum number of iterations (default: 20)\nstore_trace: Whether to store the trace/history of x (default: false)\nantichatter: Whether to use anti-chatter step length correction\nquantile: Thresholding level as quantile value, (default=.95 i.e thresholds 95% of the vector)\nalpha: Strong convexity modulus. (step length is α fracr_2^2g_2^2)\n\n\n\n\n\n","category":"function"},{"location":"","page":"Line searches","title":"Line searches","text":"bregman","category":"page"},{"location":"#SlimOptim.bregman","page":"Line searches","title":"SlimOptim.bregman","text":"bregman(A, TD, x, b, options)\n\nLinearized bregman iteration for the system\n\nfrac12 TD  x_2^2 + λ TD  x_1     st Ax = b\n\nFor example, for sparsity promoting denoising (i.e LSRTM)\n\nArguments\n\nTD: curvelet transform\nA: Forward operator (J or preconditioned J for LSRTM)\nb: observed data\nx: Initial guess\n\n\n\n\n\nbregman(fun, TD, x, b, options)\n\nLinearized bregman iteration for the system\n\nfrac12 TD  x_2^2 + λ TD  x_1     st Ax = b\n\nFor example, for sparsity promoting denoising (i.e LSRTM)\n\nArguments\n\nTD: curvelet transform\nfun: residual function, return the tuple (f = frac12Ax - b_2, g = A^T(Ax - b))\nb: observed data\nx: Initial guess\n\n\n\n\n\n","category":"function"}]
}
