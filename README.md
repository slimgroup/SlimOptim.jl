# SlimOptim

PAckage of optimizations functions forl arge scale inversion. In these implementations, the algorithm itself is 
not optimized fot speed as this oackage is designed for inverse problems where the function evaluation is the main cost (~hours for a single function + gradient evaluation) making the algorithm speed minimal.

# Algorithms

This repository currently contains three algorithms.

- Spectral Projected Gradient (`spg`)
- Projected Quasi Newton (`pqn`)
- Linearized bregman (`bregman`) (in development)

SPG and PQN are using a linesearch at each iteration. In both cases, the default line search is `BackTracking(order=3, iterations=options.maxLinesearchIter)`. This default can be modified passing the linesearch function as an input, i.e, `spg(...., linesearch=linesearch)`. We currently only support `BackTracking` type and intend to broaden the support to all line searches from [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).

# References

This package implements adapatations of `minConf_SPG` and `minConf_PQN` from the matlab implementation of M. Schmidt [1].
