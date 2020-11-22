# SlimOptim

Package of optimizations functions forl arge scale inversion. In these implementations, the algorithm itself is 
not optimized fot speed as this oackage is designed for inverse problems where the function evaluation is the main cost (~hours for a single function + gradient evaluation) making the algorithm speed minimal.

# Algorithms

This repository currently contains three algorithms.

- Spectral Projected Gradient (`spg`)
- Projected Quasi Newton (`pqn`)
- Linearized bregman (`bregman`) (in development)

SPG and PQN are using a linesearch at each iteration. In both cases, the default line search is `BackTracking(order=3, iterations=options.maxLinesearchIter)`. This default can be modified passing the linesearch function as an input, i.e, `spg(...., linesearch=linesearch)`. We currently support all but only line searches from [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).

# Usage

The usage for SPG and PQN is the same with some small differences in the options accepted. Both can be used in two ways. `spg/pqn(fun, x, proj, options, ls)` and `spg/pqn(f, g!, fg!, x, proj, options, ls)`.

# References

This package implements adapatations of `minConf_SPG` and `minConf_PQN` from the matlab implementation of M. Schmidt [[1]].

```
* M. Schmidt, E. van den Berg, M. Friedlander, K. Murphy. Optimizing Costly Functions with Simple Constraints: A Limited-Memory Projected Quasi-Newton Algorithm. AISTATS, 2009.
* M. Schmidt. minConf: projection methods for optimization with simple constraints in Matlab. http://www.cs.ubc.ca/~schmidtm/Software/minConf.html, 2008.
* Lorenz
* Mengmeng?
```

[1]:https://www.cs.ubc.ca/~schmidtm/Software/minConf.html
