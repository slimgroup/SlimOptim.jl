# SlimOptim

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://slimgroup.github.io/SlimOptim.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://slimgroup.github.io/SlimOptim.jl/stable/)
[![Build Status](https://github.com/slimgroup/SlimOptim.jl/workflows/CI-SLimOptim/badge.svg)](https://github.com/slimgroup/SlimOptim.jl/actions?query=workflow%3ACI-SLimOptim)
[![DOI](https://zenodo.org/badge/314640400.svg)](https://zenodo.org/badge/latestdoi/314640400)

Package of optimizations functions for large scale inversion. In these implementations, the algorithm itself is not optimized for speed as this package is designed for inverse problems where the function evaluation is the main cost (~hours for a single objective + gradient evaluation) making the algorithm speed minimal.

# Installation

SlimOptim is registered and can be installed with the standard julia package manager:

```julia
] add/dev SlimOptim
```

# Algorithms

This repository currently contains three algorithms.

- Spectral Projected Gradient (`spg`)
- Projected Quasi Newton (`pqn`)
- Linearized Bregman (`bregman`) (in development)

SPG and PQN are using a linesearch at each iteration. In both cases, the default line search is `BackTracking(order=3, iterations=options.maxLinesearchIter)`. This default can be modified passing the linesearch function as an input, i.e, `spg(...., linesearch=linesearch)`. We currently support line searches from [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).

# Usage

The usage for SPG and PQN is the same with some small differences in the options accepted. Both can be used in two ways. `spg/pqn(fun, x, proj, options, ls)` and `spg/pqn(f, g!, fg!, x, proj, options, ls)`.

# References

This package implements adapatations of `minConf_SPG` and `minConf_PQN` from the matlab implementation of M. Schmidt [[1]].

```
* M. Schmidt, E. van den Berg, M. Friedlander, K. Murphy. Optimizing Costly Functions with Simple Constraints: A Limited-Memory Projected Quasi-Newton Algorithm. AISTATS, 2009.
* M. Schmidt. minConf: projection methods for optimization with simple constraints in Matlab. http://www.cs.ubc.ca/~schmidtm/Software/minConf.html, 2008.
* D. A. Lorenz, F. Schöpfer, and S. Wenger, The linearized Bregman method via split feasibility problems: Analysis and generalizations, SIAM Journal on Imaging Sciences, 7 (2014), pp. 1237-1262.
* Emmanouil Daskalakis, Felix J. Herrmann, and Rachel Kuske, “Accelerating Sparse Recovery by Reducing Chatter”, SIAM Journal on Imaging Sciences, vol. 13, pp. 1211–1239, 2020
```

[1]:https://www.cs.ubc.ca/~schmidtm/Software/minConf.html
