# SlimOptim.jl Documentation

```@contents
```

# Line searches

Line search utuility function that calls [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl), this function is used for the line search at each iteration in [`spg`](@ref) and [`pqn`](@ref) and can be used by itslef as weel. For convenience the linesearches are all exported and available.

```@docs
linesearch
```

# SPG

Spectral Projected gradient algorithm adapted from [min_Conf](https://www.cs.ubc.ca/~schmidtm/Software/minConf.html) for constrained optimization.

```@docs
spg
```

The algorithms uses the following options:

```@docs
spg_options
```


# PQN

Projected Quasi-Newton algorithm adapted from [min_Conf](https://www.cs.ubc.ca/~schmidtm/Software/minConf.html) for constrained optimization.

```@docs
pqn
```

The algorithms uses the following options:

```@docs
pqn_options
```

# Linearized bregman

Linearized bregman iteration for split feasability problems.

```@docs
bregman_options
```

```@docs
bregman
```