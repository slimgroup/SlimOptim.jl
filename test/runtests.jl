# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

using SlimOptim, Test, LineSearches

@testset "SPG" begin
    include("test_spg.jl")
end

@testset "PQN" begin
    include("test_pqn.jl")
end

@testset "Linearized bregman" begin
    include("test_bregman.jl")
end

@testset "Rosenbrock accuracy" begin
    include("test_rosen.jl")
end


# Test linesearch error

sol = result([1])
ls = BackTracking(;iterations=5)

f(x) = x[]
@test linesearch(ls, sol, [1.0], f, x->-x, x->-x, 1.0, 1.0, 1.0, [1.0])[1] == 0.0
fs(x) = sqrt(-1)
@test_throws DomainError linesearch(ls, sol, [1.0], fs, x->x, x->x, 1.0, 1.0, 1.0, [1.0])