# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

using SlimOptim, Test

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