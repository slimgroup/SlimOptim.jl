# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

using LinearAlgebra

N1 = 100
N2 = div(N1, 2) + 5

@testset "Bregman test for type $(T) and weighted $(weighted)" for (T, weighted) in [(Float32, true), (ComplexF32, true), (Float32, false), (ComplexF32, false)]

    A = randn(T, N1, N2)
    x0 = 10 .* randn(T, N2)
    x0[abs.(x0) .< 1f-6] .= 1.0
    inds = rand(1:N2, div(N2, 4))
    ninds = [i for i=1:N2 if i ∉ inds]
    x0[inds] .= 0
    b = A*x0

    function obj(x)
        fun = .5*norm(A*x - b)^2
        grad = A'*(A*x - b)
        return fun, grad
    end

    opt = weighted ? bregman_options(maxIter=200, progTol=0, verbose=2, antichatter=T==Float32, w=Float32.(x0.==0)) : bregman_options(maxIter=200, progTol=0, verbose=2, antichatter=T==Float32)
    sol = bregman(obj, 1 .+ randn(T, N2), opt)

    @show sol.x[inds]
    @show x0[inds]
    @show sol.x[ninds]
    @show x0[ninds]

    part_n = i -> norm(sol.x[i] - x0[i])/(norm(x0[i]) + norm(sol.x[i]) + eps(Float32))
    part_nz = i -> norm(sol.x[i], 1)/N2
    @show part_nz(inds)
    @show part_n(ninds)

    @test part_nz(inds) < 1f-1
    @test part_n(ninds) < 1f-1
    @test sol.residual/sol.r_trace[1] < 1f-1

end