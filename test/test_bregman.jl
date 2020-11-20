using LinearAlgebra

N1 = 100
N2 = div(N1, 2) + 5
A = randn(N1, N2)

x0 = ones(N2)
inds = rand(1:N2, div(N2, 4))
ninds = [i for i=1:N2 if i ∉ inds]
x0[inds] .= 0
b = A*x0

function obj(x)
    fun = .5*norm(A*x - b)^2
    grad = A'*(A*x - b)
    return fun, grad
end

opt = bregman_options(maxIter=200, progTol=0, verbose=2)
sol = bregman(obj, 1 .+ randn(N2), opt)

@show sol.sol[inds]
@show x0[inds]
@show sol.sol[ninds]
@show x0[ninds]

@test sol.misfit/sol.f_trace[1]  < 1e-9
@test norm(sol.sol - x0)/(norm(x0) + norm(sol.sol)) < 1f-4