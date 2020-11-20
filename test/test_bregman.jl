using LinearAlgebra

N1 = 100
N2 = div(N1, 2) + 5
A = randn(N1, N2)

x0 = 10 .* randn(N2)
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

opt = bregman_options(maxIter=200, progTol=0, verbose=2)
sol = bregman(obj, 1 .+ randn(N2), opt)

@show sol.sol[inds]
@show x0[inds]
@show sol.sol[ninds]
@show x0[ninds]

part_n = i -> norm(sol.sol[i] - x0[i])/(norm(x0[i]) + norm(sol.sol[i]) + eps(Float64))
part_nz = i -> norm(sol.sol[i], 1)/N2
@show part_nz(inds)
@show part_n(ninds)

@test part_nz(inds) < 1f-1
@test part_n(ninds) < 1f-1
@test sol.misfit/sol.f_trace[1] < 1f-1