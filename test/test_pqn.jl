using LinearAlgebra

N = 10

A = diagm(1:N)

function proj(x)
    xp = deepcopy(x)
    xp[xp .< 0] .= 0
    return xp
end

x0 = 10 .+ 10 .* rand(N)
b = A*x0

function obj(x)
    fun = .5*norm(A*x - b)^2
    grad = A'*(A*x - b)
    return fun, grad
end

opt = pqn_options(maxIter=100, progTol=0, corrections=100, maxLinesearchIter=100, optTol=0, verbose=2)
sol = pqn(obj, randn(N), proj, opt)

@test sol.misfit/sol.f_trace[1]  < 1e-12
@test norm(sol.sol - x0)/(norm(x0) + norm(sol.sol)) < 1f-4

opt = pqn_options(maxIter=100, progTol=0, adjustStep=true, corrections=100, maxLinesearchIter=100, optTol=0, verbose=2)
sol = pqn(obj, randn(N), proj, opt)

@test sol.misfit/sol.f_trace[1]  < 1e-12
@test norm(sol.sol - x0)/(norm(x0) + norm(sol.sol)) < 1f-4