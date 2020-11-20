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

opt = spg_options(maxIter=100, progTol=0, optTol=0, verbose=2)
sol = spg(obj, randn(N), proj, opt)

@show norm(sol.sol - x0)/(norm(x0) + norm(sol.sol))
@show sol.misfit/sol.f_trace[1]

@test sol.misfit/sol.f_trace[1]  < 1e-9
@test norm(sol.sol - x0)/(norm(x0) + norm(sol.sol)) < 1f-4