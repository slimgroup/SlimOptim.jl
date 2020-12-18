using LinearAlgebra

N = 10
A = Diagonal(1:N)

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

for ls in [BackTracking, HagerZhang, StrongWolfe]
    opt = spg_options(maxIter=1000, progTol=1f-6, optTol=1f-6, verbose=2)
    sol = spg(obj, randn(N), proj, opt, ls())

    @show norm(sol.x - x0)/(norm(x0) + norm(sol.x))
    @show sol.ϕ/sol.ϕ_trace[1]

    @test sol.ϕ/sol.ϕ_trace[1]  < 1e-9
    @test norm(sol.x - x0)/(norm(x0) + norm(sol.x)) < 1f-4
end