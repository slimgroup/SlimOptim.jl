# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

using LinearAlgebra, SlimOptim

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

function mycallback(sol::result)
    # Print some info. ϕ_trace contains initial value so iteration is lenght-1
    println("Bonjour at iteration $(length(sol.ϕ_trace)-1) with misfit value of $(sol.ϕ)")
    println("Norm of solution is $(norm(sol.x))")
    nothing
end

function mycallback(sol::BregmanIterations)
    # Print some info. ϕ_trace contains initial value so iteration is lenght-1
    println("Bonjour at iteration $(length(sol.ϕ_trace)-1) with misfit value of $(sol.ϕ)")
    println("Norm of solution are $(norm(sol.x)), $(norm(sol.z))")
    nothing
end

# PQN
sol = pqn(obj, randn(N), proj)
sol = pqn(obj, randn(N), proj, callback=mycallback)

# SPG
sol = spg(obj, randn(N), proj)
sol = spg(obj, randn(N), proj, callback=mycallback)

# Bregman
sol = bregman(A, zeros(Float32, N), A*randn(Float32, N))
sol = bregman(A, zeros(Float32, N), A*randn(Float32, N); callback=mycallback)