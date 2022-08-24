# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020

using SlimOptim

# Rosenbrock function
f(m) = ((1-m[1])^2 + 100*(m[2]-m[1]^2)^2)
g!(g,m) = (g .= [-2*(1-m[1]) - 200*(m[2]-m[1]^2)*2*m[1], 200*(m[2]-m[1]^2)])
prj(x) = x

function obj(x) 
    g = zeros(2)
    ϕ = f(x)
    g!(g,x)
    ϕ, g
end

function fg!(g,x) 
    !isnothing(g) && g!(g,x)
    f(x)
end

function fg!(ff,g,x) 
    !isnothing(g) && g!(g,x)
    !isnothing(ff) && return f(x)
end

niter = 1000
for (opt, algo)=zip([spg_options, pqn_options], [spg, pqn])
    x0 = zeros(2)
    options = opt(maxIter=niter, progTol=1f-30, optTol=0, iniStep=2, memory=5)
    algo(obj,x0,prj,options)
    @show x0 ≈ [1, 1]
    @test x0 ≈ [1, 1]

    x1 = zeros(2)
    algo(f,g!,fg!,x1,prj,options)
    @show x1 ≈ [1, 1]
    @test x1 ≈ [1, 1]
end
