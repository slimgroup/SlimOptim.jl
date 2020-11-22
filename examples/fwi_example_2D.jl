# 2D FWI on Overthrust model using minConf library
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#

using Statistics, Random, LinearAlgebra
using JUDI, JUDI.TimeModeling, SlimOptim, HDF5, SegyIO, PyPlot

# Load starting model
path = dirname(pathof(JUDI))
n,d,o,m0 = read(h5open(path*"/../data/overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)
vmin = ones(Float32,model0.n) .* 1.3f0
vmax = ones(Float32,model0.n) .* 6.5f0
vmin[:,1:21] .= v0[:,1:21]   # keep water column fixed
vmax[:,1:21] .= v0[:,1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data
block = segy_read(path*"/../data/overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)  # 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

############################### FWI ###########################################
F0 = judiModeling(model0, src_geometry, d_obs.geometry)

# Optimization parameters
niterations = 10
batchsize = 16
fhistory_SGD = zeros(Float32,niterations)

# Projection operator for bound constraints
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2),model0.n)

ls = BackTracking(order=3, iterations=10)
# Main loop
for j=1:niterations

    # get fwi objective function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0,q[i],d_obs[i])
    direction = -gradient./norm(gradient, Inf)
    println("FWI iteration no: ",j,"; function value: ",fval)
    fhistory_SGD[j] = fval

    # linesearch
    function ϕ(α) 
        F0.model.m .= proj(model0.m .+ α * direction)
<<<<<<< HEAD
        misfit = .5*norm(F0[i]*q[i] - d_obs[i])^2
=======
        misfit = .5*norm(F0[i]*q[i] - d_obs[i])
        println(misfit)
>>>>>>> 7c63842... fixed bugs
        return misfit
    end
    step, fval = ls(ϕ, 1f0, fval, dot(gradient, direction))

    # Update model and bound projection
    model0.m = proj(model0.m .+ step .* direction)
end

figure(); imshow(sqrt.(1f0./adjoint(model0.m))); title("FWI with SGD")