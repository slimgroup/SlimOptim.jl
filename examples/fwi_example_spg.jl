# 2D FWI on Overthrust model with SPG using minConf library
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#
# Install JUDI if not installed
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Updated: December 2020


using LinearAlgebra, PyPlot, SlimOptim
using JUDI, JUDI.TimeModeling, HDF5, SegyIO

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

# Optimization parameters
fevals = 16
batchsize = 8

# Objective function for minConf library
count = 0
function objective_function(x)
    model0.m = reshape(x,model0.n);

    # fwi function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, grad = fwi_objective(model0, q[i], d_obs[i])
    grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

    global count; count+= 1
    return fval, vec(grad)
end

# Bound projection
ProjBound(x) = (x[x.< mmin].=mmin[x.< mmin];x[x.>mmax].=mmax[x.>mmax])

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)
sol = spg(objective_function, vec(model0.m), ProjBound, options)

# Plot result
imshow(reshape(sqrt.(1f0 ./ sol.sol), model0.n)', extent=[0, 10, 3, 0])
xlabel("Lateral position [km]")
ylabel("Depth [km]")