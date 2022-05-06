# Sparisty-promoting LS-RTM of the 2D Marmousi model with on-the-fly Fourier transforms
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI, SegyIO, HDF5, JOLI, PyPlot, SlimOptim

# Load migration velocity model
if ~isfile("marmousi_model.h5")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_model.h5`)
end
n, d, o, m0 = read(h5open("marmousi_model.h5", "r"), "n", "d", "o", "m0")

# Set up model structure
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Load data
if ~isfile("marmousi_2D.segy")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_2D.segy`)
end
block = segy_read("marmousi_2D.segy")
d_lin = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.03)    # 30 Hz wavelet
q = judiVector(src_geometry, wavelet)

# Set up info structure
ntComp = get_computational_nt(q.geometry, d_lin.geometry, model0)  # no. of computational time steps
info = Info(prod(model0.n), d_lin.nsrc, ntComp)

###################################################################################################

# Setup operators
opt = Options(optimal_checkpointing=false, subsampling_factor=3)
M = judiModeling(model0, q.geometry, d_lin.geometry; options=opt)
J = judiJacobian(M, q)

batchsize = 4
# Right-hand preconditioners (model topmute)
Mr = judiTopmute(model0.n, 52, 10)

C = joCurvelet2D(model0.n[1], model0.n[2]; zero_finest = true, DDT = Float32, RDT = Float32)

function breg_obj(x)
    i = randperm(d_lin.nsrc)[1:batchsize]
    Ml = judiMarineTopmute2D(30, d_lin[i].geometry)
    r = Ml*J[i]*Mr*x - Ml*d_lin[i]
    g = adjoint(Mr)*adjoint(J[i])*adjoint(Ml)*r
    return  .5f0*norm(r)^2, g[1:end]
end

opt = bregman_options(maxIter=5, verbose=2, quantile=.9, alpha=1, antichatter=true, TD=C)#, spg=true)
sol = bregman(breg_obj, 0f0.*vec(m0), opt)