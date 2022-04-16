# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020


using SlimOptim, LinearAlgebra, JOLI, TestImages, PyPlot
import TestImages: Gray

img = Float32.(Gray.(testimage("lighthouse.png")[1:2:end, 129:2:end-128]))

n = 256
k = 4
# Sparse in wavelet domain
W = joDWT(n, n; DDT=Float32, RDT=Float32)
# Or with curvelet if installed
# W = joCurvelet2D(128, 128; DDT=Float32, RDT=Float32)
A = vcat([joRomberg(n, n; DDT=Float32, RDT=Float32) for i=1:k]...)

# Make noisy data
imgn= img .+ .01f0*randn(Float32, size(img))
b = A*vec(imgn)

# setup bregamn
opt = bregman_options(maxIter=200, verbose=2, alpha=1, antichatter=true)
opt2 = bregman_options(maxIter=200, verbose=2, alpha=1, antichatter=true, spg=true)

sol = bregman(A, zeros(Float32, n*n), b; options=opt, TD=W, perc=.5)
sol2 = bregman(A, zeros(Float32, n*n), b; options=opt2, TD=W, perc=.5)

figure()
subplot(121)
plot(sol.ϕ_trace, label="std");plot(sol2.ϕ_trace, label="spg");
legend()
subplot(122)
loglog(sol.r_trace, label="std");loglog(sol2.r_trace, label="spg");
legend()


figure();
subplot(221)
imshow(img, cmap="Greys", vmin=-.5, vmax=.5)
title("True")
subplot(222)
imshow(reshape(b, n, k*n), cmap="Greys", vmin=-.5, vmax=.5)
title("Measurment")
subplot(223)
imshow(reshape(sol2.x, n, n), cmap="Greys", vmin=-.5, vmax=.5)
title("recovered")
subplot(224)
imshow(img - reshape(sol2.x, n, n), cmap="Greys", vmin=-.05, vmax=.05)
title("Difference x10")
tight_layout()
