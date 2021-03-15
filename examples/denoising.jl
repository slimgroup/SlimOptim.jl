# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: December 2020


using SlimOptim, LinearAlgebra, JOLI, TestImages, PyPlot

img = Float32.(testimage("lena_gray_16bit.png")[1:2:end, 1:2:end])

n = 128
# Sparse in wavelet domain
W = joDWT(n, n; DDT=Float32, RDT=Float32)
# Or with curvelet ifi nstalled
# W = joCurvelet2D(128, 128; DDT=Float32, RDT=Float32)
A = [joRomberg(n, n; DDT=Float32, RDT=Float32);joRomberg(n, n; DDT=Float32, RDT=Float32)];

# Make noisy data
b = A*vec(img)

# setup bregamn
opt = bregman_options(maxIter=200, verbose=2, quantile=.1, alpha=1, antichatter=true)

sol = bregman(A, W, zeros(Float32, 128*128), b, opt)

figure();
subplot(221)
imshow(img, cmap="Greys", vmin=-.5, vmax=.5)
title("True")
subplot(222)
imshow(reshape(b, 128, 256), cmap="Greys", vmin=-.5, vmax=.5)
title("Measurment")
subplot(223)
imshow(reshape(sol.x, 128, 128), cmap="Greys", vmin=-.5, vmax=.5)
title("recovered")
subplot(224)
imshow(img - reshape(sol.x, 128, 128), cmap="Greys", vmin=-.05, vmax=.05)
title("Difference x10")
tight_layout()
