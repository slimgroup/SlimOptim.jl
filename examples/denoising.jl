using SlimOptim, LinearAlgebra, JOLI, TestImages

img = Float32.(testimage("lena_gray_16bit.png")[1:2:end, 1:2:end])

# Sparse in wavelet domain
W = joDWT(128, 128; DDT=Float32, RDT=Float32)
# Or with curvelet ifi nstalled
# W = joCurvelet2D(128, 128; DDT=Float32, RDT=Float32)
A = joGaussian(200*137, 128*128; DDT=Float32, RDT=Float32)

# Make noisy data
b = A*vec(img)

# setup bregamn
opt = bregman_options(maxIter=200, verbose=2, quantile=.5)

sol = bregman(A, W, zeros(Float32, 128*128), b, opt)

figure();
subplot(221)
imshow(img, cmap="Greys", vmin=-.5, vmax=.5)
title("True")
subplot(222)
imshow(reshape(b, 200, 137), cmap="Greys", vmin=-.5, vmax=.5)
title("Measurment")
subplot(223)
imshow(reshape(sol.sol, 128, 128), cmap="Greys", vmin=-.5, vmax=.5)
title("recovered")
subplot(224)
imshow(img - reshape(sol.sol, 128, 128), cmap="Greys", vmin=-.05, vmax=.05)
title("Difference x10")
tight_layout()
