# CUDA double-single production tier

The production CUDA renderer keeps the public precision model unchanged. Double-single is an internal arithmetic choice for eligible Mandelbrot frames, not a new user-facing precision enum.

## Routing

The adaptive renderer first applies the existing precision policy. An unsafe FP32 request in `auto` mode is promoted to public `float64`. CUDA then uses the double-single direct kernel only when all of these conditions hold:

- render mode is `auto`;
- effective precision is `float64`;
- fractal kind is Mandelbrot;
- width and height are below the exact FP32 integer-index limit `2**24`;
- the existing direct FP64 pixel-grid guard has not yet selected perturbation.

Every other case delegates to the existing renderer unchanged. In particular:

- explicit `direct + float64` remains native FP64 and can be used as a reference path;
- perturbation remains the deep-zoom path;
- Julia, Burning Ship, Multibrot and Newton remain on their existing FP32/FP64 kernels;
- values outside the FP32 exponent range fall back to native FP64.

## CPU/GPU boundary

Exact decimal camera values remain on the CPU. The CPU parses them with `mpmath`, computes pixel-center origins and steps, and splits each value into FP32 high and low components. The GPU receives those components directly, constructs each coordinate with double-single arithmetic, and never first rounds the camera to absolute FP64.

Arbitrary-precision perturbation reference-orbit construction, reference selection, caching, rebasing and glitch policy remain on the CPU or in their existing perturbation components.

## Kernel

The selected implementation matches the benchmark winner:

- two FP32 components per value;
- error-free `two_sum`-style addition;
- `cuda.fma` product residuals;
- retained `lo * lo` terms;
- compensated difference of squares;
- high-component escape test;
- no global fast math.

## Metadata

Results preserve `precision=float64` because that is the effective public precision tier. They additionally report:

- `arithmetic=double-single` when the internal DS kernel ran;
- `double_single_enabled=true` for that frame.

Native FP32 and FP64 fallback frames report their corresponding arithmetic name.

## Validation boundary

The initial production tier is deliberately conservative. Automatic routing should be expanded only after physical-GPU regression coverage includes representative exterior, boundary, interior and deeper-zoom targets. Native FP64 remains available for explicit comparisons and fallback.
