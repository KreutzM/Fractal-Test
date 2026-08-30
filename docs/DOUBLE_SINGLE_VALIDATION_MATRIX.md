# CUDA Double-Single production validation matrix

This check broadens the physical validation of the internal Mandelbrot Double-Single AUTO tier introduced by PR #29. It does not change rendering or routing policy.

## Cases

The default matrix deliberately combines different numerical behaviors:

- fast exterior escape;
- stable period-two interior;
- the main-cardioid cusp;
- Seahorse Valley boundary filaments;
- a deeper Seahorse satellite;
- a Double-Single case close to the existing FP64 direct-coordinate floor;
- a deeper case that must hand off to perturbation.

All camera values are stored as decimal text and passed through the production request path without first reducing them to FP64.

## Run on the physical NVIDIA GPU

```powershell
.\scripts\check_cuda_double_single_matrix.ps1
```

The default run uses 1280×720, five measured renders per arithmetic path, and a 0.5-second warm-up per path and case. DS and native-FP64 sample order alternates after warm-up to reduce order and clock bias.

The command writes `double-single-validation-matrix.json`.

## Interpretation

The routing gate is strict: every case must use the expected production path. The six direct comparison cases must select Double-Single, while the deepest transition case must select perturbation/native FP64 arithmetic.

For Double-Single cases the report records:

- production timing and timing variation;
- explicit native-FP64 timing;
- FP64-over-DS speedup;
- full-frame inside-mask mismatch fraction;
- mean and maximum normalized value delta;
- exact production metadata and per-case GPU clock snapshots.

Frame differences are comparisons against native FP64, not pixelwise arbitrary-precision proof. Large maximum value deltas can occur at isolated chaotic boundary pixels, so mismatch fraction and mean delta are the primary frame-wide indicators.

The matrix is a validation step before changing automatic routing thresholds. It does not itself broaden routing, expose a new precision option, or modify perturbation.
