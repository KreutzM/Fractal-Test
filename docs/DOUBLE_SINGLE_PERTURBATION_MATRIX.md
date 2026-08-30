# Double-Single perturbation production matrix

This validation-only matrix broadens the physical coverage of the guarded CUDA Double-Single perturbation tier. It does not change renderer logic or routing thresholds.

## Coverage

The default 1280×720 run includes:

- the physically validated Seahorse baseline at view width `5e-13`;
- a tighter view that must reuse the same high-precision CPU reference orbit;
- a nearby pan that must construct a new reference orbit;
- the main-cardioid cusp;
- the left cardioid cusp;
- the Misiurewicz boundary at `-2`;
- an exact tiny-center case whose nonzero reference orbit is outside the FP32 exponent range and must fall back to native FP64 with `reference-magnitude-range`.

Every case compares the production `AUTO` route with explicit native FP64 perturbation. The report contains timing, routing metadata, reference reuse, full-frame inside/value differences, glitch flags, rebase flags, and GPU clock snapshots.

## Run on the RTX 3060

```powershell
.\scripts\check_cuda_double_single_perturbation_matrix.ps1
```

The command writes `double-single-perturbation-matrix.json`.

The routing gate requires the expected arithmetic path, perturbation mode, fallback reason, and reference-reuse state for every case. Numerical differences remain reported as divergence from native FP64 rather than arbitrary-precision proof.
