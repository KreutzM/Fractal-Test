# CUDA Double-Single Mandelbrot feasibility benchmark

This directory documents the isolated research code introduced for PR #28. It does not change renderer selection, the public `Precision` enum, the native FP64 path, or the existing perturbation kernel.

## Scope and CPU/GPU split

The experiment measures direct Mandelbrot rendering only. Exact decimal camera values are parsed on the CPU with `mpmath`, split into two FP32 components, and passed to CUDA as `(hi, lo)` pairs. The GPU builds each pixel coordinate as an index-based double-single expression; it never assembles the coordinate in FP64 and splits it afterwards.

The production perturbation architecture remains unchanged: arbitrary-precision reference-orbit construction, reference selection, re-anchoring, and precision policy stay on the CPU. Future GPU perturbation work must evaluate the orbit-transfer representation separately.

## Implemented arithmetic

`src/fractal_flight_studio/research/double_single.py` contains:

- error-free `two_sum`;
- `quick_two_sum`, with its `abs(a) >= abs(b)` precondition documented and checked by the CPU test helper;
- FMA-based `two_product` using `cuda.fma`, which Numba CUDA maps to round-to-nearest `fma.rn.f32` for FP32 arguments;
- double-single addition, subtraction, multiplication, multiplication by FP32, and square;
- a fully compensated difference-of-squares path for cancellation-sensitive Mandelbrot real updates;
- variants with and without `lo * lo`;
- index-based double-single coordinate generation;
- high-only, full double-single, and adaptive-band escape tests.

The cancellation-prone hot path uses general `two_sum` rather than assuming the `quick_two_sum` magnitude precondition. Global fast math is not enabled.

The adaptive escape band accounts for represented low components plus a conservative FP32 evaluation margin. It is intentionally treated as an experimental heuristic, not as a proof of total accumulated orbit error.

## Variants

The benchmark executes:

1. direct FP32;
2. direct native FP64;
3. generic double-single multiplication with full escape;
4. specialized double-single without `lo * lo`, high-only escape;
5. specialized double-single with `lo * lo`, high-only escape;
6. specialized double-single with `lo * lo`, full escape;
7. specialized double-single with `lo * lo`, adaptive escape.

Smooth iteration for double-single variants is derived from the FP32 high-component magnitude. The experiment therefore isolates orbit and classification precision rather than attempting double-single transcendental functions.

## Local RTX 3060 command

From PowerShell in the repository root:

```powershell
.\scripts\benchmark_double_single.ps1
```

The launcher now uses `benchmark_double_single_stable.py`. Its default physical-GPU run uses 1280×720 pixels, nine measured repetitions, at least one second of unmeasured warm-up per variant, and sustained timing batches targeting roughly 250 ms.

A smaller functional run can be started as:

```powershell
.\scripts\benchmark_double_single.ps1 `
    -Width 320 `
    -Height 180 `
    -Repeats 3 `
    -WarmupSeconds 0.25 `
    -BatchTargetSeconds 0.05 `
    -ReferenceSamples 12
```

The command writes `double-single-benchmark-results.json` and a sibling directory containing PTX, optional SASS, and grayscale error maps. SASS inspection requires `nvdisasm` on `PATH`; a missing tool is recorded rather than treated as a benchmark failure.

## Measurement method

The stable report separates:

- first launch including JIT compilation;
- duration-based unmeasured post-compilation warm-up;
- production-shaped kernels that write only escape and smooth outputs;
- sustained pure-kernel batches measured with CUDA events and divided by launch count;
- one-launch end-to-end wall time with identical escape/smooth readback for FP32, FP64, and DS;
- diagnostic DS orbit capture executed once after timing;
- an exact output-equality check between production-shaped and diagnostic DS kernels;
- pixels/s and executed iterations/s;
- timing minima, maxima, and coefficients of variation;
- registers per thread, local memory per thread, static shared memory, maximum threads per block, active blocks per SM, and theoretical thread occupancy;
- per-signature resource values when Numba returns generic-dispatcher dictionaries;
- PTX/SASS instruction counts, explicit FMA use, FP64 arithmetic, and local-memory loads/stores;
- deterministic high-precision sample comparisons;
- escape-iteration delta, smooth-iteration delta, final-orbit error, and CPU double-single orbit error growth;
- full-frame error maps relative to native FP64;
- coordinate-grid uniqueness and an empirical view-width floor;
- subnormal-component observations.

The PTX counter accepts both default-rounding opcodes such as `mul.f32` and explicit forms such as `mul.rn.f32`. The report distinguishes actual FP64 arithmetic instructions from incidental `.f64` text mentions.

`nvidia-smi` is captured before warm-up, after warm-up, and after the measured batches of every variant. These snapshots help identify unstable clocks or P-states but are not interpreted as energy measurements.

## Findings from the first two RTX 3060 runs

The PTX and resource inspection is positive:

- the DS kernels contain explicit `fma.rn.f32`;
- no DS variant contains FP64 arithmetic;
- specialized kernels use 36–38 registers per thread;
- no local memory was reported;
- six 256-thread blocks fit per SM, corresponding to full theoretical thread occupancy.

The 1280×720 run did not provide a fair final timing comparison. FP32, FP64, generic DS, and the no-`lo * lo` specialized kernel were sampled in P2 near 1.84–1.93 GHz. The later `lo * lo` variants were sampled in P3 near 0.66–0.77 GHz, and their individual launch times changed by factors of roughly three to five. A fixed count of three warm-up launches was therefore insufficient.

The full, adaptive, and high-only `lo * lo` variants produced byte-identical escape and smooth error maps for the tested frame. Their additional escape-test arithmetic did not improve classification on this target. The high-only escape test remains the production candidate until a broader target matrix demonstrates a benefit from the more expensive alternatives.

Across both runs, specialized DS remained clearly faster than native FP64 whenever clocks were comparable. The exact `lo * lo` versus no-`lo * lo` speed difference still requires the sustained-batch run.

The 1280×720 frame comparison also confirms that DS is an intermediate precision tier rather than a bit-identical FP64 replacement: the specialized variants differed from native FP64 at roughly 0.76% of pixels in escape iteration, compared with about 30.18% for FP32. The high-only `lo * lo` variant had a mean absolute smooth-iteration delta of about 0.393 relative to FP64. These differences were concentrated along chaotic boundary structures, and a single target is not sufficient to define final routing thresholds.

## CI behavior

CI runs CPU arithmetic/reference tests, validates PTX/resource/version parsing and stable batch selection with synthetic inputs, and launches a tiny kernel through the CUDA simulator in a subprocess. The simulator proves only that the device functions and kernel interfaces execute; it is never used as a performance result. Physical-GPU conclusions must come from the JSON report produced on the RTX 3060.

## Interpretation gate

A production double-single direct path should proceed only if the RTX 3060 report shows all of the following:

- explicit FP32 FMA in PTX and preferably SASS;
- no unexpected FP64 arithmetic in the specialized kernel;
- no material register spilling or occupancy collapse;
- stable, materially lower warm kernel and end-to-end time than native FP64;
- acceptable boundary classification and smooth-iteration errors across several representative targets;
- a useful coordinate-resolution interval between direct FP32 and perturbation.

The resource and generated-code gates are already satisfied by the second run. The stable timing gate and broader accuracy matrix remain open.

## Follow-on roadmap

1. **PR #28 — isolated benchmark:** this experiment only.
2. **Productive double-single direct path:** add an internal direct precision tier, exact CPU-to-DS camera transfer, production buffer integration, fallback policy, and renderer-level regressions. Keep native FP64 as reference/fallback.
3. **Mixed-precision perturbation:** preserve high-precision CPU reference-orbit construction while benchmarking FP64, double-single, triple-single, and scaled GPU transfer/evaluation formats. Do not assume the direct-kernel winner is also the best perturbation representation.
4. **Scaled perturbation:** introduce mantissa-plus-exponent or block-scaled deltas for the FP32 exponent limit, with explicit rebasing/glitch semantics and targeted fallback for difficult pixels.

## Primary references

- NVIDIA, *Floating Point and IEEE 754* and the PTX ISA description of `fma.rn.f32`.
- Numba CUDA kernel API documentation for `cuda.fma`, `inspect_asm`, `inspect_sass`, and register/resource inspection.
- T. Dekker, “A Floating-Point Technique for Extending the Available Precision,” *Numerische Mathematik* 18 (1971), 224–242.
- T. Ogita, S. M. Rump, and S. Oishi, “Accurate Sum and Dot Product,” *SIAM Journal on Scientific Computing* 26(6), 1955–1988 (2005), DOI 10.1137/030601818.
- M. Joldes, J.-M. Muller, and V. Popescu, “Tight and Rigorous Error Bounds for Basic Building Blocks of Double-Word Arithmetic,” *ACM TOMS* 44(2) (2017), DOI 10.1145/3121432.
