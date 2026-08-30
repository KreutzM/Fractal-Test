# CUDA double-single perturbation

The isolated research benchmark established that the native FP64 Mandelbrot perturbation delta recurrence can be replaced by FP32 double-single arithmetic on the validated RTX 3060 target. Mandelbrot `auto` rendering can therefore use an internal double-single delta kernel after the direct-coordinate tier hands off to perturbation.

The high-precision reference orbit is still built on the CPU. Reference selection, re-anchoring, precision budgeting and perturbation routing remain unchanged; only GPU recurrence, escape, glitch and rebase arithmetic changes.

## Conservative production routing

Double-single perturbation is attempted only when all of the following hold:

- the request uses Mandelbrot `auto` mode;
- adaptive precision has promoted the public precision to `float64`;
- perturbation has already been selected;
- both image dimensions are below `2**24`;
- every finite reference-orbit component can be represented as two FP32 values without nonzero underflow;
- every nonzero reference sample is large enough that the FP32-normal glitch floor cannot replace a smaller native-FP64 magnitude;
- the relative origin and pixel steps remain nonzero after FP32 high/low splitting.

Any failed guard delegates to the existing native FP64 perturbation kernel. Explicit `render_mode=perturbation` also remains native FP64 as a reference and safety path.

The four FP32 orbit arrays use the same transfer size as the two FP64 arrays. The measured gain comes from FP32 arithmetic throughput rather than reduced orbit bandwidth.

## Metadata

Results retain public `precision=float64` and `render_mode=perturbation`. Internal routing is exposed through:

- `arithmetic=double-single` or `float64`;
- `double_single_enabled`;
- `double_single_mode=perturbation` when active;
- `double_single_perturbation_enabled`;
- `double_single_fallback_reason` when AUTO retained FP64.

## Research benchmark

Run the isolated native-FP64 versus double-single kernel benchmark with:

```powershell
.\scripts\benchmark_double_single_perturbation.ps1
```

It writes `double-single-perturbation-results.json` with CUDA-event timings, PTX and resource inspection, reference split diagnostics, and full-frame inside/value/glitch/rebase differences.

## Physical production check

Run the actual AUTO production route with:

```powershell
.\scripts\check_cuda_double_single_perturbation_production.ps1
```

The default check renders the validated Seahorse deep-zoom target at 1280×720, 1200 iterations and 384 reference bits. It compares AUTO double-single perturbation with explicit native-FP64 perturbation and writes:

```text
double-single-perturbation-production-check.json
```

The report includes warm end-to-end timings, routing metadata, GPU clock snapshots and complete inside/value/glitch/rebase differences.
