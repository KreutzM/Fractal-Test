# FP64 perturbation benchmark

Use the dedicated benchmark to compare the CPU and CUDA implementations of the same forced Mandelbrot perturbation request:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_perturbation.py --backend all --repeats 3
```

The default workload uses the packaged `seahorse-satellite` target at 1280x720 with its recommended iteration count and reference precision. A 1080p run can be started with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_perturbation.py `
  --backend all `
  --target seahorse-satellite `
  --width 1920 `
  --height 1080 `
  --repeats 5 `
  --output perturbation-1080p.json
```

## Measurement method

Each backend receives a fresh renderer instance. Before timing the target, the benchmark performs a same-size forced-perturbation render at a distant location. This compiles Numba kernels, initializes the CUDA context when present, and allocates target-sized buffers without warming the measured target reference.

The benchmark then records:

- `cold_wall_seconds`: first target frame with a newly built reference orbit;
- `warm_wall_seconds_median`: median target frame after the reference is reused;
- `warm_fps` and `warm_mpix_per_second` based on wall time;
- renderer details such as reference upload time, backend precision and active device;
- `rgb_sha256` for CPU/CUDA output comparison;
- the active Numba thread count.

Linear tone mapping is used so automatic tone-state analysis does not distort the CPU/GPU perturbation comparison. Wall time includes reference preparation, fractal computation, colorization and required host readback.

## Reading the comparison

The JSON report contains:

- `warm_cuda_speedup_over_cpu = CPU warm time / CUDA warm time`;
- `cold_cuda_speedup_over_cpu = CPU cold time / CUDA cold time`.

A factor greater than `1.0` means CUDA is faster. A factor below `1.0` means the CPU is faster. `rgb_outputs_match` should be `true`; otherwise performance results should not be used for automatic backend selection until the output difference is understood.

Run the benchmark with no other heavy CPU or GPU workloads. For the Ryzen 9 5950X, repeat the CPU test with different `NUMBA_NUM_THREADS` values before deciding which thread count or backend should be selected automatically.
