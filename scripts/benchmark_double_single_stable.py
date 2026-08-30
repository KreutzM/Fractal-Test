from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import mpmath as mp
import numpy as np
from numba import cuda

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_double_single as legacy
from fractal_flight_studio.research import double_single as ds


@dataclass(frozen=True, slots=True)
class StableBenchmarkConfig:
    target_id: str = "seahorse-valley"
    width: int = 1280
    height: int = 720
    max_iterations: int | None = None
    repeats: int = 9
    warmup_launches: int = 3
    warmup_seconds: float = 1.0
    batch_target_seconds: float = 0.25
    min_batch_launches: int = 4
    max_batch_launches: int = 256
    reference_samples: int = 24
    reference_bits: int = 256
    inspect_assembly: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("dimensions must be positive")
        if self.repeats < 3:
            raise ValueError("repeats must be at least three")
        if self.warmup_launches < 0:
            raise ValueError("warmup launches must not be negative")
        if self.warmup_seconds < 0.0:
            raise ValueError("warmup seconds must not be negative")
        if self.batch_target_seconds <= 0.0:
            raise ValueError("batch target seconds must be positive")
        if self.min_batch_launches < 1:
            raise ValueError("minimum batch launches must be positive")
        if self.max_batch_launches < self.min_batch_launches:
            raise ValueError("maximum batch launches must not be smaller than minimum")
        if self.reference_samples < 1:
            raise ValueError("reference samples must be at least one")
        if self.reference_bits < 96:
            raise ValueError("reference bits must be at least 96")


def make_ds_performance_kernel(
    *,
    include_lo_lo: bool,
    escape_mode: int,
    generic_multiply: bool,
):
    """Create a production-shaped DS kernel without diagnostic orbit output."""

    include_lo_lo_constant = bool(include_lo_lo)
    escape_mode_constant = int(escape_mode)
    generic_multiply_constant = bool(generic_multiply)

    @cuda.jit
    def kernel(
        escape_iterations,
        smooth_iterations,
        width,
        height,
        x0_hi,
        x0_lo,
        y0_hi,
        y0_lo,
        dx_hi,
        dx_lo,
        dy_hi,
        dy_lo,
        max_iterations,
        escape_squared,
    ):
        px, py = cuda.grid(2)
        if px >= width or py >= height:
            return
        crh, crl = ds.ds_coordinate(x0_hi, x0_lo, dx_hi, dx_lo, px)
        cih, cil = ds.ds_coordinate(y0_hi, y0_lo, dy_hi, dy_lo, py)
        iteration, smooth, _, _, _, _ = ds.ds_mandelbrot_point(
            crh,
            crl,
            cih,
            cil,
            max_iterations,
            escape_squared,
            include_lo_lo_constant,
            escape_mode_constant,
            generic_multiply_constant,
        )
        escape_iterations[py, px] = iteration
        smooth_iterations[py, px] = smooth

    return kernel


PERFORMANCE_KERNELS = {
    "fp32": ds.mandelbrot_f32_kernel,
    "fp64": ds.mandelbrot_f64_kernel,
    "ds-generic-full": make_ds_performance_kernel(
        include_lo_lo=True,
        escape_mode=ds.ESCAPE_FULL,
        generic_multiply=True,
    ),
    "ds-specialized-no-lo2-high": make_ds_performance_kernel(
        include_lo_lo=False,
        escape_mode=ds.ESCAPE_HIGH_ONLY,
        generic_multiply=False,
    ),
    "ds-specialized-lo2-high": make_ds_performance_kernel(
        include_lo_lo=True,
        escape_mode=ds.ESCAPE_HIGH_ONLY,
        generic_multiply=False,
    ),
    "ds-specialized-lo2-full": make_ds_performance_kernel(
        include_lo_lo=True,
        escape_mode=ds.ESCAPE_FULL,
        generic_multiply=False,
    ),
    "ds-specialized-lo2-adaptive": make_ds_performance_kernel(
        include_lo_lo=True,
        escape_mode=ds.ESCAPE_ADAPTIVE,
        generic_multiply=False,
    ),
}


def _selected_target(config: StableBenchmarkConfig):
    target = legacy.deep_zoom_target(config.target_id)
    iterations = config.max_iterations or target.recommended_iterations
    return target, iterations


def _launch_geometry(config: StableBenchmarkConfig):
    target, _ = _selected_target(config)
    return ds.coordinate_components(
        target.center_x_text,
        target.center_y_text,
        target.view_width_text,
        config.width,
        config.height,
        precision_bits=config.reference_bits,
    )


def _performance_arguments(
    name: str,
    config: StableBenchmarkConfig,
    d_iterations,
    d_smooth,
):
    target, iterations = _selected_target(config)
    x0, y0, dx, dy = _launch_geometry(config)
    escape_squared = np.float32(4.0)
    if name == "fp32":
        return (
            d_iterations,
            d_smooth,
            config.width,
            config.height,
            np.float32(x0.as_float()),
            np.float32(y0.as_float()),
            np.float32(dx.as_float()),
            np.float32(dy.as_float()),
            iterations,
            escape_squared,
        )
    if name == "fp64":
        with mp.workprec(config.reference_bits):
            cx = mp.mpf(target.center_x_text)
            cy = mp.mpf(target.center_y_text)
            vw = mp.mpf(target.view_width_text)
            vh = vw * config.height / config.width
            step_x = vw / config.width
            step_y = vh / config.height
            x0_exact = cx - vw / 2 + step_x / 2
            y0_exact = cy - vh / 2 + step_y / 2
        return (
            d_iterations,
            d_smooth,
            config.width,
            config.height,
            np.float64(x0_exact),
            np.float64(y0_exact),
            np.float64(step_x),
            np.float64(step_y),
            iterations,
            np.float64(4.0),
        )
    return (
        d_iterations,
        d_smooth,
        config.width,
        config.height,
        x0.hi,
        x0.lo,
        y0.hi,
        y0.lo,
        dx.hi,
        dx.lo,
        dy.hi,
        dy.lo,
        iterations,
        escape_squared,
    )


def choose_batch_launches(
    per_launch_seconds: float,
    target_seconds: float,
    minimum: int,
    maximum: int,
) -> int:
    if per_launch_seconds <= 0.0:
        return maximum
    requested = math.ceil(target_seconds / per_launch_seconds)
    return max(minimum, min(maximum, requested))


def _timed_batch(kernel, blocks, threads, args, launches: int) -> float:
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record()
    for _ in range(launches):
        kernel[blocks, threads](*args)
    end.record()
    elapsed = legacy._sync_event_seconds(start, end)
    return elapsed / launches


def _warm_kernel(
    kernel,
    blocks,
    threads,
    args,
    config: StableBenchmarkConfig,
) -> int:
    started = time.perf_counter()
    launches = 0
    chunk = max(1, config.min_batch_launches)
    while launches < config.warmup_launches or time.perf_counter() - started < config.warmup_seconds:
        for _ in range(chunk):
            kernel[blocks, threads](*args)
        launches += chunk
        cuda.synchronize()
    return launches


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _benchmark_performance_kernel(
    name: str,
    config: StableBenchmarkConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    kernel = PERFORMANCE_KERNELS[name]
    target, iterations = _selected_target(config)
    shape = (config.height, config.width)
    d_iterations = cuda.device_array(shape, dtype=np.int32)
    d_smooth = cuda.device_array(shape, dtype=np.float32)
    threads = (16, 16)
    blocks = (
        (config.width + threads[0] - 1) // threads[0],
        (config.height + threads[1] - 1) // threads[1],
    )
    args = _performance_arguments(name, config, d_iterations, d_smooth)

    compile_started = time.perf_counter()
    kernel[blocks, threads](*args)
    cuda.synchronize()
    compile_and_first_seconds = time.perf_counter() - compile_started

    before_warmup = legacy._nvidia_smi_snapshot()
    actual_warmup_launches = _warm_kernel(kernel, blocks, threads, args, config)
    after_warmup = legacy._nvidia_smi_snapshot()

    pilot_launches = max(1, config.min_batch_launches)
    pilot_seconds = _timed_batch(kernel, blocks, threads, args, pilot_launches)
    batch_launches = choose_batch_launches(
        pilot_seconds,
        config.batch_target_seconds,
        config.min_batch_launches,
        config.max_batch_launches,
    )

    kernel_times: list[float] = []
    end_to_end_times: list[float] = []
    host_iterations = np.empty(shape, dtype=np.int32)
    host_smooth = np.empty(shape, dtype=np.float32)
    for _ in range(config.repeats):
        kernel_times.append(_timed_batch(kernel, blocks, threads, args, batch_launches))

        wall_started = time.perf_counter()
        kernel[blocks, threads](*args)
        d_iterations.copy_to_host(host_iterations)
        d_smooth.copy_to_host(host_smooth)
        cuda.synchronize()
        end_to_end_times.append(time.perf_counter() - wall_started)

    after_samples = legacy._nvidia_smi_snapshot()
    kernel_median = statistics.median(kernel_times)
    wall_median = statistics.median(end_to_end_times)
    pixels = config.width * config.height
    executed_iterations = int(np.sum(host_iterations, dtype=np.int64))
    record: dict[str, Any] = {
        "variant": name,
        "target": target.id,
        "size": [config.width, config.height],
        "max_iterations": iterations,
        "compile_and_first_launch_seconds": compile_and_first_seconds,
        "warmup_launches_requested": config.warmup_launches,
        "warmup_seconds_requested": config.warmup_seconds,
        "warmup_launches_actual": actual_warmup_launches,
        "pilot_seconds_per_launch": pilot_seconds,
        "batch_launches": batch_launches,
        "batch_target_seconds": config.batch_target_seconds,
        "kernel_seconds_median": kernel_median,
        "kernel_seconds_minimum": min(kernel_times),
        "kernel_seconds_maximum": max(kernel_times),
        "kernel_timing_cv": _coefficient_of_variation(kernel_times),
        "end_to_end_seconds_median": wall_median,
        "end_to_end_timing_cv": _coefficient_of_variation(end_to_end_times),
        "kernel_mpix_per_second": pixels / kernel_median / 1_000_000,
        "end_to_end_mpix_per_second": pixels / wall_median / 1_000_000,
        "iterations_per_second": executed_iterations / kernel_median,
        "kernel_samples_seconds": kernel_times,
        "end_to_end_samples_seconds": end_to_end_times,
        "nvidia_smi_before_warmup": before_warmup,
        "nvidia_smi_after_warmup": after_warmup,
        "nvidia_smi_after_samples": after_samples,
        "diagnostic_orbit_in_timed_kernel": False,
        "end_to_end_readback": ["escape_iterations", "smooth_iterations"],
    }
    record.update(legacy._resource_summary(kernel, threads[0] * threads[1]))
    record.update(legacy._assembly_summary(kernel, name, output_dir, config.inspect_assembly))
    return record, host_iterations.copy(), host_smooth.copy()


def _capture_diagnostic_orbit(
    name: str,
    config: StableBenchmarkConfig,
    expected_iterations: np.ndarray,
    expected_smooth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = ds.KERNEL_VARIANTS[name]
    shape = (config.height, config.width)
    d_iterations = cuda.device_array(shape, dtype=np.int32)
    d_smooth = cuda.device_array(shape, dtype=np.float32)
    d_orbit = cuda.device_array((config.height, config.width, 4), dtype=np.float32)
    threads = (16, 16)
    blocks = (
        (config.width + threads[0] - 1) // threads[0],
        (config.height + threads[1] - 1) // threads[1],
    )
    args = legacy._kernel_arguments(name, config, d_iterations, d_smooth, d_orbit)
    kernel[blocks, threads](*args)
    cuda.synchronize()
    host_iterations = d_iterations.copy_to_host()
    host_smooth = d_smooth.copy_to_host()
    host_orbit = d_orbit.copy_to_host()
    if not np.array_equal(host_iterations, expected_iterations):
        raise RuntimeError(f"{name} diagnostic and performance escape outputs differ")
    if not np.array_equal(host_smooth, expected_smooth):
        raise RuntimeError(f"{name} diagnostic and performance smooth outputs differ")
    return host_iterations, host_smooth, host_orbit


def run(config: StableBenchmarkConfig, output: Path) -> dict[str, Any]:
    if not legacy._cuda_available():
        raise RuntimeError(
            "CUDA is not available; run this benchmark on the physical NVIDIA GPU"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for name in PERFORMANCE_KERNELS:
        record, escape, smooth = _benchmark_performance_kernel(name, config, output_dir)
        if name.startswith("ds-"):
            escape, smooth, orbit = _capture_diagnostic_orbit(
                name, config, escape, smooth
            )
            record["diagnostic_matches_performance"] = True
        else:
            orbit = None
        records.append(record)
        outputs[name] = (escape, smooth, orbit)
        print(
            f"{name:34s} {record['kernel_seconds_median'] * 1000:9.3f} ms  "
            f"CV={record['kernel_timing_cv'] * 100:5.2f}%  "
            f"batch={record['batch_launches']:3d}  "
            f"regs={record.get('registers_per_thread', '?')}"
        )

    report = {
        "schema_version": 2,
        "purpose": (
            "isolated production-shaped CUDA FP32 double-single Mandelbrot "
            "feasibility study"
        ),
        "production_renderer_modified": False,
        "system": {
            "platform": legacy.platform.platform(),
            "python": legacy.platform.python_version(),
            "numpy": np.__version__,
            "numba": legacy.numba.__version__,
            "numba_cuda_module": getattr(cuda, "__file__", None),
            "device": legacy._device_metadata(),
            "nvidia_smi_snapshot": legacy._nvidia_smi_snapshot(),
        },
        "config": asdict(config),
        "grid": legacy._grid_report(config),
        "performance": records,
        "accuracy": legacy._accuracy_report(config, outputs, output_dir),
        "methodology": {
            "kernel_timing": (
                "CUDA events around a sustained batch of launches; elapsed time is "
                "divided by launch count and the median of batches is reported"
            ),
            "end_to_end_timing": (
                "one production-shaped kernel launch plus readback of escape and smooth "
                "outputs only; diagnostic orbit output is excluded"
            ),
            "cold_timing": "first launch includes JIT compilation and is reported separately",
            "warmup": (
                "unmeasured launches continue until both the launch-count and duration "
                "requirements are met"
            ),
            "accuracy_capture": (
                "diagnostic DS kernels with final-orbit output run once after timing and "
                "must exactly match production-shaped escape and smooth outputs"
            ),
            "fast_math": False,
            "fp64_role": (
                "correctness/performance comparison only; no production routing changes"
            ),
            "smooth_iteration": (
                "computed from FP32 high-component magnitude for DS variants"
            ),
        },
        "open_risks": [
            (
                "The adaptive escape band is conservative and experimentally validated "
                "here, not a proof of total accumulated orbit error."
            ),
            "Double-single extends mantissa precision but not the FP32 exponent range.",
            "SASS inspection requires nvdisasm on PATH.",
            (
                "A single target cannot establish the final direct-render precision policy; "
                "representative exterior, boundary, interior, and deeper-zoom targets remain "
                "necessary before production routing."
            ),
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clock-stabilized production-shaped FP32, FP64, and "
            "double-single Mandelbrot benchmark"
        )
    )
    parser.add_argument("--target", default="seahorse-valley")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup-launches", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--batch-target-seconds", type=float, default=0.25)
    parser.add_argument("--min-batch-launches", type=int, default=4)
    parser.add_argument("--max-batch-launches", type=int, default=256)
    parser.add_argument("--reference-samples", type=int, default=24)
    parser.add_argument("--reference-bits", type=int, default=256)
    parser.add_argument("--no-assembly", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("double-single-benchmark-results.json"),
    )
    args = parser.parse_args()
    try:
        config = StableBenchmarkConfig(
            target_id=args.target,
            width=args.width,
            height=args.height,
            max_iterations=args.iterations,
            repeats=args.repeats,
            warmup_launches=args.warmup_launches,
            warmup_seconds=args.warmup_seconds,
            batch_target_seconds=args.batch_target_seconds,
            min_batch_launches=args.min_batch_launches,
            max_batch_launches=args.max_batch_launches,
            reference_samples=args.reference_samples,
            reference_bits=args.reference_bits,
            inspect_assembly=not args.no_assembly,
        )
        run(config, args.output)
    except (KeyError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
