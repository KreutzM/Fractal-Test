from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time
from typing import Any

import numpy as np
from numba import cuda
import numba

from fractal_flight_studio.deep_zoom import (
    GLITCH_TOLERANCE,
    MIN_REFERENCE_MAGNITUDE_SQUARED,
    prepare_perturbation,
)
from fractal_flight_studio.models import FractalKind, Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.cuda import _cuda_perturb_kernel_f64
from fractal_flight_studio.research.double_single_perturbation import (
    double_single_perturbation_kernel,
    split_reference_orbit,
    split_scalar,
)


@dataclass(frozen=True, slots=True)
class Config:
    width: int = 1280
    height: int = 720
    center_x_text: str = "-0.743643887037151"
    center_y_text: str = "0.13182590420533"
    view_width_text: str = "0.0000000000005"
    max_iterations: int = 1200
    reference_bits: int = 384
    repeats: int = 7
    warmup_seconds: float = 1.0
    batch_target_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("dimensions must be positive")
        if self.max_iterations < 1 or self.reference_bits < 96:
            raise ValueError("iterations and reference precision are too small")
        if self.repeats < 1 or self.warmup_seconds < 0 or self.batch_target_seconds <= 0:
            raise ValueError("invalid timing configuration")


def _request(config: Config) -> RenderRequest:
    return RenderRequest(
        width=config.width,
        height=config.height,
        viewport=Viewport(
            float(config.center_x_text),
            float(config.center_y_text),
            float(config.view_width_text),
        ),
        fractal=FractalKind.MANDELBROT,
        max_iterations=config.max_iterations,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=config.reference_bits,
        center_x_text=config.center_x_text,
        center_y_text=config.center_y_text,
        view_width_text=config.view_width_text,
    )


def _nvidia_smi_snapshot() -> dict[str, str] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    query = "name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"query": query, "raw": completed.stdout.strip()}
    except Exception as exc:
        return {"error": repr(exc)}


def _event_seconds(start, end) -> float:
    end.synchronize()
    return float(cuda.event_elapsed_time(start, end)) / 1000.0


def _timed_batch(kernel, blocks, threads, args, launches: int) -> float:
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record()
    for _ in range(launches):
        kernel[blocks, threads](*args)
    end.record()
    return _event_seconds(start, end) / launches


def _warm(kernel, blocks, threads, args, seconds: float) -> int:
    started = time.perf_counter()
    launches = 0
    while launches < 2 or time.perf_counter() - started < seconds:
        kernel[blocks, threads](*args)
        launches += 1
        if launches % 8 == 0:
            cuda.synchronize()
    cuda.synchronize()
    return launches


def _batch_launches(per_launch_seconds: float, target_seconds: float) -> int:
    if per_launch_seconds <= 0:
        return 64
    return max(1, min(256, math.ceil(target_seconds / per_launch_seconds)))


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _normalize_resource(value: Any, reduction) -> tuple[int | None, dict[str, int] | None]:
    if isinstance(value, dict):
        converted = {str(k): int(v) for k, v in value.items()}
        return reduction(converted.values()) if converted else None, converted
    if value is None:
        return None, None
    return int(value), None


def _resources(kernel, threads_per_block: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, getter, reduction in (
        ("registers_per_thread", "get_regs_per_thread", max),
        ("static_shared_memory_bytes", "get_shared_mem_per_block", max),
        ("max_threads_per_block", "get_max_threads_per_block", min),
        ("local_memory_bytes_per_thread", "get_local_mem_per_thread", max),
    ):
        try:
            raw = getattr(kernel, getter)()
            value, by_signature = _normalize_resource(raw, reduction)
            result[name] = value
            if by_signature is not None:
                result[f"{name}_by_signature"] = by_signature
        except Exception as exc:
            result[f"{name}_error"] = repr(exc)
    try:
        active = cuda.current_context().get_active_blocks_per_multiprocessor(kernel, threads_per_block, 0)
        result["active_blocks_per_multiprocessor"] = int(active)
    except Exception as exc:
        result["occupancy_error"] = repr(exc)
    return result


def _ptx_summary(kernel) -> dict[str, Any]:
    try:
        raw = kernel.inspect_asm()
        text = "\n".join(raw.values()) if isinstance(raw, dict) else str(raw)
    except Exception as exc:
        return {"error": repr(exc)}
    arithmetic_f64 = len(re.findall(r"\b(?:add|sub|mul|fma|div|sqrt)(?:\.[a-z0-9_]+)*\.f64\b", text))
    return {
        "explicit_fma_f32": len(re.findall(r"\bfma\.rn\.f32\b", text)),
        "f64_arithmetic": arithmetic_f64,
        "local_loads": len(re.findall(r"\bld\.local\b", text)),
        "local_stores": len(re.findall(r"\bst\.local\b", text)),
        "ftz_mentions": len(re.findall(r"\.ftz\b", text)),
    }


def _variant_record(name, kernel, blocks, threads, args, config: Config):
    compile_started = time.perf_counter()
    kernel[blocks, threads](*args)
    cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started
    before_warmup = _nvidia_smi_snapshot()
    warmup_launches = _warm(kernel, blocks, threads, args, config.warmup_seconds)
    after_warmup = _nvidia_smi_snapshot()
    pilot = _timed_batch(kernel, blocks, threads, args, 1)
    launches = _batch_launches(pilot, config.batch_target_seconds)
    samples = [_timed_batch(kernel, blocks, threads, args, launches) for _ in range(config.repeats)]
    after_samples = _nvidia_smi_snapshot()
    return {
        "name": name,
        "compile_and_first_seconds": compile_seconds,
        "warmup_launches": warmup_launches,
        "batch_launches": launches,
        "kernel_samples_seconds": samples,
        "kernel_median_seconds": statistics.median(samples),
        "kernel_minimum_seconds": min(samples),
        "kernel_maximum_seconds": max(samples),
        "kernel_coefficient_of_variation": _cv(samples),
        "nvidia_smi_before_warmup": before_warmup,
        "nvidia_smi_after_warmup": after_warmup,
        "nvidia_smi_after_samples": after_samples,
        "resources": _resources(kernel, threads[0] * threads[1]),
        "ptx": _ptx_summary(kernel),
    }


def _copy_outputs(d_values, d_inside, d_glitch, d_rebase):
    return (
        d_values.copy_to_host(),
        d_inside.copy_to_host(),
        d_glitch.copy_to_host(),
        d_rebase.copy_to_host(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FP64 and FP32 double-single perturbation delta kernels")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--batch-target-seconds", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("double-single-perturbation-results.json"))
    args = parser.parse_args()
    config = Config(
        width=args.width,
        height=args.height,
        repeats=args.repeats,
        warmup_seconds=args.warmup_seconds,
        batch_target_seconds=args.batch_target_seconds,
    )
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    request = _request(config)
    request.validate()
    perturb = prepare_perturbation(request)
    split_orbit = split_reference_orbit(perturb.orbit_real, perturb.orbit_imag)
    orbit_real_reconstructed = split_orbit.real_hi.astype(np.float64) + split_orbit.real_lo.astype(np.float64)
    orbit_imag_reconstructed = split_orbit.imag_hi.astype(np.float64) + split_orbit.imag_lo.astype(np.float64)
    real_underflow = (perturb.orbit_real != 0.0) & (orbit_real_reconstructed == 0.0)
    imag_underflow = (perturb.orbit_imag != 0.0) & (orbit_imag_reconstructed == 0.0)

    shape = (config.height, config.width)
    d_values = cuda.device_array(shape, dtype=np.float32)
    d_inside = cuda.device_array(shape, dtype=np.bool_)
    d_glitch = cuda.device_array(shape, dtype=np.bool_)
    d_rebase = cuda.device_array(shape, dtype=np.bool_)
    d_orbit_real = cuda.to_device(perturb.orbit_real)
    d_orbit_imag = cuda.to_device(perturb.orbit_imag)
    d_real_hi = cuda.to_device(split_orbit.real_hi)
    d_real_lo = cuda.to_device(split_orbit.real_lo)
    d_imag_hi = cuda.to_device(split_orbit.imag_hi)
    d_imag_lo = cuda.to_device(split_orbit.imag_lo)

    threads = (16, 16)
    blocks = ((config.width + 15) // 16, (config.height + 15) // 16)
    fp64_args = (
        d_values, d_inside, d_glitch, d_rebase,
        config.width, config.height,
        np.float64(perturb.x0_rel), np.float64(perturb.y0_rel),
        np.float64(perturb.dx), np.float64(perturb.dy),
        config.max_iterations, np.float64(4.0),
        d_orbit_real, d_orbit_imag, perturb.reference_rebase_limit,
        np.float64(GLITCH_TOLERANCE), np.float64(MIN_REFERENCE_MAGNITUDE_SQUARED),
    )
    x0, y0, dx, dy = map(split_scalar, (perturb.x0_rel, perturb.y0_rel, perturb.dx, perturb.dy))
    ds_args = (
        d_values, d_inside, d_glitch, d_rebase,
        config.width, config.height,
        x0.hi, x0.lo, y0.hi, y0.lo, dx.hi, dx.lo, dy.hi, dy.lo,
        config.max_iterations, np.float32(4.0),
        d_real_hi, d_real_lo, d_imag_hi, d_imag_lo,
        perturb.reference_rebase_limit,
        np.float32(GLITCH_TOLERANCE), np.float32(np.finfo(np.float32).tiny),
    )

    fp64 = _variant_record("native-fp64", _cuda_perturb_kernel_f64, blocks, threads, fp64_args, config)
    _cuda_perturb_kernel_f64[blocks, threads](*fp64_args)
    cuda.synchronize()
    fp64_outputs = _copy_outputs(d_values, d_inside, d_glitch, d_rebase)

    ds = _variant_record("double-single", double_single_perturbation_kernel, blocks, threads, ds_args, config)
    double_single_perturbation_kernel[blocks, threads](*ds_args)
    cuda.synchronize()
    ds_outputs = _copy_outputs(d_values, d_inside, d_glitch, d_rebase)

    fp64_values, fp64_inside, fp64_glitch, fp64_rebase = fp64_outputs
    ds_values, ds_inside, ds_glitch, ds_rebase = ds_outputs
    value_delta = np.abs(ds_values.astype(np.float64) - fp64_values.astype(np.float64))
    report = {
        "schema_version": 1,
        "purpose": "isolated CUDA FP64 versus double-single perturbation delta benchmark",
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "device": str(cuda.current_context().device.name),
            "nvidia_smi": _nvidia_smi_snapshot(),
        },
        "config": asdict(config),
        "reference": {
            "orbit_length": int(perturb.orbit_real.size),
            "reference_rebase_limit": int(perturb.reference_rebase_limit),
            "fp64_transfer_bytes": int(perturb.orbit_real.nbytes + perturb.orbit_imag.nbytes),
            "double_single_transfer_bytes": int(split_orbit.transfer_bytes),
            "maximum_real_split_error": float(np.max(np.abs(orbit_real_reconstructed - perturb.orbit_real))),
            "maximum_imag_split_error": float(np.max(np.abs(orbit_imag_reconstructed - perturb.orbit_imag))),
            "underflowed_nonzero_real_components": int(np.count_nonzero(real_underflow)),
            "underflowed_nonzero_imag_components": int(np.count_nonzero(imag_underflow)),
            "native_min_reference_magnitude_squared": float(MIN_REFERENCE_MAGNITUDE_SQUARED),
            "double_single_min_reference_magnitude_squared": float(np.finfo(np.float32).tiny),
        },
        "variants": [fp64, ds],
        "comparison": {
            "fp64_over_double_single_kernel_speedup": fp64["kernel_median_seconds"] / ds["kernel_median_seconds"],
            "inside_mismatch_pixels": int(np.count_nonzero(ds_inside != fp64_inside)),
            "inside_mismatch_fraction": float(np.mean(ds_inside != fp64_inside)),
            "mean_absolute_value_delta": float(np.mean(value_delta)),
            "maximum_absolute_value_delta": float(np.max(value_delta)),
            "glitch_flag_mismatch_pixels": int(np.count_nonzero(ds_glitch != fp64_glitch)),
            "rebase_flag_mismatch_pixels": int(np.count_nonzero(ds_rebase != fp64_rebase)),
            "fp64_glitch_pixels": int(np.count_nonzero(fp64_glitch)),
            "double_single_glitch_pixels": int(np.count_nonzero(ds_glitch)),
            "fp64_rebase_pixels": int(np.count_nonzero(fp64_rebase)),
            "double_single_rebase_pixels": int(np.count_nonzero(ds_rebase)),
        },
        "interpretation": {
            "reference_orbit": "built on the CPU at arbitrary precision, then compared as FP64 versus two-FP32 transfer formats",
            "scope": "research only; production perturbation routing and kernel remain unchanged",
            "accuracy": "output differences are measured against the existing native FP64 perturbation kernel",
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
