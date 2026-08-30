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
from typing import Any, Callable

import mpmath as mp
import numpy as np
from PIL import Image
from numba import cuda
import numba

from fractal_flight_studio.deep_zoom_targets import deep_zoom_target
from fractal_flight_studio.research.double_single import (
    KERNEL_VARIANTS,
    coordinate_at,
    coordinate_components,
    grid_unique_fraction,
    mandelbrot_double_single_cpu,
    mandelbrot_reference,
    subnormal_relevance,
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    target_id: str = "seahorse-valley"
    width: int = 1280
    height: int = 720
    max_iterations: int | None = None
    repeats: int = 9
    warmup_launches: int = 3
    reference_samples: int = 24
    reference_bits: int = 256
    inspect_assembly: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("dimensions must be positive")
        if self.repeats < 1:
            raise ValueError("repeats must be at least one")
        if self.warmup_launches < 0:
            raise ValueError("warmup launches must not be negative")
        if self.reference_samples < 1:
            raise ValueError("reference samples must be at least one")
        if self.reference_bits < 96:
            raise ValueError("reference bits must be at least 96")


def _cuda_available() -> bool:
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def _json_version(value: Any) -> int | list[int]:
    """Normalize CUDA version APIs that return either an integer or a tuple."""

    if isinstance(value, (tuple, list)):
        return [int(part) for part in value]
    return int(value)


def _device_metadata() -> dict[str, Any]:
    context = cuda.current_context()
    device = context.device
    raw_name = device.name
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode(errors="replace")
    data: dict[str, Any] = {
        "name": str(raw_name),
        "compute_capability": list(device.compute_capability),
        "multiprocessor_count": int(getattr(device, "MULTIPROCESSOR_COUNT", 0)),
        "max_threads_per_multiprocessor": int(
            getattr(device, "MAX_THREADS_PER_MULTIPROCESSOR", 0)
        ),
        "warp_size": int(getattr(device, "WARP_SIZE", 32)),
    }
    try:
        free_memory, total_memory = context.get_memory_info()
        data["free_memory_bytes"] = int(free_memory)
        data["total_memory_bytes"] = int(total_memory)
    except Exception as exc:
        data["memory_info_error"] = repr(exc)
    try:
        data["cuda_runtime_version"] = _json_version(cuda.runtime.get_version())
    except Exception as exc:
        data["cuda_runtime_version_error"] = repr(exc)
    try:
        from numba.cuda.cudadrv import driver

        data["cuda_driver_version"] = _json_version(driver.driver.get_version())
    except Exception as exc:
        data["cuda_driver_version_error"] = repr(exc)
    return data


def _selected_target(config: BenchmarkConfig):
    target = deep_zoom_target(config.target_id)
    iterations = config.max_iterations or target.recommended_iterations
    return target, iterations


def _launch_geometry(config: BenchmarkConfig):
    target, _ = _selected_target(config)
    return coordinate_components(
        target.center_x_text,
        target.center_y_text,
        target.view_width_text,
        config.width,
        config.height,
        precision_bits=config.reference_bits,
    )


def _kernel_arguments(name: str, config: BenchmarkConfig, d_iterations, d_smooth, d_orbit):
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
        d_orbit,
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


def _sync_event_seconds(start, end) -> float:
    end.synchronize()
    return float(cuda.event_elapsed_time(start, end)) / 1000.0


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(str(item) for item in value.values())
    return str(value)


_PTX_OPCODE_PATTERN = re.compile(
    r"^\s*(?:@[!%A-Za-z0-9_.$]+\s+)?([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)\b",
    re.MULTILINE,
)


def _ptx_instruction_counts(ptx: str) -> dict[str, int]:
    lowered = ptx.lower()
    opcodes = [match.group(1).lower() for match in _PTX_OPCODE_PATTERN.finditer(ptx)]

    def family_count(family: str, scalar: str) -> int:
        prefix = f"{family}."
        suffix = f".{scalar}"
        return sum(opcode.startswith(prefix) and opcode.endswith(suffix) for opcode in opcodes)

    f64_families = ("fma", "mad", "mul", "add", "sub", "div", "sqrt", "rsqrt")
    return {
        "fma_rn_f32": sum(opcode == "fma.rn.f32" for opcode in opcodes),
        "fma_f32": family_count("fma", "f32"),
        "mul_f32": family_count("mul", "f32"),
        "add_f32": family_count("add", "f32"),
        "sub_f32": family_count("sub", "f32"),
        "f64_arithmetic": sum(
            family_count(family, "f64") for family in f64_families
        ),
        "f64_mentions": lowered.count(".f64"),
        "local_loads": sum(opcode.startswith("ld.local") for opcode in opcodes),
        "local_stores": sum(opcode.startswith("st.local") for opcode in opcodes),
        "ftz_mentions": lowered.count(".ftz"),
    }


def _assembly_summary(kernel, name: str, output_dir: Path, enabled: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not enabled:
        return result
    try:
        ptx = _extract_text(kernel.inspect_asm())
        ptx_path = output_dir / f"{name}.ptx"
        ptx_path.write_text(ptx, encoding="utf-8")
        counts = _ptx_instruction_counts(ptx)
        result["ptx_file"] = str(ptx_path)
        result["ptx_instruction_counts"] = counts
        if name.startswith("ds-"):
            result["ptx_validation"] = {
                "explicit_fp32_fma_present": counts["fma_rn_f32"] > 0,
                "no_fp64_arithmetic": counts["f64_arithmetic"] == 0,
                "no_ftz_modifier": counts["ftz_mentions"] == 0,
            }
    except Exception as exc:
        result["ptx_error"] = repr(exc)

    try:
        sass = _extract_text(kernel.inspect_sass())
        sass_path = output_dir / f"{name}.sass.txt"
        sass_path.write_text(sass, encoding="utf-8")
        upper = sass.upper()
        result["sass_file"] = str(sass_path)
        result["sass_instruction_counts"] = {
            "FFMA": upper.count("FFMA"),
            "FADD": upper.count("FADD"),
            "FMUL": upper.count("FMUL"),
            "DADD": upper.count("DADD"),
            "DMUL": upper.count("DMUL"),
            "LDL": upper.count("LDL"),
            "STL": upper.count("STL"),
        }
    except Exception as exc:
        result["sass_error"] = repr(exc)
    return result


def _normalize_resource_metric(
    value: Any,
    *,
    reducer: Callable[[list[int]], int],
) -> tuple[int, dict[str, int] | None]:
    """Normalize Numba resource APIs for specialized and generic dispatchers."""

    if isinstance(value, dict):
        by_signature = {str(signature): int(metric) for signature, metric in value.items()}
        if not by_signature:
            raise ValueError("kernel resource metric did not contain a compiled signature")
        return reducer(list(by_signature.values())), by_signature
    return int(value), None


def _capture_resource_metric(
    result: dict[str, Any],
    *,
    key: str,
    error_key: str,
    value: Any,
    reducer: Callable[[list[int]], int],
) -> None:
    try:
        normalized, by_signature = _normalize_resource_metric(value, reducer=reducer)
        result[key] = normalized
        if by_signature is not None:
            result[f"{key}_by_signature"] = by_signature
    except Exception as exc:
        result[error_key] = repr(exc)


def _resource_summary(kernel, threads_per_block: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    _capture_resource_metric(
        result,
        key="registers_per_thread",
        error_key="registers_error",
        value=kernel.get_regs_per_thread(),
        reducer=max,
    )
    _capture_resource_metric(
        result,
        key="static_shared_memory_bytes",
        error_key="shared_memory_error",
        value=kernel.get_shared_mem_per_block(),
        reducer=max,
    )
    _capture_resource_metric(
        result,
        key="max_threads_per_block",
        error_key="max_threads_error",
        value=kernel.get_max_threads_per_block(),
        reducer=min,
    )
    try:
        compiled = next(iter(kernel.overloads.values()))
        cufunc = compiled._codelibrary.get_cufunc()
        context = cuda.current_context()
        active = int(
            context.get_active_blocks_per_multiprocessor(cufunc, threads_per_block, 0)
        )
        device = context.device
        max_threads_sm = int(getattr(device, "MAX_THREADS_PER_MULTIPROCESSOR", 0))
        result["active_blocks_per_multiprocessor"] = active
        result["theoretical_thread_occupancy"] = (
            active * threads_per_block / max_threads_sm if max_threads_sm else None
        )
        result["local_memory_bytes_per_thread"] = int(compiled.local_mem_per_thread)
    except Exception as exc:
        result["occupancy_error"] = repr(exc)
    return result


def _benchmark_kernel(
    name: str,
    config: BenchmarkConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:
    kernel = KERNEL_VARIANTS[name]
    target, iterations = _selected_target(config)
    shape = (config.height, config.width)
    d_iterations = cuda.device_array(shape, dtype=np.int32)
    d_smooth = cuda.device_array(shape, dtype=np.float32)
    d_orbit = cuda.device_array((config.height, config.width, 4), dtype=np.float32)
    threads = (16, 16)
    blocks = (
        (config.width + threads[0] - 1) // threads[0],
        (config.height + threads[1] - 1) // threads[1],
    )
    args = _kernel_arguments(name, config, d_iterations, d_smooth, d_orbit)

    compile_started = time.perf_counter()
    kernel[blocks, threads](*args)
    cuda.synchronize()
    compile_and_first_seconds = time.perf_counter() - compile_started

    for _ in range(config.warmup_launches):
        kernel[blocks, threads](*args)
    cuda.synchronize()
    before_samples = _nvidia_smi_snapshot()

    kernel_times: list[float] = []
    end_to_end_times: list[float] = []
    host_iterations = np.empty(shape, dtype=np.int32)
    host_smooth = np.empty(shape, dtype=np.float32)
    host_orbit = np.empty((config.height, config.width, 4), dtype=np.float32)
    for _ in range(config.repeats):
        wall_started = time.perf_counter()
        start = cuda.event(timing=True)
        end = cuda.event(timing=True)
        start.record()
        kernel[blocks, threads](*args)
        end.record()
        kernel_seconds = _sync_event_seconds(start, end)
        d_iterations.copy_to_host(host_iterations)
        d_smooth.copy_to_host(host_smooth)
        if name.startswith("ds-"):
            d_orbit.copy_to_host(host_orbit)
        cuda.synchronize()
        kernel_times.append(kernel_seconds)
        end_to_end_times.append(time.perf_counter() - wall_started)
    after_samples = _nvidia_smi_snapshot()

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
        "warmup_launches": config.warmup_launches,
        "kernel_seconds_median": kernel_median,
        "end_to_end_seconds_median": wall_median,
        "kernel_mpix_per_second": pixels / kernel_median / 1_000_000,
        "end_to_end_mpix_per_second": pixels / wall_median / 1_000_000,
        "iterations_per_second": executed_iterations / kernel_median,
        "kernel_samples_seconds": kernel_times,
        "end_to_end_samples_seconds": end_to_end_times,
        "nvidia_smi_before_samples": before_samples,
        "nvidia_smi_after_samples": after_samples,
    }
    record.update(_resource_summary(kernel, threads[0] * threads[1]))
    record.update(_assembly_summary(kernel, name, output_dir, config.inspect_assembly))
    return (
        record,
        host_iterations.copy(),
        host_smooth.copy(),
        host_orbit.copy() if name.startswith("ds-") else None,
    )


def _sample_indices(width: int, height: int, count: int) -> list[tuple[int, int]]:
    candidates = {
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
        (width // 2, height // 2),
    }
    rng = np.random.default_rng(20260729)
    while len(candidates) < min(count, width * height):
        candidates.add((int(rng.integers(0, width)), int(rng.integers(0, height))))
    return sorted(candidates)[:count]


def _exact_pixel(
    target,
    width: int,
    height: int,
    px: int,
    py: int,
    bits: int,
) -> tuple[mp.mpf, mp.mpf]:
    with mp.workprec(bits):
        vw = mp.mpf(target.view_width_text)
        vh = vw * height / width
        dx = vw / width
        dy = vh / height
        x = mp.mpf(target.center_x_text) - vw / 2 + dx / 2 + px * dx
        y = mp.mpf(target.center_y_text) - vh / 2 + dy / 2 + py * dy
        return +x, +y


def _accuracy_report(
    config: BenchmarkConfig,
    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    output_dir: Path,
) -> dict[str, Any]:
    target, iterations = _selected_target(config)
    sample_records: list[dict[str, Any]] = []
    orbit_max_errors: list[float] = []
    x0, y0, dx, dy = _launch_geometry(config)
    ds_components = [x0, y0, dx, dy]
    for px, py in _sample_indices(config.width, config.height, config.reference_samples):
        x_exact, y_exact = _exact_pixel(
            target,
            config.width,
            config.height,
            px,
            py,
            config.reference_bits,
        )
        reference = mandelbrot_reference(
            mp.nstr(x_exact, 100),
            mp.nstr(y_exact, 100),
            iterations,
            precision_bits=config.reference_bits,
        )
        cr = coordinate_at(x0, dx, px)
        ci = coordinate_at(y0, dy, py)
        ds_components.extend((cr, ci))
        ds_cpu = mandelbrot_double_single_cpu(cr, ci, iterations)
        per_variant: dict[str, Any] = {}
        for name, (escape, smooth, orbit) in outputs.items():
            actual_iteration = int(escape[py, px])
            actual_smooth = float(smooth[py, px])
            per_variant[name] = {
                "escape_iteration": actual_iteration,
                "escape_iteration_delta": actual_iteration - reference.escape_iteration,
                "smooth_iteration": actual_smooth,
                "smooth_iteration_delta": actual_smooth - reference.smooth_iteration,
                "escaped_matches": (actual_iteration < iterations) == reference.escaped,
            }
            if orbit is not None and ds_cpu.orbit:
                final = orbit[py, px]
                ref_zr, ref_zi = reference.orbit[
                    min(len(reference.orbit), len(ds_cpu.orbit)) - 1
                ]
                zr_error = abs(
                    (float(final[0]) + float(final[1])) - float(mp.mpf(ref_zr))
                )
                zi_error = abs(
                    (float(final[2]) + float(final[3])) - float(mp.mpf(ref_zi))
                )
                per_variant[name]["final_orbit_absolute_error"] = max(
                    zr_error, zi_error
                )
        for (zr_ds, zi_ds), (zr_ref, zi_ref) in zip(ds_cpu.orbit, reference.orbit):
            error = max(
                abs(zr_ds.as_float() - float(mp.mpf(zr_ref))),
                abs(zi_ds.as_float() - float(mp.mpf(zi_ref))),
            )
            orbit_max_errors.append(error)
        sample_records.append(
            {
                "pixel": [px, py],
                "coordinate": [mp.nstr(x_exact, 50), mp.nstr(y_exact, 50)],
                "reference_escape_iteration": reference.escape_iteration,
                "reference_smooth_iteration": reference.smooth_iteration,
                "variants": per_variant,
            }
        )

    summary: dict[str, Any] = {}
    for name, (escape, smooth, _) in outputs.items():
        reference_name = "fp64" if name != "fp64" and "fp64" in outputs else name
        ref_escape, ref_smooth, _ = outputs[reference_name]
        escape_delta = np.abs(escape.astype(np.int64) - ref_escape.astype(np.int64))
        smooth_delta = np.abs(
            smooth.astype(np.float64) - ref_smooth.astype(np.float64)
        )
        summary[name] = {
            "versus_frame_reference": reference_name,
            "escape_iteration_mismatch_pixels": int(np.count_nonzero(escape_delta)),
            "maximum_escape_iteration_delta": int(np.max(escape_delta)),
            "mean_absolute_smooth_delta": float(np.mean(smooth_delta)),
            "maximum_absolute_smooth_delta": float(np.max(smooth_delta)),
        }
        if name != reference_name:
            _write_error_map(escape_delta, output_dir / f"{name}-escape-error.png")
            _write_error_map(smooth_delta, output_dir / f"{name}-smooth-error.png")
    return {
        "high_precision_reference_bits": config.reference_bits,
        "sample_count": len(sample_records),
        "samples": sample_records,
        "frame_comparison_summary": summary,
        "cpu_ds_orbit_max_absolute_error": max(orbit_max_errors, default=0.0),
        "subnormal_analysis": subnormal_relevance(ds_components),
    }


def _write_error_map(values: np.ndarray, path: Path) -> None:
    finite = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    maximum = float(np.max(finite))
    if maximum <= 0.0:
        scaled = np.zeros(finite.shape, dtype=np.uint8)
    else:
        scaled = np.asarray(
            np.clip(np.log1p(finite) / math.log1p(maximum) * 255.0, 0, 255),
            dtype=np.uint8,
        )
    Image.fromarray(scaled).save(path)


def _grid_report(config: BenchmarkConfig) -> dict[str, Any]:
    target, _ = _selected_target(config)
    x0, y0, dx, dy = _launch_geometry(config)
    current = {
        "x_unique_fraction": grid_unique_fraction(x0, dx, config.width),
        "y_unique_fraction": grid_unique_fraction(y0, dy, config.height),
    }
    with mp.workprec(config.reference_bits):
        width = mp.mpf(target.view_width_text)
        last_unique = width
        first_failed = None
        for _ in range(96):
            x0_t, y0_t, dx_t, dy_t = coordinate_components(
                target.center_x_text,
                target.center_y_text,
                mp.nstr(width, 100),
                config.width,
                config.height,
                precision_bits=config.reference_bits,
            )
            if (
                grid_unique_fraction(x0_t, dx_t, config.width) < 1.0
                or grid_unique_fraction(y0_t, dy_t, config.height) < 1.0
            ):
                first_failed = width
                break
            last_unique = width
            width /= 2
    return {
        "coordinate_generation": (
            "CPU arbitrary precision -> explicit hi/lo -> "
            "GPU index * DS step + DS origin"
        ),
        "current": current,
        "smallest_tested_unique_view_width": mp.nstr(last_unique, 30),
        "first_failed_view_width": (
            mp.nstr(first_failed, 30) if first_failed is not None else None
        ),
        "index_limit": 1 << 24,
    }


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if executable is None:
        return None
    fields = "name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"query": fields, "raw": completed.stdout.strip()}
    except Exception as exc:
        return {"error": repr(exc)}


def run(config: BenchmarkConfig, output: Path) -> dict[str, Any]:
    if not _cuda_available():
        raise RuntimeError(
            "CUDA is not available; run this benchmark on the physical NVIDIA GPU"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for name in KERNEL_VARIANTS:
        record, escape, smooth, orbit = _benchmark_kernel(name, config, output_dir)
        records.append(record)
        outputs[name] = (escape, smooth, orbit)
        print(
            f"{name:34s} {record['kernel_seconds_median'] * 1000:9.3f} ms  "
            f"{record['kernel_mpix_per_second']:9.3f} MPix/s  "
            f"regs={record.get('registers_per_thread', '?')}"
        )

    report = {
        "schema_version": 1,
        "purpose": "isolated CUDA FP32 double-single Mandelbrot feasibility study",
        "production_renderer_modified": False,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "numba_cuda_module": getattr(cuda, "__file__", None),
            "device": _device_metadata(),
            "nvidia_smi_snapshot": _nvidia_smi_snapshot(),
        },
        "config": asdict(config),
        "grid": _grid_report(config),
        "performance": records,
        "accuracy": _accuracy_report(config, outputs, output_dir),
        "methodology": {
            "kernel_timing": "CUDA events around the kernel only; median of warm launches",
            "end_to_end_timing": (
                "host wall clock including kernel, output readback, and synchronization"
            ),
            "cold_timing": "first launch includes JIT compilation and is reported separately",
            "warmup": (
                "unmeasured launches run after compilation and before samples to reduce "
                "clock-ramp and first-use effects"
            ),
            "fast_math": False,
            "energy": (
                "not inferred from short samples; nvidia-smi is recorded only as an "
                "environment snapshot"
            ),
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
                "here, not a proof of total orbit error."
            ),
            "Double-single extends mantissa precision but not the FP32 exponent range.",
            "SASS inspection requires nvdisasm on PATH.",
            (
                "Short benchmark runs remain sensitive to GPU clocks and system load; "
                "inspect the per-variant nvidia-smi snapshots and repeat unstable runs."
            ),
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark isolated FP32, native FP64, and CUDA double-single "
            "Mandelbrot kernels"
        )
    )
    parser.add_argument("--target", default="seahorse-valley")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup-launches", type=int, default=3)
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
        config = BenchmarkConfig(
            target_id=args.target,
            width=args.width,
            height=args.height,
            max_iterations=args.iterations,
            repeats=args.repeats,
            warmup_launches=args.warmup_launches,
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
