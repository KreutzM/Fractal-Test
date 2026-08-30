from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any

import numba
import numpy as np

from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.auto import AdaptivePrecisionRenderer
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer


CENTER_X = "-0.743643887037151"
CENTER_Y = "0.13182590420533"
VIEW_WIDTH = "0.0000000000005"


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


def _request(width: int, height: int, *, native_fp64: bool) -> RenderRequest:
    return RenderRequest(
        width=width,
        height=height,
        viewport=Viewport(float(CENTER_X), float(CENTER_Y), float(VIEW_WIDTH)),
        max_iterations=1200,
        precision=Precision.FLOAT64 if native_fp64 else Precision.FLOAT32,
        render_mode=RenderMode.PERTURBATION if native_fp64 else RenderMode.AUTO,
        reference_bits=384,
        center_x_text=CENTER_X,
        center_y_text=CENTER_Y,
        view_width_text=VIEW_WIDTH,
    )


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    deviation = statistics.pstdev(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "coefficient_of_variation": deviation / mean if mean else 0.0,
    }


def _measure(renderer, request: RenderRequest, repeats: int, warmup_seconds: float):
    warmup_started = time.perf_counter()
    warmup_launches = 0
    result = None
    while time.perf_counter() - warmup_started < warmup_seconds or warmup_launches < 2:
        result = renderer.render(request)
        warmup_launches += 1

    samples: list[float] = []
    for _ in range(repeats):
        result = renderer.render(request)
        samples.append(float(result.elapsed_seconds))
    assert result is not None
    return result, warmup_launches, _timing_summary(samples)


def run(
    width: int,
    height: int,
    repeats: int,
    warmup_seconds: float,
    output: Path,
) -> dict[str, Any]:
    if width < 1 or height < 1:
        raise ValueError("dimensions must be positive")
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    if warmup_seconds < 0:
        raise ValueError("warmup seconds must not be negative")

    ds_backend = CudaRenderer()
    native_backend = CudaRenderer()
    if not ds_backend.is_available():
        raise RuntimeError("CUDA is not available")

    auto_request = _request(width, height, native_fp64=False)
    native_request = _request(width, height, native_fp64=True)
    auto_renderer = AdaptivePrecisionRenderer(ds_backend)

    before = _nvidia_smi_snapshot()
    ds_result, ds_warmups, ds_timing = _measure(
        auto_renderer, auto_request, repeats, warmup_seconds
    )
    ds_glitch = np.array(ds_backend._glitch_host, copy=True)
    ds_rebase = np.array(ds_backend._rebase_host, copy=True)
    between = _nvidia_smi_snapshot()
    fp64_result, fp64_warmups, fp64_timing = _measure(
        native_backend, native_request, repeats, warmup_seconds
    )
    fp64_glitch = np.array(native_backend._glitch_host, copy=True)
    fp64_rebase = np.array(native_backend._rebase_host, copy=True)
    after = _nvidia_smi_snapshot()

    if ds_result.details.get("arithmetic") != "double-single":
        raise RuntimeError(
            "AUTO perturbation did not route to double-single: "
            f"{ds_result.details.get('arithmetic')!r}"
        )
    if ds_result.details.get("render_mode") != "perturbation":
        raise RuntimeError("AUTO request did not route to perturbation")
    if ds_result.details.get("double_single_mode") != "perturbation":
        raise RuntimeError("AUTO request did not report the DS perturbation tier")
    if fp64_result.details.get("arithmetic") != Precision.FLOAT64.value:
        raise RuntimeError("explicit perturbation did not retain native FP64")

    value_delta = np.abs(
        ds_result.values.astype(np.float64) - fp64_result.values.astype(np.float64)
    )
    inside_mismatches = int(np.count_nonzero(ds_result.inside != fp64_result.inside))
    glitch_mismatches = int(np.count_nonzero(ds_glitch != fp64_glitch))
    rebase_mismatches = int(np.count_nonzero(ds_rebase != fp64_rebase))
    ds_median = float(ds_timing["median_seconds"])
    fp64_median = float(fp64_timing["median_seconds"])
    report = {
        "schema_version": 1,
        "purpose": "production CUDA double-single perturbation routing and FP64 comparison",
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "nvidia_smi_before": before,
            "nvidia_smi_between": between,
            "nvidia_smi_after": after,
        },
        "config": {
            "width": width,
            "height": height,
            "center_x_text": CENTER_X,
            "center_y_text": CENTER_Y,
            "view_width_text": VIEW_WIDTH,
            "max_iterations": auto_request.max_iterations,
            "reference_bits": auto_request.reference_bits,
            "repeats": repeats,
            "warmup_seconds": warmup_seconds,
        },
        "double_single": {
            "warmup_launches": ds_warmups,
            "timing": ds_timing,
            "details": ds_result.details,
        },
        "native_fp64": {
            "warmup_launches": fp64_warmups,
            "timing": fp64_timing,
            "details": fp64_result.details,
        },
        "comparison": {
            "fp64_over_double_single_speedup": fp64_median / ds_median,
            "inside_mismatch_pixels": inside_mismatches,
            "inside_mismatch_fraction": inside_mismatches / (width * height),
            "mean_absolute_value_delta": float(np.mean(value_delta)),
            "maximum_absolute_value_delta": float(np.max(value_delta)),
            "glitch_flag_mismatch_pixels": glitch_mismatches,
            "rebase_flag_mismatch_pixels": rebase_mismatches,
            "fp64_glitch_pixels": int(np.count_nonzero(fp64_glitch)),
            "double_single_glitch_pixels": int(np.count_nonzero(ds_glitch)),
            "fp64_rebase_pixels": int(np.count_nonzero(fp64_rebase)),
            "double_single_rebase_pixels": int(np.count_nonzero(ds_rebase)),
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check production AUTO double-single perturbation against native FP64"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("double-single-perturbation-production-check.json"),
    )
    args = parser.parse_args()
    try:
        report = run(
            args.width,
            args.height,
            args.repeats,
            args.warmup_seconds,
            args.output,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    comparison = report["comparison"]
    print(
        "Double-single perturbation production speedup over native FP64: "
        f"{comparison['fp64_over_double_single_speedup']:.3f}x"
    )
    print(
        "Inside/glitch/rebase mismatches: "
        f"{comparison['inside_mismatch_pixels']}/"
        f"{comparison['glitch_flag_mismatch_pixels']}/"
        f"{comparison['rebase_flag_mismatch_pixels']}"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
