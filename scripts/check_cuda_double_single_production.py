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

import numpy as np
import numba

from fractal_flight_studio.deep_zoom_targets import deep_zoom_target
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.auto import AdaptivePrecisionRenderer
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer


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


def _request(target_id: str, width: int, height: int, *, native_fp64: bool) -> RenderRequest:
    target = deep_zoom_target(target_id)
    return RenderRequest(
        width=width,
        height=height,
        viewport=Viewport(
            float(target.center_x_text),
            float(target.center_y_text),
            float(target.view_width_text),
        ),
        fractal=target.fractal,
        max_iterations=target.recommended_iterations,
        precision=Precision.FLOAT64 if native_fp64 else Precision.FLOAT32,
        render_mode=RenderMode.DIRECT if native_fp64 else RenderMode.AUTO,
        reference_bits=target.reference_bits,
        center_x_text=target.center_x_text,
        center_y_text=target.center_y_text,
        view_width_text=target.view_width_text,
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
    target_id: str,
    width: int,
    height: int,
    repeats: int,
    warmup_seconds: float,
    output: Path,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    if warmup_seconds < 0:
        raise ValueError("warmup seconds must not be negative")

    ds_backend = CudaRenderer()
    native_backend = CudaRenderer()
    if not ds_backend.is_available():
        raise RuntimeError("CUDA is not available")

    auto_request = _request(target_id, width, height, native_fp64=False)
    native_request = _request(target_id, width, height, native_fp64=True)
    auto_renderer = AdaptivePrecisionRenderer(ds_backend)

    before = _nvidia_smi_snapshot()
    ds_result, ds_warmups, ds_timing = _measure(
        auto_renderer, auto_request, repeats, warmup_seconds
    )
    between = _nvidia_smi_snapshot()
    fp64_result, fp64_warmups, fp64_timing = _measure(
        native_backend, native_request, repeats, warmup_seconds
    )
    after = _nvidia_smi_snapshot()

    if ds_result.details.get("arithmetic") != "double-single":
        raise RuntimeError(
            "AUTO request did not route to double-single: "
            f"{ds_result.details.get('arithmetic')!r}"
        )
    if fp64_result.details.get("arithmetic") != Precision.FLOAT64.value:
        raise RuntimeError(
            "explicit direct request did not retain native FP64: "
            f"{fp64_result.details.get('arithmetic')!r}"
        )

    value_delta = np.abs(
        ds_result.values.astype(np.float64) - fp64_result.values.astype(np.float64)
    )
    inside_mismatches = int(np.count_nonzero(ds_result.inside != fp64_result.inside))
    ds_median = float(ds_timing["median_seconds"])
    fp64_median = float(fp64_timing["median_seconds"])
    report = {
        "schema_version": 1,
        "purpose": "production CUDA double-single routing and FP64 comparison",
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
            "target_id": target_id,
            "width": width,
            "height": height,
            "max_iterations": auto_request.max_iterations,
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
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the production CUDA double-single AUTO route against native FP64"
    )
    parser.add_argument("--target", default="seahorse-valley")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("double-single-production-check.json"),
    )
    args = parser.parse_args()
    try:
        report = run(
            args.target,
            args.width,
            args.height,
            args.repeats,
            args.warmup_seconds,
            args.output,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    comparison = report["comparison"]
    print(
        "Double-single production speedup over native FP64: "
        f"{comparison['fp64_over_double_single_speedup']:.3f}x"
    )
    print(
        "Inside-mask mismatches: "
        f"{comparison['inside_mismatch_pixels']} "
        f"({comparison['inside_mismatch_fraction']:.4%})"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
