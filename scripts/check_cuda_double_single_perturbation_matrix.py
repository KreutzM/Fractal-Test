from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import platform
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Iterable

import numba
import numpy as np

from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.auto import AdaptivePrecisionRenderer
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer


@dataclass(frozen=True, slots=True)
class ValidationCase:
    id: str
    category: str
    center_x_text: str
    center_y_text: str
    view_width_text: str
    max_iterations: int
    reference_bits: int
    expected_arithmetic: str
    expected_fallback_reason: str
    expected_reference_reused: bool | None = None


DEFAULT_CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        id="seahorse-baseline",
        category="filament",
        center_x_text="-0.743643887037151",
        center_y_text="0.13182590420533",
        view_width_text="5e-13",
        max_iterations=1200,
        reference_bits=384,
        expected_arithmetic="double-single",
        expected_fallback_reason="",
        expected_reference_reused=False,
    ),
    ValidationCase(
        id="seahorse-reference-reuse",
        category="reference-reuse",
        center_x_text="-0.743643887037151",
        center_y_text="0.13182590420533",
        view_width_text="1e-13",
        max_iterations=1200,
        reference_bits=384,
        expected_arithmetic="double-single",
        expected_fallback_reason="",
        expected_reference_reused=True,
    ),
    ValidationCase(
        id="seahorse-pan-new-reference",
        category="pan",
        center_x_text="-0.743643887037141",
        center_y_text="0.131825904205325",
        view_width_text="2e-13",
        max_iterations=1400,
        reference_bits=448,
        expected_arithmetic="double-single",
        expected_fallback_reason="",
        expected_reference_reused=False,
    ),
    ValidationCase(
        id="main-cardioid-cusp",
        category="parabolic-boundary",
        center_x_text="0.25",
        center_y_text="0.0",
        view_width_text="5e-13",
        max_iterations=1600,
        reference_bits=384,
        expected_arithmetic="double-single",
        expected_fallback_reason="",
        expected_reference_reused=False,
    ),
    ValidationCase(
        id="left-cardioid-cusp",
        category="cusp-boundary",
        center_x_text="-0.75",
        center_y_text="0.0",
        view_width_text="5e-13",
        max_iterations=1600,
        reference_bits=384,
        expected_arithmetic="double-single",
        expected_fallback_reason="",
        expected_reference_reused=False,
    ),
    ValidationCase(
        id="misiurewicz-minus-two",
        category="misiurewicz-boundary",
        center_x_text="-2.0",
        center_y_text="0.0",
        view_width_text="5e-13",
        max_iterations=1200,
        reference_bits=384,
        expected_arithmetic="double-single",
        expected_fallback_reason="",
        expected_reference_reused=False,
    ),
    ValidationCase(
        id="reference-magnitude-fallback",
        category="guarded-fallback",
        center_x_text="1e-200",
        center_y_text="0.0",
        view_width_text="1e-220",
        max_iterations=600,
        reference_bits=512,
        expected_arithmetic="float64",
        expected_fallback_reason="reference-magnitude-range",
        expected_reference_reused=False,
    ),
)


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


def _request(case: ValidationCase, width: int, height: int, *, native_fp64: bool) -> RenderRequest:
    return RenderRequest(
        width=width,
        height=height,
        viewport=Viewport(
            float(case.center_x_text),
            float(case.center_y_text),
            float(case.view_width_text),
        ),
        max_iterations=case.max_iterations,
        precision=Precision.FLOAT64 if native_fp64 else Precision.FLOAT32,
        render_mode=RenderMode.PERTURBATION if native_fp64 else RenderMode.AUTO,
        reference_bits=case.reference_bits,
        center_x_text=case.center_x_text,
        center_y_text=case.center_y_text,
        view_width_text=case.view_width_text,
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


def _warm(renderer, request: RenderRequest, seconds: float):
    started = time.perf_counter()
    launches = 0
    result = None
    while time.perf_counter() - started < seconds or launches < 2:
        result = renderer.render(request)
        launches += 1
    assert result is not None
    return result, launches


def _measure_pair(
    auto_renderer,
    auto_request: RenderRequest,
    native_renderer,
    native_request: RenderRequest,
    repeats: int,
    warmup_seconds: float,
):
    auto_result, auto_warmups = _warm(auto_renderer, auto_request, warmup_seconds)
    native_result, native_warmups = _warm(native_renderer, native_request, warmup_seconds)
    auto_samples: list[float] = []
    native_samples: list[float] = []
    for index in range(repeats):
        if index % 2:
            native_result = native_renderer.render(native_request)
            auto_result = auto_renderer.render(auto_request)
        else:
            auto_result = auto_renderer.render(auto_request)
            native_result = native_renderer.render(native_request)
        auto_samples.append(float(auto_result.elapsed_seconds))
        native_samples.append(float(native_result.elapsed_seconds))
    return (
        auto_result,
        auto_warmups,
        _timing_summary(auto_samples),
        native_result,
        native_warmups,
        _timing_summary(native_samples),
    )


def summarize_cases(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    route_failures = [record["id"] for record in items if not record["route_matches"]]
    ds_comparisons = [
        record["comparison"]
        for record in items
        if record["expected_arithmetic"] == "double-single"
    ]
    return {
        "case_count": len(items),
        "double_single_case_count": len(ds_comparisons),
        "fallback_case_count": len(items) - len(ds_comparisons),
        "route_failures": route_failures,
        "routing_gate_passed": not route_failures,
        "minimum_fp64_over_double_single_speedup": (
            min(item["fp64_over_auto_speedup"] for item in ds_comparisons)
            if ds_comparisons
            else None
        ),
        "maximum_inside_mismatch_fraction": max(
            (item["inside_mismatch_fraction"] for item in ds_comparisons),
            default=None,
        ),
        "maximum_glitch_flag_mismatch_pixels": max(
            (item["glitch_flag_mismatch_pixels"] for item in ds_comparisons),
            default=None,
        ),
        "maximum_rebase_flag_mismatch_pixels": max(
            (item["rebase_flag_mismatch_pixels"] for item in ds_comparisons),
            default=None,
        ),
        "maximum_absolute_value_delta": max(
            (item["maximum_absolute_value_delta"] for item in ds_comparisons),
            default=None,
        ),
    }


def run(
    cases: Iterable[ValidationCase],
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

    cases = tuple(cases)
    if not cases:
        raise ValueError("at least one validation case is required")

    auto_backend = CudaRenderer()
    native_backend = CudaRenderer()
    if not auto_backend.is_available():
        raise RuntimeError("CUDA is not available")
    auto_renderer = AdaptivePrecisionRenderer(auto_backend)

    before = _nvidia_smi_snapshot()
    records: list[dict[str, Any]] = []
    for case in cases:
        auto_request = _request(case, width, height, native_fp64=False)
        native_request = _request(case, width, height, native_fp64=True)
        (
            auto_result,
            auto_warmups,
            auto_timing,
            native_result,
            native_warmups,
            native_timing,
        ) = _measure_pair(
            auto_renderer,
            auto_request,
            native_backend,
            native_request,
            repeats,
            warmup_seconds,
        )

        auto_glitch = np.array(auto_backend._glitch_host, copy=True)
        auto_rebase = np.array(auto_backend._rebase_host, copy=True)
        native_glitch = np.array(native_backend._glitch_host, copy=True)
        native_rebase = np.array(native_backend._rebase_host, copy=True)
        value_delta = np.abs(
            auto_result.values.astype(np.float64) - native_result.values.astype(np.float64)
        )
        inside_mismatches = int(np.count_nonzero(auto_result.inside != native_result.inside))
        auto_median = float(auto_timing["median_seconds"])
        native_median = float(native_timing["median_seconds"])
        comparison = {
            "fp64_over_auto_speedup": native_median / auto_median,
            "inside_mismatch_pixels": inside_mismatches,
            "inside_mismatch_fraction": inside_mismatches / (width * height),
            "mean_absolute_value_delta": float(np.mean(value_delta)),
            "maximum_absolute_value_delta": float(np.max(value_delta)),
            "glitch_flag_mismatch_pixels": int(np.count_nonzero(auto_glitch != native_glitch)),
            "rebase_flag_mismatch_pixels": int(np.count_nonzero(auto_rebase != native_rebase)),
            "native_fp64_glitch_pixels": int(np.count_nonzero(native_glitch)),
            "auto_glitch_pixels": int(np.count_nonzero(auto_glitch)),
            "native_fp64_rebase_pixels": int(np.count_nonzero(native_rebase)),
            "auto_rebase_pixels": int(np.count_nonzero(auto_rebase)),
        }

        actual_arithmetic = str(auto_result.details.get("arithmetic"))
        actual_fallback = str(auto_result.details.get("double_single_fallback_reason", ""))
        actual_reused = bool(auto_result.details.get("reference_reused"))
        route_matches = (
            actual_arithmetic == case.expected_arithmetic
            and auto_result.details.get("render_mode") == "perturbation"
            and actual_fallback == case.expected_fallback_reason
            and (
                case.expected_reference_reused is None
                or actual_reused == case.expected_reference_reused
            )
            and native_result.details.get("arithmetic") == Precision.FLOAT64.value
            and native_result.details.get("render_mode") == "perturbation"
        )
        records.append(
            {
                "id": case.id,
                "category": case.category,
                "case": asdict(case),
                "expected_arithmetic": case.expected_arithmetic,
                "actual_arithmetic": actual_arithmetic,
                "expected_fallback_reason": case.expected_fallback_reason,
                "actual_fallback_reason": actual_fallback,
                "expected_reference_reused": case.expected_reference_reused,
                "actual_reference_reused": actual_reused,
                "route_matches": route_matches,
                "auto": {
                    "warmup_launches": auto_warmups,
                    "timing": auto_timing,
                    "details": auto_result.details,
                },
                "native_fp64": {
                    "warmup_launches": native_warmups,
                    "timing": native_timing,
                    "details": native_result.details,
                },
                "comparison": comparison,
                "nvidia_smi_after_case": _nvidia_smi_snapshot(),
            }
        )
        print(
            f"{case.id:34s} route={actual_arithmetic:13s} "
            f"fallback={actual_fallback or '-':26s} "
            f"speedup={comparison['fp64_over_auto_speedup']:.3f}x "
            f"inside={comparison['inside_mismatch_fraction']:.4%}"
        )

    report = {
        "schema_version": 1,
        "purpose": "production CUDA double-single perturbation multi-target validation matrix",
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "nvidia_smi_before": before,
            "nvidia_smi_after": _nvidia_smi_snapshot(),
        },
        "config": {
            "width": width,
            "height": height,
            "repeats": repeats,
            "warmup_seconds": warmup_seconds,
        },
        "cases": records,
        "summary": summarize_cases(records),
        "interpretation": {
            "routing_gate": (
                "all cases must use the expected arithmetic, perturbation mode, fallback reason, "
                "and reference reuse state"
            ),
            "accuracy": (
                "inside, value, glitch and rebase differences are measured against explicit "
                "native FP64 perturbation; they are divergence metrics, not arbitrary-precision proof"
            ),
            "performance": (
                "AUTO and explicit FP64 order alternates after separate duration-based warm-up"
            ),
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate production AUTO double-single perturbation across multiple targets"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("double-single-perturbation-matrix.json"),
    )
    args = parser.parse_args()
    try:
        report = run(
            DEFAULT_CASES,
            args.width,
            args.height,
            args.repeats,
            args.warmup_seconds,
            args.output,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    summary = report["summary"]
    print(f"Routing gate passed: {summary['routing_gate_passed']}")
    print(f"Route failures: {summary['route_failures']}")
    print(f"Wrote {args.output}")
    return 0 if summary["routing_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
