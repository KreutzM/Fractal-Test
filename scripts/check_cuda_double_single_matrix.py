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

from fractal_flight_studio.models import (
    FractalKind,
    Precision,
    RenderMode,
    RenderRequest,
    Viewport,
)
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
    reference_bits: int = 256
    expected_arithmetic: str = "double-single"
    expected_render_mode: str = "direct"


DEFAULT_CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        id="exterior-fast-escape",
        category="exterior",
        center_x_text="1.0",
        center_y_text="1.0",
        view_width_text="0.0001",
        max_iterations=256,
    ),
    ValidationCase(
        id="period-two-interior",
        category="interior",
        center_x_text="-1.0",
        center_y_text="0.0",
        view_width_text="0.0001",
        max_iterations=800,
    ),
    ValidationCase(
        id="main-cardioid-cusp",
        category="boundary",
        center_x_text="0.25",
        center_y_text="0.0",
        view_width_text="0.0001",
        max_iterations=1200,
    ),
    ValidationCase(
        id="seahorse-valley",
        category="filament",
        center_x_text="-0.743643887037",
        center_y_text="0.131825904205",
        view_width_text="0.0001",
        max_iterations=800,
    ),
    ValidationCase(
        id="seahorse-satellite",
        category="satellite",
        center_x_text="-0.74364386269",
        center_y_text="0.13182590271",
        view_width_text="0.00000013526",
        max_iterations=1200,
    ),
    ValidationCase(
        id="double-single-near-fp64-floor",
        category="transition",
        center_x_text="-0.743643887037151",
        center_y_text="0.13182590420533",
        view_width_text="0.000000000002",
        max_iterations=1200,
        reference_bits=384,
    ),
    ValidationCase(
        id="perturbation-handoff",
        category="transition",
        center_x_text="-0.743643887037151",
        center_y_text="0.13182590420533",
        view_width_text="0.0000000000005",
        max_iterations=1200,
        reference_bits=384,
        expected_arithmetic="float64",
        expected_render_mode="perturbation",
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
        fractal=FractalKind.MANDELBROT,
        max_iterations=case.max_iterations,
        precision=Precision.FLOAT64 if native_fp64 else Precision.FLOAT32,
        render_mode=RenderMode.DIRECT if native_fp64 else RenderMode.AUTO,
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


def _warm(renderer, request: RenderRequest, seconds: float) -> tuple[Any, int]:
    started = time.perf_counter()
    launches = 0
    result = None
    while time.perf_counter() - started < seconds or launches < 2:
        result = renderer.render(request)
        launches += 1
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


def _measure_one(renderer, request: RenderRequest, repeats: int, warmup_seconds: float):
    result, warmups = _warm(renderer, request, warmup_seconds)
    samples: list[float] = []
    for _ in range(repeats):
        result = renderer.render(request)
        samples.append(float(result.elapsed_seconds))
    return result, warmups, _timing_summary(samples)


def summarize_cases(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    route_failures = [record["id"] for record in items if not record["route_matches"]]
    comparisons = [record["comparison"] for record in items if record.get("comparison")]
    return {
        "case_count": len(items),
        "route_failures": route_failures,
        "routing_gate_passed": not route_failures,
        "comparison_count": len(comparisons),
        "minimum_fp64_over_double_single_speedup": (
            min(item["fp64_over_double_single_speedup"] for item in comparisons)
            if comparisons
            else None
        ),
        "maximum_inside_mismatch_fraction": (
            max(item["inside_mismatch_fraction"] for item in comparisons)
            if comparisons
            else None
        ),
        "mean_case_absolute_value_delta": (
            statistics.fmean(item["mean_absolute_value_delta"] for item in comparisons)
            if comparisons
            else None
        ),
        "maximum_absolute_value_delta": (
            max(item["maximum_absolute_value_delta"] for item in comparisons)
            if comparisons
            else None
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
    if width <= 0 or height <= 0:
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
        actual_arithmetic: str
        comparison: dict[str, Any] | None = None
        native_section: dict[str, Any] | None = None

        if case.expected_arithmetic == "double-single":
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
            actual_arithmetic = str(auto_result.details.get("arithmetic"))
            if native_result.details.get("arithmetic") != Precision.FLOAT64.value:
                raise RuntimeError(
                    f"{case.id}: explicit direct request did not retain native FP64"
                )
            value_delta = np.abs(
                auto_result.values.astype(np.float64)
                - native_result.values.astype(np.float64)
            )
            mismatches = int(np.count_nonzero(auto_result.inside != native_result.inside))
            auto_median = float(auto_timing["median_seconds"])
            native_median = float(native_timing["median_seconds"])
            comparison = {
                "fp64_over_double_single_speedup": native_median / auto_median,
                "inside_mismatch_pixels": mismatches,
                "inside_mismatch_fraction": mismatches / (width * height),
                "mean_absolute_value_delta": float(np.mean(value_delta)),
                "maximum_absolute_value_delta": float(np.max(value_delta)),
            }
            native_section = {
                "warmup_launches": native_warmups,
                "timing": native_timing,
                "details": native_result.details,
            }
        else:
            auto_result, auto_warmups, auto_timing = _measure_one(
                auto_renderer, auto_request, repeats, warmup_seconds
            )
            actual_arithmetic = str(auto_result.details.get("arithmetic"))

        records.append(
            {
                "id": case.id,
                "category": case.category,
                "case": asdict(case),
                "expected_arithmetic": case.expected_arithmetic,
                "actual_arithmetic": actual_arithmetic,
                "actual_render_mode": auto_result.details.get("render_mode"),
                "route_matches": (
                    actual_arithmetic == case.expected_arithmetic
                    and auto_result.details.get("render_mode") == case.expected_render_mode
                ),
                "auto": {
                    "warmup_launches": auto_warmups,
                    "timing": auto_timing,
                    "details": auto_result.details,
                },
                "native_fp64": native_section,
                "comparison": comparison,
                "nvidia_smi_after_case": _nvidia_smi_snapshot(),
            }
        )
        print(
            f"{case.id:38s} route={actual_arithmetic:14s} "
            + (
                f"speedup={comparison['fp64_over_double_single_speedup']:.3f}x "
                f"inside={comparison['inside_mismatch_fraction']:.4%}"
                if comparison
                else "routing-only"
            )
        )

    report = {
        "schema_version": 1,
        "purpose": "production CUDA double-single multi-target validation matrix",
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
            "routing_gate": "all cases must use their expected arithmetic route",
            "accuracy": (
                "inside/value differences are measured against explicit native FP64; "
                "they are divergence metrics, not arbitrary-precision proof"
            ),
            "performance": (
                "case measurements alternate AUTO DS and explicit FP64 order after "
                "separate duration-based warm-up"
            ),
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate production CUDA double-single routing across representative cases"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("double-single-validation-matrix.json"),
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
    print(f"Wrote {args.output}")
    return 0 if summary["routing_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
