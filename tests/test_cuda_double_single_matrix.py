from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_cuda_double_single_matrix.py"
SPEC = importlib.util.spec_from_file_location("check_cuda_double_single_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_default_matrix_covers_required_categories_and_transition() -> None:
    categories = {case.category for case in MODULE.DEFAULT_CASES}
    assert {"exterior", "interior", "boundary", "filament", "satellite", "transition"} <= categories
    assert any(case.expected_arithmetic == "double-single" for case in MODULE.DEFAULT_CASES)
    handoff = next(case for case in MODULE.DEFAULT_CASES if case.id == "perturbation-handoff")
    assert handoff.expected_arithmetic == "float64"
    assert handoff.expected_render_mode == "perturbation"


def test_requests_preserve_exact_camera_text_and_route_intent() -> None:
    case = MODULE.DEFAULT_CASES[3]
    auto = MODULE._request(case, 1280, 720, native_fp64=False)
    native = MODULE._request(case, 1280, 720, native_fp64=True)
    assert auto.center_x_text == case.center_x_text
    assert auto.center_y_text == case.center_y_text
    assert auto.view_width_text == case.view_width_text
    assert auto.precision.value == "float32"
    assert auto.render_mode.value == "auto"
    assert native.precision.value == "float64"
    assert native.render_mode.value == "direct"


def test_default_cases_match_current_production_routing_boundaries() -> None:
    from fractal_flight_studio.deep_zoom import should_use_perturbation
    from fractal_flight_studio.renderers.auto import AdaptivePrecisionRenderer
    from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer

    for case in MODULE.DEFAULT_CASES:
        request = MODULE._request(case, 1280, 720, native_fp64=False)
        effective, _ = AdaptivePrecisionRenderer._effective_request(request)
        if case.expected_arithmetic == "double-single":
            assert not should_use_perturbation(effective), case.id
            assert CudaRenderer._can_use_double_single(effective), case.id
        else:
            assert should_use_perturbation(effective), case.id
            assert not CudaRenderer._can_use_double_single(effective), case.id


def test_timing_summary_reports_median_range_and_variation() -> None:
    summary = MODULE._timing_summary([1.0, 2.0, 3.0])
    assert summary["median_seconds"] == 2.0
    assert summary["minimum_seconds"] == 1.0
    assert summary["maximum_seconds"] == 3.0
    assert summary["coefficient_of_variation"] == pytest.approx(0.408248290463863)


def test_summary_reports_route_failures_and_worst_case_metrics() -> None:
    records = [
        {
            "id": "a",
            "route_matches": True,
            "comparison": {
                "fp64_over_double_single_speedup": 4.0,
                "inside_mismatch_fraction": 0.001,
                "mean_absolute_value_delta": 0.002,
                "maximum_absolute_value_delta": 0.8,
            },
        },
        {
            "id": "b",
            "route_matches": False,
            "comparison": {
                "fp64_over_double_single_speedup": 2.5,
                "inside_mismatch_fraction": 0.003,
                "mean_absolute_value_delta": 0.004,
                "maximum_absolute_value_delta": 0.9,
            },
        },
        {"id": "handoff", "route_matches": True, "comparison": None},
    ]
    summary = MODULE.summarize_cases(records)
    assert summary["route_failures"] == ["b"]
    assert summary["routing_gate_passed"] is False
    assert summary["comparison_count"] == 2
    assert summary["minimum_fp64_over_double_single_speedup"] == 2.5
    assert summary["maximum_inside_mismatch_fraction"] == 0.003
    assert summary["mean_case_absolute_value_delta"] == pytest.approx(0.003)
    assert summary["maximum_absolute_value_delta"] == 0.9
