from __future__ import annotations

from pathlib import Path
import sys

import pytest

from fractal_flight_studio.deep_zoom import PerturbationReferenceCache, should_use_perturbation
from fractal_flight_studio.renderers.cuda_double_single_perturbation import (
    DoubleSinglePerturbationUnavailable,
    prepare_perturbation_launch,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_cuda_double_single_perturbation_matrix as matrix  # noqa: E402


def test_matrix_cases_cover_ds_reuse_and_guarded_fallback() -> None:
    ids = [case.id for case in matrix.DEFAULT_CASES]
    assert len(ids) == len(set(ids)) == 7
    assert sum(case.expected_arithmetic == "double-single" for case in matrix.DEFAULT_CASES) == 6
    fallback = matrix.DEFAULT_CASES[-1]
    assert fallback.expected_arithmetic == "float64"
    assert fallback.expected_fallback_reason == "reference-magnitude-range"
    assert matrix.DEFAULT_CASES[1].expected_reference_reused is True


def test_matrix_ds_cases_are_perturbation_safe_and_reuse_reference() -> None:
    cache = PerturbationReferenceCache()
    for index, case in enumerate(matrix.DEFAULT_CASES[:-1]):
        request = matrix._request(case, 1280, 720, native_fp64=False)
        assert should_use_perturbation(request)
        perturb = cache.prepare(request)
        prepare_perturbation_launch(perturb)
        if index == 0:
            assert perturb.reference_reused is False
        elif index == 1:
            assert perturb.reference_reused is True


def test_matrix_fallback_case_is_rejected_by_ds_guard() -> None:
    case = matrix.DEFAULT_CASES[-1]
    request = matrix._request(case, 1280, 720, native_fp64=False)
    assert should_use_perturbation(request)
    perturb = PerturbationReferenceCache().prepare(request)
    with pytest.raises(DoubleSinglePerturbationUnavailable, match="reference-magnitude-range"):
        prepare_perturbation_launch(perturb)


def test_matrix_summary_separates_ds_and_fallback_cases() -> None:
    records = [
        {
            "id": "ds",
            "expected_arithmetic": "double-single",
            "route_matches": True,
            "comparison": {
                "fp64_over_auto_speedup": 2.5,
                "inside_mismatch_fraction": 0.001,
                "glitch_flag_mismatch_pixels": 2,
                "rebase_flag_mismatch_pixels": 3,
                "maximum_absolute_value_delta": 0.5,
            },
        },
        {
            "id": "fallback",
            "expected_arithmetic": "float64",
            "route_matches": True,
            "comparison": {
                "fp64_over_auto_speedup": 1.0,
                "inside_mismatch_fraction": 0.0,
                "glitch_flag_mismatch_pixels": 0,
                "rebase_flag_mismatch_pixels": 0,
                "maximum_absolute_value_delta": 0.0,
            },
        },
    ]
    summary = matrix.summarize_cases(records)
    assert summary["routing_gate_passed"] is True
    assert summary["double_single_case_count"] == 1
    assert summary["fallback_case_count"] == 1
    assert summary["minimum_fp64_over_double_single_speedup"] == 2.5
