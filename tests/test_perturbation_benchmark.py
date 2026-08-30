from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_flight_studio.deep_zoom_targets import deep_zoom_target
from fractal_flight_studio.models import Precision, RenderMode


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_perturbation.py"
    spec = importlib.util.spec_from_file_location("fractal_perturbation_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Clock:
    def __init__(self, step: float = 0.01) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class _FakeRenderer:
    def __init__(self, name: str = "cpu-numba") -> None:
        self.name = name
        self.target_calls = 0
        self.requests = []

    def is_available(self) -> bool:
        return True

    def render_frame(self, request, **kwargs):
        self.requests.append(request)
        is_target = request.center_x_text != "0.25"
        if is_target:
            reused = self.target_calls > 0
            self.target_calls += 1
        else:
            reused = False
        return SimpleNamespace(
            rgb=np.zeros((request.height, request.width, 3), dtype=np.uint8),
            backend=self.name,
            elapsed_seconds=0.02 if self.name == "cpu-numba" else 0.01,
            details={
                "render_mode": "perturbation",
                "precision": "float64",
                "reference_reused": reused,
                "reference_bits": request.reference_bits,
                "reference_upload_seconds": 0.001,
            },
        )


def test_request_for_target_forces_exact_fp64_perturbation() -> None:
    benchmark = _load_module()
    target = deep_zoom_target("seahorse-satellite")
    config = benchmark.PerturbationBenchmarkConfig(width=640, height=360, repeats=2)

    request = benchmark._request_for_target(target, config)

    assert request.render_mode is RenderMode.PERTURBATION
    assert request.precision is Precision.FLOAT64
    assert request.center_x_text == target.center_x_text
    assert request.center_y_text == target.center_y_text
    assert request.view_width_text == target.view_width_text
    assert request.max_iterations == target.recommended_iterations
    assert request.reference_bits == target.reference_bits


def test_backend_benchmark_separates_cold_and_reused_frames() -> None:
    benchmark = _load_module()
    target = deep_zoom_target("seahorse-satellite")
    config = benchmark.PerturbationBenchmarkConfig(width=64, height=36, repeats=3)
    request = benchmark._request_for_target(target, config)
    renderer = _FakeRenderer()

    record = benchmark._benchmark_backend(
        renderer,
        request,
        config.repeats,
        clock=_Clock(),
    )

    assert len(renderer.requests) == 5
    assert renderer.requests[0].center_x_text == "0.25"
    assert record["cold_reference_reused"] is False
    assert record["warm_reference_reused"] is True
    assert record["cold_wall_seconds"] == pytest.approx(0.01)
    assert record["warm_wall_seconds_median"] == pytest.approx(0.01)
    assert record["warm_details"]["reference_reused"] is True


def test_comparison_reports_cpu_or_cuda_winner() -> None:
    benchmark = _load_module()
    records = [
        {
            "backend": "cpu-numba",
            "cold_wall_seconds": 2.0,
            "warm_wall_seconds_median": 1.0,
        },
        {
            "backend": "cuda-numba",
            "cold_wall_seconds": 1.0,
            "warm_wall_seconds_median": 2.0,
        },
    ]

    comparison = benchmark._comparison(records)

    assert comparison == {
        "rgb_outputs_match": True,
        "warm_winner": "cpu-numba",
        "warm_cuda_speedup_over_cpu": 0.5,
        "cold_winner": "cuda-numba",
        "cold_cuda_speedup_over_cpu": 2.0,
    }


def test_main_writes_json_report(tmp_path, monkeypatch) -> None:
    benchmark = _load_module()
    output = tmp_path / "perturbation.json"
    renderer = _FakeRenderer()

    monkeypatch.setattr(benchmark, "_backend_names", lambda _preference: ["cpu"])
    monkeypatch.setattr(benchmark, "_create_renderer", lambda _name: renderer)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_perturbation.py",
            "--backend",
            "cpu",
            "--width",
            "64",
            "--height",
            "36",
            "--repeats",
            "2",
            "--output",
            str(output),
        ],
    )

    assert benchmark.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["target"]["id"] == "seahorse-satellite"
    assert report["config"]["size"] == [64, 36]
    assert report["results"][0]["backend"] == "cpu-numba"
    assert report["results"][0]["warm_reference_reused"] is True
    assert report["comparison"] is None
