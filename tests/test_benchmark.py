from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.tonemapping import ToneMapState


def _load_benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("fractal_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_compatible_preserves_tone_state_and_numpy_scalars() -> None:
    benchmark = _load_benchmark_module()
    state = ToneMapState(
        mode="auto",
        scene_key=("mandelbrot", np.int64(256)),
        low=np.float64(0.25),
        high=12.5,
        strength=3.0,
        gamma=0.9,
    )

    converted = benchmark._json_compatible(
        {"tone_state": state, "count": np.int32(7), "flags": (True, False)}
    )

    assert converted == {
        "tone_state": {
            "mode": "auto",
            "scene_key": ["mandelbrot", 256],
            "low": 0.25,
            "high": 12.5,
            "strength": 3.0,
            "gamma": 0.9,
        },
        "count": 7,
        "flags": [True, False],
    }
    json.dumps(converted)


def test_benchmark_writes_report_with_tone_state(tmp_path, monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    output = tmp_path / "benchmark.json"
    tone_state = ToneMapState("auto", ("tiny",), 0.1, 2.0, 4.0, 0.8)

    class FakeRenderer:
        def render_frame(self, request):
            return SimpleNamespace(
                backend="fake",
                elapsed_seconds=0.01,
                details={"tone_state": tone_state, "sample_count": np.int64(4)},
            )

    monkeypatch.setattr(benchmark, "_backend_names", lambda _preference: ["cpu"])
    monkeypatch.setattr(benchmark, "available_renderers", lambda: ("fake",))
    monkeypatch.setattr(benchmark, "select_renderer", lambda _name: FakeRenderer())
    monkeypatch.setattr(
        benchmark,
        "_scenarios",
        lambda: [("tiny", RenderRequest(width=2, height=2, max_iterations=4))],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark.py", "--backend", "cpu", "--repeats", "1", "--output", str(output)],
    )

    assert benchmark.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    details = report["results"][0]["details"]
    assert details["tone_state"]["mode"] == "auto"
    assert details["tone_state"]["scene_key"] == ["tiny"]
    assert details["sample_count"] == 4
