from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import json
import platform
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from fractal_flight_studio.models import FractalKind, RenderRequest, Viewport
from fractal_flight_studio.renderers import available_renderers, select_renderer


def _json_compatible(value: Any) -> Any:
    """Convert benchmark metadata to JSON values without hiding unknown types."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(
        f"benchmark metadata contains unsupported value {type(value).__name__}"
    )


def _scenarios() -> list[tuple[str, RenderRequest]]:
    return [
        (
            "mandelbrot-overview-1080p",
            RenderRequest(width=1920, height=1080, max_iterations=400),
        ),
        (
            "mandelbrot-boundary-1080p",
            RenderRequest(
                width=1920,
                height=1080,
                viewport=Viewport(-0.743643887037151, 0.131825904205330, 0.003),
                max_iterations=1000,
            ),
        ),
        (
            "julia-720p",
            RenderRequest(
                width=1280,
                height=720,
                fractal=FractalKind.JULIA,
                max_iterations=600,
            ),
        ),
    ]


def _backend_names(preference: str) -> list[str]:
    if preference != "all":
        return [preference]
    available = available_renderers()
    names = ["cpu"]
    if "cuda-numba" in available:
        names.append("cuda")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Steady-state benchmark of the actual display-frame render path"
    )
    parser.add_argument("--backend", default="all", choices=["all", "auto", "cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    records: list[dict[str, object]] = []
    for backend_name in _backend_names(args.backend):
        renderer = select_renderer(backend_name)
        # JIT and context warm-up; scenario-sized warm-up below also allocates
        # the persistent output buffers before the timed steady-state runs.
        renderer.render_frame(RenderRequest(width=32, height=32, max_iterations=16))

        for scenario_name, request in _scenarios():
            first_started = time.perf_counter()
            first = renderer.render_frame(request)
            first_wall = time.perf_counter() - first_started

            frame_times: list[float] = []
            wall_times: list[float] = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = renderer.render_frame(request)
                wall = time.perf_counter() - started
                frame_times.append(result.elapsed_seconds)
                wall_times.append(wall)

            frame_median = statistics.median(frame_times)
            wall_median = statistics.median(wall_times)
            megapixels = request.width * request.height / 1_000_000
            record = {
                "scenario": scenario_name,
                "fractal": request.fractal.value,
                "size": [request.width, request.height],
                "pixels": request.width * request.height,
                "iterations": request.max_iterations,
                "view_width": request.viewport.width,
                "backend": result.backend,
                "precision": request.precision.value,
                "first_frame_seconds": first.elapsed_seconds,
                "first_wall_seconds": first_wall,
                "steady_frame_seconds_median": frame_median,
                "steady_wall_seconds_median": wall_median,
                "steady_frame_fps": 1.0 / frame_median,
                "steady_mpix_per_second": megapixels / frame_median,
                "details": _json_compatible(result.details),
            }
            records.append(record)
            print(
                f"{result.backend:11s} {scenario_name:30s} "
                f"{frame_median * 1000:8.2f} ms  "
                f"{1.0 / frame_median:7.2f} FPS  "
                f"{megapixels / frame_median:8.2f} MPix/s"
            )

    report = {
        "system": platform.platform(),
        "python": platform.python_version(),
        "available_backends": list(available_renderers()),
        "repeats": args.repeats,
        "measurement": "render_frame: fractal + colorization + required host readback; excludes Tk display",
        "results": records,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
