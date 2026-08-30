from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from fractal_flight_studio.models import (
    FractalKind,
    Precision,
    RenderMode,
    RenderRequest,
    Viewport,
)
from fractal_flight_studio.renderers.cuda_double_single import launch_geometry
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer
from fractal_flight_studio.research.double_single import coordinate_components


def test_production_launch_geometry_matches_research_candidate() -> None:
    request = RenderRequest(
        width=1280,
        height=720,
        reference_bits=256,
        center_x_text="-0.7436438479745",
        center_y_text="0.1318259432675",
        view_width_text="1e-10",
        viewport=Viewport(-0.7436438479745, 0.1318259432675, 1e-10),
    )
    actual = launch_geometry(request)
    x0, y0, dx, dy = coordinate_components(
        request.center_x_text,
        request.center_y_text,
        request.view_width_text,
        request.width,
        request.height,
        precision_bits=request.reference_bits,
    )
    expected = tuple(
        component
        for value in (x0, y0, dx, dy)
        for component in (value.hi, value.lo)
    )
    assert actual == expected


def test_double_single_direct_selection_is_conservative() -> None:
    safe = RenderRequest(
        width=800,
        height=600,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.AUTO,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-10",
        viewport=Viewport(-0.5, 0.0, 1e-10),
    )
    assert CudaRenderer._can_use_double_single(safe) is True
    assert CudaRenderer._can_use_double_single(
        RenderRequest(
            width=safe.width,
            height=safe.height,
            precision=safe.precision,
            render_mode=RenderMode.DIRECT,
            center_x_text=safe.center_x_text,
            center_y_text=safe.center_y_text,
            view_width_text=safe.view_width_text,
            viewport=safe.viewport,
        )
    ) is False
    assert CudaRenderer._can_use_double_single(
        RenderRequest(
            width=safe.width,
            height=safe.height,
            precision=Precision.FLOAT32,
            render_mode=safe.render_mode,
            center_x_text=safe.center_x_text,
            center_y_text=safe.center_y_text,
            view_width_text=safe.view_width_text,
            viewport=safe.viewport,
        )
    ) is False
    assert CudaRenderer._can_use_double_single(
        RenderRequest(
            width=safe.width,
            height=safe.height,
            fractal=FractalKind.JULIA,
            precision=safe.precision,
            render_mode=safe.render_mode,
            center_x_text=safe.center_x_text,
            center_y_text=safe.center_y_text,
            view_width_text=safe.view_width_text,
            viewport=safe.viewport,
        )
    ) is False

    exhausted = RenderRequest(
        width=800,
        height=600,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.AUTO,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-13",
        viewport=Viewport(-0.5, 0.0, 1e-13),
    )
    assert CudaRenderer._can_use_double_single(exhausted) is False


def test_cuda_auto_uses_production_double_single_in_simulator() -> None:
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.auto import AdaptivePrecisionRenderer
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer

request = RenderRequest(
    width=20, height=14, max_iterations=60, escape_radius=2.0,
    precision=Precision.FLOAT32, render_mode=RenderMode.AUTO,
    center_x_text="-0.5", center_y_text="0.0", view_width_text="1e-6",
    viewport=Viewport(-0.5, 0.0, 1e-6),
)
renderer = AdaptivePrecisionRenderer(CudaRenderer())
actual = renderer.render(request)
reference = CpuRenderer().render(
    RenderRequest(
        width=request.width, height=request.height,
        max_iterations=request.max_iterations, escape_radius=request.escape_radius,
        precision=Precision.FLOAT64, render_mode=RenderMode.DIRECT,
        center_x_text=request.center_x_text, center_y_text=request.center_y_text,
        view_width_text=request.view_width_text, viewport=request.viewport,
    )
)
assert actual.details["requested_precision"] == "float32"
assert actual.details["precision"] == "float64"
assert actual.details["arithmetic"] == "double-single"
assert actual.details["precision_promoted"] is True
assert actual.details["double_single_enabled"] is True
assert actual.details["render_mode"] == "direct"
assert np.array_equal(actual.inside, reference.inside)
assert np.allclose(actual.values, reference.values, atol=5e-5)
frame = renderer.render_frame(request, tone_mapping="linear")
assert frame.rgb.shape == (request.height, request.width, 3)
assert frame.details["precision"] == "float64"
assert frame.details["arithmetic"] == "double-single"
assert frame.details["double_single_enabled"] is True
assert frame.details["optimized_frame_path"] is True
print("production double-single CUDA path passed")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert "passed" in completed.stdout
