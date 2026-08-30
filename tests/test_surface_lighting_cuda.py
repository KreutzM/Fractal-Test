from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_cuda_surface_lighting_matches_reference_in_simulator() -> None:
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np

from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.palettes import tone_mapped_colorize
from fractal_flight_studio.renderers import CudaRenderer
from fractal_flight_studio.surface_lighting import SurfaceLightingSettings, apply_surface_lighting

settings = SurfaceLightingSettings(
    enabled=True,
    strength=2.4,
    azimuth_degrees=210.0,
    elevation_degrees=52.0,
    ambient=0.3,
    diffuse=0.7,
)
requests = (
    RenderRequest(width=48, height=32, max_iterations=80),
    RenderRequest(
        width=40,
        height=28,
        max_iterations=100,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=256,
        center_x_text="-0.743643887037158704752191506114774",
        center_y_text="0.131825904205311970493132056385139",
        view_width_text="1e-20",
        viewport=Viewport(-0.7436438870371587, 0.13182590420531198, 1e-20),
    ),
)
renderer = CudaRenderer()
for request in requests:
    rendered = renderer.render(request)
    colored, _state, _details = tone_mapped_colorize(
        rendered.values,
        rendered.inside,
        palette="inferno",
        tone_mapping="linear",
    )
    expected = apply_surface_lighting(
        rendered.values, rendered.inside, colored, settings
    )
    actual = renderer.render_frame(
        request,
        palette="inferno",
        tone_mapping="linear",
        surface_lighting=settings,
    )
    assert np.array_equal(actual.rgb, expected)
    assert actual.details["optimized_frame_path"] is True
    assert actual.details["surface_lighting_enabled"] is True
    assert actual.details["surface_lighting_seconds"] >= 0.0
    assert actual.details["surface_lighting_strength"] == settings.strength
    assert actual.details["surface_lighting_timing_scope"] == "host kernel launch"
    assert actual.details["surface_lighting_height_source"] == "tone-mapped"
    assert actual.details["surface_lighting_slope_scale"] == 4.0
    assert actual.details["surface_lighting_flat_neutral"] is True
    assert actual.details["transfer"] == "single RGB readback"

request = requests[0]
baseline = renderer.render_frame(request, tone_mapping="linear")
disabled = renderer.render_frame(
    request,
    tone_mapping="linear",
    surface_lighting=SurfaceLightingSettings(enabled=False),
)
assert np.array_equal(disabled.rgb, baseline.rgb)
assert disabled.details["surface_lighting_enabled"] is False
assert disabled.details["surface_lighting_seconds"] == 0.0

try:
    renderer.render_frame(
        request,
        tone_mapping="linear",
        surface_lighting=True,
    )
except ValueError as exc:
    assert "SurfaceLightingSettings" in str(exc)
else:
    raise AssertionError("invalid surface lighting argument was accepted")

print("CUDA surface lighting simulator validation passed")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert "passed" in completed.stdout


def test_cuda_surface_lighting_preserves_locked_auto_tone_path_in_simulator() -> None:
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np

from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.renderers import CudaRenderer
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.surface_lighting import SurfaceLightingSettings

request = RenderRequest(width=56, height=40, max_iterations=120)
settings = SurfaceLightingSettings(
    enabled=True,
    strength=1.8,
    azimuth_degrees=325.0,
    elevation_degrees=48.0,
    ambient=0.32,
    diffuse=0.68,
)
key = ("surface-lighting-auto",)
cpu_analysis = CpuRenderer().render_frame(
    request,
    tone_mapping="auto",
    tone_scene_key=key,
    tone_smoothing=1.0,
)
locked_state = cpu_analysis.details["tone_state"]
cpu = CpuRenderer().render_frame(
    request,
    tone_mapping="auto",
    tone_state=locked_state,
    tone_scene_key=key,
    tone_state_locked=True,
    surface_lighting=settings,
)
cuda = CudaRenderer().render_frame(
    request,
    tone_mapping="auto",
    tone_state=locked_state,
    tone_scene_key=key,
    tone_state_locked=True,
    surface_lighting=settings,
)
assert np.array_equal(cuda.rgb, cpu.rgb)
assert cuda.details["tone_state_locked"] is True
assert cuda.details["tone_sample_count"] == 0
assert cuda.details["optimized_frame_path"] is True
assert cuda.details["surface_lighting_enabled"] is True
assert cuda.details["transfer"] == "single RGB readback"
print("CUDA locked auto-tone surface lighting passed")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert "passed" in completed.stdout


def test_physical_validation_cases_match_router_expectations() -> None:
    from scripts.check_cuda_surface_lighting import _auto_relief_request, _cases
    from fractal_flight_studio.deep_zoom import should_use_perturbation

    planned = {
        case.id: (
            "perturbation" if should_use_perturbation(case.request) else "direct"
        )
        for case in _cases(320, 180)
    }

    assert planned == {
        "direct-fp32": "direct",
        "direct-double-single": "direct",
        "deep-double-single-perturbation": "perturbation",
    }
    assert should_use_perturbation(_auto_relief_request(320, 180)) is False
