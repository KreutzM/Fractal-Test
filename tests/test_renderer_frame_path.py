from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from fractal_flight_studio.renderers import select_renderer


def test_renderer_instances_are_reused():
    assert select_renderer("cpu") is select_renderer("cpu")


def test_cuda_frame_path_in_simulator():
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.renderers import CudaRenderer

request = RenderRequest(width=48, height=32, max_iterations=50)
cpu = CpuRenderer().render_frame(request, tone_mapping="linear")
renderer = CudaRenderer()
first = renderer.render_frame(request, tone_mapping="linear")
second = renderer.render_frame(request, tone_mapping="linear")
assert first.rgb.shape == (32, 48, 3)
assert first.rgb.dtype == np.uint8
assert np.array_equal(first.rgb, cpu.rgb)
assert first.details["optimized_frame_path"] is True
assert first.details["persistent_buffers"] is True
assert second.details["allocation_seconds"] == 0.0
assert second.details["palette_upload_seconds"] == 0.0
print("optimized CUDA frame path passed")
'''
    completed = subprocess.run([sys.executable, "-c", code], env=env, check=True, text=True, capture_output=True)
    assert "passed" in completed.stdout


def test_cuda_perturbation_matches_cpu_in_simulator():
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.renderers import CudaRenderer

request = RenderRequest(
    width=40, height=28, max_iterations=80, precision=Precision.FLOAT64,
    render_mode=RenderMode.PERTURBATION, reference_bits=256,
    center_x_text="-0.743643887037158704752191506114774",
    center_y_text="0.131825904205311970493132056385139",
    view_width_text="1e-20",
    viewport=Viewport(-0.7436438870371587, 0.13182590420531198, 1e-20),
)
cpu = CpuRenderer().render_frame(request, tone_mapping="linear")
renderer = CudaRenderer()
cuda = renderer.render_frame(request, tone_mapping="linear")
cuda_second = renderer.render_frame(request, tone_mapping="linear")
assert np.array_equal(cuda.rgb, cpu.rgb)
assert np.array_equal(cuda_second.rgb, cpu.rgb)
assert cuda.details["render_mode"] == "perturbation"
assert cuda.details["reference_bits"] == 256
assert cuda.details["rebasing_enabled"] is True
assert cuda.details["glitch_detection_enabled"] is True
assert cuda_second.details["reference_reused"] is True
assert cuda_second.details["reference_upload_seconds"] == 0.0
print("perturbation CUDA path passed")
'''
    completed = subprocess.run([sys.executable, "-c", code], env=env, check=True, text=True, capture_output=True)
    assert "passed" in completed.stdout


def test_cuda_perturbation_reports_rebases_in_simulator():
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.renderers import CudaRenderer

request = RenderRequest(
    width=96, height=72, max_iterations=200, precision=Precision.FLOAT64,
    render_mode=RenderMode.PERTURBATION, center_x_text="-0.5", center_y_text="0.0",
    view_width_text="3.5", viewport=Viewport(-0.5, 0.0, 3.5),
)
cpu = CpuRenderer().render(request)
cuda = CudaRenderer().render(request)
assert np.array_equal(cuda.inside, cpu.inside)
assert np.allclose(cuda.values, cpu.values, atol=5e-5)
assert cuda.details["rebasing_enabled"] is True
assert cuda.details["glitch_detection_enabled"] is True
assert cuda.details["rebase_pixels"] == cpu.details["rebase_pixels"]
assert cuda.details["glitch_pixels"] == cpu.details["glitch_pixels"]
assert cuda.details["rebase_pixels"] > 0
print("perturbation CUDA rebasing path passed")
'''
    completed = subprocess.run([sys.executable, "-c", code], env=env, check=True, text=True, capture_output=True)
    assert "passed" in completed.stdout


def test_cuda_auto_tone_mapping_stays_on_optimized_gpu_path():
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.renderers import CudaRenderer

request = RenderRequest(width=64, height=48, max_iterations=120)
key = ("tone-demo",)
cpu = CpuRenderer().render_frame(request, tone_mapping="auto", tone_scene_key=key, tone_smoothing=1.0)
renderer = CudaRenderer()
first = renderer.render_frame(request, tone_mapping="auto", tone_scene_key=key, tone_smoothing=1.0)
second = renderer.render_frame(
    request, tone_mapping="auto", tone_state=first.details["tone_state"],
    tone_scene_key=key, tone_smoothing=0.08,
)
locked_state = first.details["tone_state"]
cpu_locked = CpuRenderer().render_frame(
    request, tone_mapping="auto", tone_state=locked_state,
    tone_scene_key=key, tone_state_locked=True,
)
cuda_locked = renderer.render_frame(
    request, tone_mapping="auto", tone_state=locked_state,
    tone_scene_key=key, tone_state_locked=True,
)
assert np.array_equal(first.rgb, cpu.rgb)
assert np.array_equal(cuda_locked.rgb, cpu_locked.rgb)
assert first.details["optimized_frame_path"] is True
assert first.details["tone_mapping"] == "auto"
assert first.details["tone_sample_count"] > 0
assert first.details["transfer"] == "stratified tone sample + single RGB readback"
assert second.details["tone_scene_reset"] is False
assert second.details["allocation_seconds"] == 0.0
assert second.details["palette_upload_seconds"] == 0.0
assert cpu_locked.details["tone_state_locked"] is True
assert cuda_locked.details["tone_state_locked"] is True
assert cuda_locked.details["tone_sample_count"] == 0
assert cuda_locked.details["transfer"] == "single RGB readback"
print("optimized CUDA auto tone path passed")
'''
    completed = subprocess.run([sys.executable, "-c", code], env=env, check=True, text=True, capture_output=True)
    assert "passed" in completed.stdout
