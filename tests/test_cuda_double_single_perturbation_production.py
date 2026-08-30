from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from fractal_flight_studio.models import (
    FractalKind,
    Precision,
    RenderMode,
    RenderRequest,
    Viewport,
)
from fractal_flight_studio.renderers.cuda_double_single_perturbation import (
    DoubleSinglePerturbationUnavailable,
    prepare_perturbation_launch,
    split_reference_orbit,
)
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer


def _perturb(real, imag, **overrides):
    values = {
        "orbit_real": np.asarray(real, dtype=np.float64),
        "orbit_imag": np.asarray(imag, dtype=np.float64),
        "x0_rel": -2.5e-13,
        "y0_rel": -1.5e-13,
        "dx": 3.0e-16,
        "dy": 3.0e-16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_production_perturbation_split_preserves_safe_inputs() -> None:
    perturb = _perturb([0.0, -0.75, -0.2], [0.0, 0.1, 0.3])
    launch = prepare_perturbation_launch(perturb)
    assert launch.orbit.transfer_bytes == 4 * 3 * np.dtype(np.float32).itemsize
    reconstructed = (
        launch.orbit.real_hi.astype(np.float64)
        + launch.orbit.real_lo.astype(np.float64)
    )
    assert np.allclose(reconstructed, perturb.orbit_real, rtol=0.0, atol=2e-15)
    assert float(launch.dx.hi) + float(launch.dx.lo) != 0.0


def test_production_perturbation_rejects_reference_magnitude_below_fp32_floor() -> None:
    with pytest.raises(DoubleSinglePerturbationUnavailable, match="reference-magnitude-range"):
        split_reference_orbit(
            np.array([0.0, 1e-30], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        )


def test_production_perturbation_rejects_relative_coordinate_underflow() -> None:
    perturb = _perturb([0.0, -0.75], [0.0, 0.1], dx=1e-300)
    with pytest.raises(DoubleSinglePerturbationUnavailable, match="dx-underflow"):
        prepare_perturbation_launch(perturb)


def test_double_single_perturbation_selection_is_auto_only() -> None:
    request = RenderRequest(
        width=800,
        height=600,
        fractal=FractalKind.MANDELBROT,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.AUTO,
        viewport=Viewport(-0.5, 0.0, 1e-15),
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-15",
    )
    perturb = object()
    assert CudaRenderer._can_use_double_single_perturbation(request, perturb) is True
    explicit = RenderRequest(
        width=request.width,
        height=request.height,
        fractal=request.fractal,
        precision=request.precision,
        render_mode=RenderMode.PERTURBATION,
        viewport=request.viewport,
        center_x_text=request.center_x_text,
        center_y_text=request.center_y_text,
        view_width_text=request.view_width_text,
    )
    assert CudaRenderer._can_use_double_single_perturbation(explicit, perturb) is False


def test_cuda_auto_uses_double_single_perturbation_in_simulator() -> None:
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = r'''
import numpy as np
from numba import cuda
if not hasattr(cuda, "fma"):
    cuda.fma = lambda a, b, c: a * b + c
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer

common = dict(
    width=8, height=6, max_iterations=40, escape_radius=2.0,
    precision=Precision.FLOAT64, reference_bits=192,
    center_x_text="-0.743643887037151",
    center_y_text="0.13182590420533",
    view_width_text="1e-15",
    viewport=Viewport(-0.743643887037151, 0.13182590420533, 1e-15),
)
auto = CudaRenderer().render(RenderRequest(render_mode=RenderMode.AUTO, **common))
native = CudaRenderer().render(RenderRequest(render_mode=RenderMode.PERTURBATION, **common))
assert auto.details["render_mode"] == "perturbation"
assert auto.details["arithmetic"] == "double-single"
assert auto.details["double_single_enabled"] is True
assert auto.details["double_single_mode"] == "perturbation"
assert auto.details["double_single_perturbation_enabled"] is True
assert auto.details["double_single_fallback_reason"] == ""
assert native.details["arithmetic"] == "float64"
assert native.details["double_single_enabled"] is False
assert native.details["double_single_fallback_reason"] == "routing-policy"
assert np.array_equal(auto.inside, native.inside)
assert np.array_equal(auto.values, native.values)
print("production double-single perturbation CUDA path passed")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert "passed" in completed.stdout
