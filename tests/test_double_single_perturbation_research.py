from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fractal_flight_studio.research.double_single_perturbation import (
    split_float64_array,
    split_reference_orbit,
    split_scalar,
)


def test_split_float64_array_preserves_about_double_single_precision() -> None:
    values = np.array([0.0, 1.0, -0.743643887037151, 1e-20, -1e20], dtype=np.float64)
    high, low = split_float64_array(values)
    reconstructed = high.astype(np.float64) + low.astype(np.float64)
    error = np.abs(reconstructed - values)
    scale = np.maximum(1.0, np.abs(values))
    assert high.dtype == np.float32
    assert low.dtype == np.float32
    assert np.all(error / scale < 2.0**-45)


def test_split_reference_orbit_uses_same_transfer_size_as_fp64() -> None:
    real = np.linspace(-2.0, 2.0, 17, dtype=np.float64)
    imag = np.linspace(1.0, -1.0, 17, dtype=np.float64)
    split = split_reference_orbit(real, imag)
    assert split.transfer_bytes == real.nbytes + imag.nbytes


def test_split_reference_orbit_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        split_reference_orbit(np.zeros(2), np.zeros(3))


def test_split_float64_array_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        split_float64_array(np.array([np.inf], dtype=np.float64))


def test_split_scalar_reconstructs_relative_coordinate() -> None:
    value = -3.90625e-16
    split = split_scalar(value)
    assert abs(split.as_float() - value) < 1e-28


def test_double_single_perturbation_kernel_runs_in_cuda_simulator() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = r'''
import numpy as np
from numba import cuda
if not hasattr(cuda, "fma"):
    cuda.fma = lambda a, b, c: a * b + c
from fractal_flight_studio.research.double_single_perturbation import (
    double_single_perturbation_kernel,
    split_reference_orbit,
    split_scalar,
)

shape = (2, 2)
values = cuda.device_array(shape, dtype=np.float32)
inside = cuda.device_array(shape, dtype=np.bool_)
glitch = cuda.device_array(shape, dtype=np.bool_)
rebase = cuda.device_array(shape, dtype=np.bool_)
real = np.zeros(4, dtype=np.float64)
imag = np.zeros(4, dtype=np.float64)
split = split_reference_orbit(real, imag)
x0 = split_scalar(-0.1)
y0 = split_scalar(-0.1)
dx = split_scalar(0.1)
dy = split_scalar(0.1)
double_single_perturbation_kernel[(1, 1), (2, 2)](
    values, inside, glitch, rebase, 2, 2,
    x0.hi, x0.lo, y0.hi, y0.lo, dx.hi, dx.lo, dy.hi, dy.lo,
    3, np.float32(4.0),
    cuda.to_device(split.real_hi), cuda.to_device(split.real_lo),
    cuda.to_device(split.imag_hi), cuda.to_device(split.imag_lo),
    3, np.float32(1e-6), np.float32(np.finfo(np.float32).tiny),
)
cuda.synchronize()
host_values = values.copy_to_host()
host_inside = inside.copy_to_host()
host_glitch = glitch.copy_to_host()
host_rebase = rebase.copy_to_host()
assert np.all(np.isfinite(host_values))
assert np.all(host_inside)
assert host_glitch.dtype == np.bool_
assert host_rebase.dtype == np.bool_
print("ok")
'''
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
