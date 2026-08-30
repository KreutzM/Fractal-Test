from __future__ import annotations

import subprocess

from fractal_flight_studio.gpu_info import CudaStatus, _query_nvidia_smi


def test_cuda_status_summary_for_available_device():
    status = CudaStatus(
        available=True,
        device_name="NVIDIA GeForce RTX 3060",
        compute_capability="8.6",
        driver_version="999.1",
        numba_version="0.65",
        numba_cuda_version="0.22",
        nvidia_smi_found=True,
    )
    assert "RTX 3060" in status.summary
    assert "CC 8.6" in status.summary
    assert "numba-cuda: 0.22" in status.report()


def test_cuda_status_summary_contains_failure_reason():
    status = CudaStatus(available=False, reason="driver missing")
    assert status.summary == "CUDA nicht verfügbar: driver missing"


def test_query_nvidia_smi_parses_first_gpu():
    def fake_runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="NVIDIA GeForce RTX 3060, 555.12, 8.6\n",
            stderr="",
        )

    assert _query_nvidia_smi("nvidia-smi.exe", runner=fake_runner) == (
        "NVIDIA GeForce RTX 3060",
        "555.12",
        "8.6",
    )
