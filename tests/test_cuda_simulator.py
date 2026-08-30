import os
from pathlib import Path
import subprocess
import sys


def test_cuda_kernel_matches_cpu_in_simulator():
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "fractal_flight_studio.cuda_smoke"],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "passed" in completed.stdout
