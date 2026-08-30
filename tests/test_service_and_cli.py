from pathlib import Path
import subprocess
import sys

from PIL import Image

from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.service import save_png
from fractal_flight_studio.surface_lighting import SurfaceLightingSettings


def test_save_png(tmp_path: Path):
    output = tmp_path / "test.png"
    result = save_png(RenderRequest(width=48, height=32, max_iterations=30), output, backend="cpu")
    assert result.backend == "cpu-numba"
    with Image.open(output) as image:
        assert image.size == (48, 32)
        assert image.mode == "RGB"


def test_save_png_with_surface_lighting(tmp_path: Path):
    output = tmp_path / "lit.png"
    result = save_png(
        RenderRequest(width=48, height=32, max_iterations=30),
        output,
        backend="cpu",
        tone_mapping="linear",
        surface_lighting=SurfaceLightingSettings(enabled=True, strength=2.0),
    )
    assert result.details["surface_lighting_enabled"] is True
    assert result.details["surface_lighting_strength"] == 2.0
    with Image.open(output) as image:
        assert image.size == (48, 32)
        assert image.mode == "RGB"


def test_cli_render(tmp_path: Path):
    output = tmp_path / "cli.png"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fractal_flight_studio.cli",
            "render",
            "--backend",
            "cpu",
            "--width",
            "40",
            "--height",
            "30",
            "--iterations",
            "25",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert output.exists()
    assert "saved" in completed.stdout
