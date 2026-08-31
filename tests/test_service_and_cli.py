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


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_export_frames_resumes(tmp_path: Path):
    plan_path = (
        REPO_ROOT
        / "examples"
        / "flight_plans"
        / "twin-spiral-nocturne.fractal-flight.json"
    )
    base = [
        sys.executable,
        "-m",
        "fractal_flight_studio.cli",
        "export-frames",
        "--plan",
        str(plan_path),
        "--backend",
        "cpu",
        "--width",
        "16",
        "--height",
        "12",
        "--iterations",
        "24",
        "--fps",
        "2",
    ]
    first = subprocess.run(
        [*base, "--output-dir", str(tmp_path), "--stop", "3"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "3 rendered, 0 skipped" in first.stdout
    second = subprocess.run(
        [*base, "--output-dir", str(tmp_path), "--stop", "5"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "2 rendered, 3 skipped" in second.stdout
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [f"frame_{i:05d}.png" for i in range(5)]
    with Image.open(tmp_path / "frame_00000.png") as image:
        assert image.size == (16, 12)
        assert image.mode == "RGB"
