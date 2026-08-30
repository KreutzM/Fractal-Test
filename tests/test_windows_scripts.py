from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_launcher_installs_cuda_extra_for_nvidia():
    text = (ROOT / "scripts" / "run_windows.ps1").read_text()
    assert "nvidia-smi.exe" in text
    assert "${ProjectRoot}[cuda12]" in text
    assert "fractal_flight_studio.doctor" in text


def test_manual_cuda_enable_script_uses_project_venv():
    text = (ROOT / "scripts" / "enable_cuda.ps1").read_text()
    assert '.venv\\Scripts\\python.exe' in text
    assert "${ProjectRoot}[cuda12]" in text
