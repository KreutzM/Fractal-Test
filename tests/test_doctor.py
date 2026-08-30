from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace
from typing import cast

from fractal_flight_studio.doctor import (
    CORE_ERROR,
    CORE_OK,
    STATE_ERROR,
    STATE_OK,
    STATE_WARNING,
    TKINTER_IS_CORE,
    CheckResult,
    HealthReport,
    check_python,
    format_report,
    main,
    run_health_check,
)
from fractal_flight_studio.gpu_info import CudaStatus

ALL_PACKAGES = ("numpy", "numba", "pillow", "mpmath")

CUDA_UNAVAILABLE = CudaStatus(available=False, reason="no driver on this box")
CUDA_AVAILABLE = CudaStatus(
    available=True,
    device_name="NVIDIA GeForce RTX 3060",
    compute_capability="8.6",
    driver_version="555.12",
    numba_version="0.61",
    numba_cuda_version="0.22",
    nvidia_smi_found=True,
)


class FakeEnv:
    """Deterministic stand-in for an installation with switchable faults."""

    def __init__(
        self,
        python_version: tuple[int, int, int] = (3, 12, 4),
        packages: tuple[str, ...] = ALL_PACKAGES,
        tk_ok: bool = True,
        ffmpeg_found: str | None = "/usr/bin/ffmpeg",
        cuda: CudaStatus = CUDA_UNAVAILABLE,
        cpu_ok: bool = True,
    ) -> None:
        self.python_version = python_version
        self._packages = set(packages)
        self._tk_ok = tk_ok
        self._ffmpeg_found = ffmpeg_found
        self._cuda = cuda
        self._cpu_ok = cpu_ok

    def package_version(self, name: str) -> str | None:
        return "1.2.3" if name in self._packages else None

    def import_module(self, name: str) -> ModuleType:
        if name == "tkinter":
            if not self._tk_ok:
                raise ImportError("No module named 'tkinter'")
            return cast(ModuleType, SimpleNamespace(__version__=""))
        distribution = {"PIL": "pillow"}.get(name, name)
        if distribution not in self._packages:
            raise ImportError(f"No module named '{name}'")
        return cast(ModuleType, SimpleNamespace(__version__="1.2.3"))

    def find_executable(self, name: str) -> str | None:
        return self._ffmpeg_found if name == "ffmpeg" else None

    def inspect_cuda(self) -> CudaStatus:
        return self._cuda

    def render_cpu_test_frame(self) -> dict:
        if not self._cpu_ok:
            raise RuntimeError("numba compile failure")
        return {"backend": "cpu-numba", "width": 16, "height": 16, "elapsed_seconds": 0.1}


def _check(report: HealthReport, key: str) -> CheckResult:
    return next(c for c in report.checks if c.key == key)


def _states(report: HealthReport) -> dict[str, str]:
    return {c.key: c.state for c in report.checks}


def test_healthy_cpu_only_installation_exits_zero():
    report = run_health_check(FakeEnv(cuda=CUDA_UNAVAILABLE, ffmpeg_found=None))
    assert report.exit_code == CORE_OK
    states = _states(report)
    assert states["cuda"] == STATE_WARNING
    assert states["ffmpeg"] == STATE_WARNING
    assert states["cpu_renderer"] == STATE_OK
    assert states["python"] == STATE_OK


def test_healthy_cuda_installation_exits_zero_and_reports_details():
    report = run_health_check(FakeEnv(cuda=CUDA_AVAILABLE))
    assert report.exit_code == CORE_OK
    cuda = _check(report, "cuda")
    assert cuda.state == STATE_OK
    assert "RTX 3060" in cuda.detail
    assert cuda.data["compute_capability"] == "8.6"
    assert cuda.data["driver_version"] == "555.12"


def test_missing_ffmpeg_is_warning_not_failure():
    report = run_health_check(FakeEnv(ffmpeg_found=None))
    assert _check(report, "ffmpeg").state == STATE_WARNING
    assert report.exit_code == CORE_OK


def test_missing_tkinter_state_is_documented_choice():
    report = run_health_check(FakeEnv(tk_ok=False))
    tk = _check(report, "tkinter")
    # Issue #1: decide and document whether Tkinter is core. This desktop-first
    # repository says yes, so a missing Tkinter must be a visible core failure.
    assert TKINTER_IS_CORE is True
    assert tk.state == STATE_ERROR
    assert report.exit_code == CORE_ERROR
    assert "Desktop-GUI" in tk.detail


def test_unsupported_python_is_core_failure():
    env = FakeEnv(python_version=(3, 10, 12))
    report = run_health_check(env)
    assert _check(report, "python").state == STATE_ERROR
    assert report.exit_code == CORE_ERROR
    assert _states(report)["cpu_renderer"] == STATE_OK  # rest stays healthy

    # 3.10 is too old, 3.11/3.12/3.13 supported, 3.14 not yet.
    assert check_python(FakeEnv(python_version=(3, 11, 0))).state == STATE_OK
    assert check_python(FakeEnv(python_version=(3, 13, 9))).state == STATE_OK
    assert check_python(FakeEnv(python_version=(3, 14, 0))).state == STATE_ERROR


def test_missing_required_package_is_core_failure():
    report = run_health_check(FakeEnv(packages=("numba", "pillow", "mpmath")))
    assert _check(report, "package:numpy").state == STATE_ERROR
    assert report.exit_code == CORE_ERROR

    report = run_health_check(FakeEnv(packages=("numpy", "numba", "mpmath")))
    assert _check(report, "package:pillow").state == STATE_ERROR
    assert report.exit_code == CORE_ERROR


def test_broken_cpu_renderer_is_core_failure():
    report = run_health_check(FakeEnv(cpu_ok=False))
    assert _check(report, "cpu_renderer").state == STATE_ERROR
    assert report.exit_code == CORE_ERROR


def test_json_output_is_deterministic_and_schema_stable():
    env = FakeEnv(cuda=CUDA_AVAILABLE)
    first = run_health_check(env).to_dict()
    second = run_health_check(env).to_dict()
    assert first == second  # deterministic for the same environment

    parsed = json.loads(json.dumps(first))
    assert parsed["schema_version"] == 1
    assert parsed["tool"] == "fractal-doctor"
    assert set(parsed) == {
        "schema_version",
        "tool",
        "exit_code",
        "core_ok",
        "summary",
        "checks",
    }
    assert set(parsed["summary"]) == {"ok", "warning", "error"}
    for check in parsed["checks"]:
        assert set(check) == {"key", "label", "state", "required", "detail", "data"}
        assert check["state"] in {STATE_OK, STATE_WARNING, STATE_ERROR}


def test_json_cli_flag_emits_valid_json(capsys, monkeypatch):
    monkeypatch.setattr(
        "fractal_flight_studio.doctor.run_health_check",
        lambda: run_health_check(FakeEnv()),
    )
    assert main(["--json"]) == CORE_OK
    out = json.loads(capsys.readouterr().out)
    assert out["core_ok"] is True
    assert {c["key"] for c in out["checks"]} >= {"python", "cuda", "ffmpeg", "tkinter"}


def test_human_output_separates_core_from_optional():
    text = format_report(run_health_check(FakeEnv(cuda=CUDA_UNAVAILABLE, ffmpeg_found=None)))
    assert "Pflicht (Kern):" in text
    assert "Optionale Fähigkeiten:" in text
    assert "[WARNUNG]" in text
    assert "[FEHLER]" not in text
    assert "Kerninstallation gesund" in text
    assert "Exit 0" in text

    failing = format_report(run_health_check(FakeEnv(python_version=(3, 9, 0))))
    assert "[FEHLER]" in failing
    assert "Kerninstallation fehlerhaft" in failing
    assert "Python-Version" in failing.splitlines()[-1]


def test_cuda_details_reuse_gpu_info_report():
    report = run_health_check(FakeEnv(cuda=CUDA_AVAILABLE))
    text = format_report(report)
    assert "CUDA-Details:" in text
    assert "numba-cuda: 0.22" in text  # rendered by gpu_info.CudaStatus.report()
    cuda_data = _check(report, "cuda").data
    assert set(cuda_data) == set(CudaStatus.__dataclass_fields__)
