"""Installation health check for Fractal Flight Studio.

The doctor separates *required core health* (Python version, core packages,
CPU renderer, Tkinter for the desktop GUI) from *optional capabilities*
(CUDA, FFmpeg).  A healthy CPU-only installation must exit 0 even when
CUDA and FFmpeg are missing, so the launcher scripts can keep using
``fractal-doctor`` as a hardware overview without failing CPU-only users.

Environment probing (:func:`run_health_check` and the ``check_*``
functions) is kept independent from CLI presentation so both can be
unit-tested without a GPU, FFmpeg or a display.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from types import ModuleType
from typing import Protocol, Sequence

from .gpu_info import CudaStatus, inspect_cuda

MIN_SUPPORTED_PYTHON = (3, 11)
MAX_SUPPORTED_PYTHON = (3, 13)

STATE_OK = "OK"
STATE_WARNING = "WARNING"
STATE_ERROR = "ERROR"

#: Core health is required for the desktop application.  This repository
#: ships a Tkinter desktop GUI as its primary entry point (``fractal-studio``),
#: so a missing Tkinter is reported as a core failure; the offline CLI
#: renderer stays usable, which the check detail states explicitly.
TKINTER_IS_CORE = True

CORE_OK = 0
CORE_ERROR = 1


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single doctor check."""

    key: str
    label: str
    state: str  # STATE_OK | STATE_WARNING | STATE_ERROR
    required: bool
    detail: str
    data: dict | None = None

    @property
    def failed_required(self) -> bool:
        return self.required and self.state == STATE_ERROR


class ProbeEnv(Protocol):
    """Structural interface of the environment used by the probes.

    The real implementation is :class:`SystemProbeEnv`; tests inject fakes
    so every scenario from the issue (missing package, no CUDA, unsupported
    Python, ...) is reproducible without touching the host.
    """

    python_version: tuple[int, int, int]

    def package_version(self, name: str) -> str | None:
        raise NotImplementedError

    def import_module(self, name: str) -> ModuleType:
        raise NotImplementedError

    def find_executable(self, name: str) -> str | None:
        raise NotImplementedError

    def inspect_cuda(self) -> CudaStatus:
        raise NotImplementedError

    def render_cpu_test_frame(self) -> dict:
        raise NotImplementedError


class SystemProbeEnv:
    """Default probe environment backed by the running interpreter."""

    def __init__(self) -> None:
        info = sys.version_info
        self.python_version: tuple[int, int, int] = (info.major, info.minor, info.micro)

    def package_version(self, name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    def import_module(self, name: str):
        return importlib.import_module(name)

    def find_executable(self, name: str) -> str | None:
        return shutil.which(name)

    def inspect_cuda(self) -> CudaStatus:
        return inspect_cuda()

    def render_cpu_test_frame(self) -> dict:
        """Render a tiny frame through the real CPU path to prove usability."""
        from .models import RenderRequest
        from .renderers.cpu import CpuRenderer

        renderer = CpuRenderer()
        result = renderer.render(RenderRequest(width=16, height=16, max_iterations=8))
        return {
            "backend": result.backend,
            "width": 16,
            "height": 16,
            "elapsed_seconds": round(result.elapsed_seconds, 4),
        }


def _format_python(version: Sequence[int]) -> str:
    return ".".join(str(part) for part in version[:3])


def check_python(env: ProbeEnv) -> CheckResult:
    version = tuple(env.python_version[:3])
    text = _format_python(version)
    supported = MIN_SUPPORTED_PYTHON <= version[:2] <= MAX_SUPPORTED_PYTHON
    if supported:
        return CheckResult(
            key="python",
            label="Python-Version",
            state=STATE_OK,
            required=True,
            detail=f"Python {text} (unterstützt: 3.11–3.13)",
            data={"version": text},
        )
    return CheckResult(
        key="python",
        label="Python-Version",
        state=STATE_ERROR,
        required=True,
        detail=(
            f"Python {text} wird nicht unterstützt "
            "(erforderlich: 3.11–3.13); bitte eine unterstützte Version verwenden"
        ),
        data={"version": text},
    )


#: (distribution name, import name, human label)
_REQUIRED_PACKAGES = (
    ("numpy", "numpy", "NumPy"),
    ("numba", "numba", "Numba"),
    ("pillow", "PIL", "Pillow"),
    ("mpmath", "mpmath", "mpmath"),
)


def check_package(
    env: ProbeEnv, distribution: str, import_name: str, label: str
) -> CheckResult:
    detail_data: dict = {"distribution": distribution}
    try:
        module = env.import_module(import_name)
    except Exception as exc:  # ImportError plus any broken-install surprise
        return CheckResult(
            key=f"package:{distribution}",
            label=label,
            state=STATE_ERROR,
            required=True,
            detail=f"{label} ist nicht installierbar: {exc}",
            data=detail_data,
        )
    version = getattr(module, "__version__", None) or env.package_version(distribution)
    detail_data["version"] = version
    return CheckResult(
        key=f"package:{distribution}",
        label=label,
        state=STATE_OK,
        required=True,
        detail=f"{label} {version or '( Versionsnummer unbekannt )'}",
        data=detail_data,
    )


def check_cpu_renderer(env: ProbeEnv) -> CheckResult:
    try:
        info = env.render_cpu_test_frame()
    except Exception as exc:
        return CheckResult(
            key="cpu_renderer",
            label="CPU-Renderer",
            state=STATE_ERROR,
            required=True,
            detail=f"CPU-Renderer nicht verwendbar: {exc}",
        )
    return CheckResult(
        key="cpu_renderer",
        label="CPU-Renderer",
        state=STATE_OK,
        required=True,
        detail=f"Testbild gerendert über '{info['backend']}'",
        data=info,
    )


def check_tkinter(env: ProbeEnv) -> CheckResult:
    try:
        env.import_module("tkinter")
    except Exception as exc:
        return CheckResult(
            key="tkinter",
            label="Tkinter (Desktop-GUI)",
            state=STATE_ERROR if TKINTER_IS_CORE else STATE_WARNING,
            required=TKINTER_IS_CORE,
            detail=(
                "Tkinter fehlt — die Desktop-GUI 'fractal-studio' ist nicht "
                f"startbar ({exc}); Offline-Rendering bleibt möglich. "
                "Installationspaket oder Python-Build ohne Tk-Unterstützung?"
            ),
        )
    return CheckResult(
        key="tkinter",
        label="Tkinter (Desktop-GUI)",
        state=STATE_OK,
        required=TKINTER_IS_CORE,
        detail="Tkinter verfügbar (Headless-Tests ohne Display möglich)",
    )


def check_ffmpeg(env: ProbeEnv) -> CheckResult:
    executable = env.find_executable("ffmpeg")
    if executable is None:
        return CheckResult(
            key="ffmpeg",
            label="FFmpeg (MP4-Export)",
            state=STATE_WARNING,
            required=False,
            detail=(
                "optional — nicht gefunden; MP4-Export deaktiviert "
                "(PNG-Sequenzen und interaktive App bleiben nutzbar)"
            ),
        )
    return CheckResult(
        key="ffmpeg",
        label="FFmpeg (MP4-Export)",
        state=STATE_OK,
        required=False,
        detail=f"gefunden: {executable}",
        data={"executable": executable},
    )


def check_cuda(env: ProbeEnv) -> CheckResult:
    status = env.inspect_cuda()
    data = asdict(status)
    if status.available:
        return CheckResult(
            key="cuda",
            label="CUDA (optionale Beschleunigung)",
            state=STATE_OK,
            required=False,
            detail=status.summary,
            data=data,
        )
    detail = status.summary
    if status.nvidia_smi_found:
        detail += " — NVIDIA-GPU erkannt, CUDA-Pfad jedoch nicht aktiv."
    else:
        detail += " — CPU-Rendering ist vollständig nutzbar."
    return CheckResult(
        key="cuda",
        label="CUDA (optionale Beschleunigung)",
        state=STATE_WARNING,
        required=False,
        detail=detail,
        data=data,
    )


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Full doctor outcome; ``exit_code`` is the CLI contract."""

    checks: tuple[CheckResult, ...]
    exit_code: int

    @property
    def core_ok(self) -> bool:
        return self.exit_code == CORE_OK

    @property
    def has_errors(self) -> bool:
        return any(check.state == STATE_ERROR for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.state == STATE_WARNING for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "tool": "fractal-doctor",
            "exit_code": self.exit_code,
            "core_ok": self.core_ok,
            "summary": {
                "ok": sum(c.state == STATE_OK for c in self.checks),
                "warning": sum(c.state == STATE_WARNING for c in self.checks),
                "error": sum(c.state == STATE_ERROR for c in self.checks),
            },
            "checks": [asdict(check) for check in self.checks],
        }


def run_health_check(env: ProbeEnv | None = None) -> HealthReport:
    """Probe the installation; presentation-free and unit-testable."""

    env = env or SystemProbeEnv()
    checks = [check_python(env)]
    checks.extend(
        check_package(env, distribution, import_name, label)
        for distribution, import_name, label in _REQUIRED_PACKAGES
    )
    checks.append(check_cpu_renderer(env))
    checks.append(check_tkinter(env))
    checks.append(check_ffmpeg(env))
    checks.append(check_cuda(env))
    exit_code = CORE_ERROR if any(c.failed_required for c in checks) else CORE_OK
    return HealthReport(checks=tuple(checks), exit_code=exit_code)


def format_report(report: HealthReport) -> str:
    """Human-readable output; optional capabilities are clearly separated."""

    markers = {STATE_OK: "[ OK ]   ", STATE_WARNING: "[WARNUNG]", STATE_ERROR: "[FEHLER]"}
    required_checks = [c for c in report.checks if c.required]
    optional_checks = [c for c in report.checks if not c.required]
    lines = ["Fractal Flight Studio — Installations-Health-Check", ""]
    lines.append("Pflicht (Kern):")
    lines.extend(f"  {markers[c.state]} {c.label}: {c.detail}" for c in required_checks)
    lines.append("")
    lines.append("Optionale Fähigkeiten:")
    lines.extend(f"  {markers[c.state]} {c.label}: {c.detail}" for c in optional_checks)
    lines.append("")
    cuda_check = next(c for c in report.checks if c.key == "cuda")
    if cuda_check.data:
        status = CudaStatus(**cuda_check.data)
        lines.append("CUDA-Details:")
        for line in status.report().splitlines():
            lines.append(f"  {line}")
        lines.append("")
    if report.core_ok:
        if report.has_warnings:
            lines.append(
                "Ergebnis: Kerninstallation gesund (Exit 0) — optionale "
                "Einschränkungen siehe WARNUNG-Einträge."
            )
        else:
            lines.append("Ergebnis: Installation vollständig gesund (Exit 0).")
    else:
        failing = ", ".join(
            c.label for c in report.checks if c.state == STATE_ERROR and c.required
        )
        lines.append(f"Ergebnis: Kerninstallation fehlerhaft (Exit 1) — {failing}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fractal-doctor",
        description=(
            "Installations-Health-Check für Fractal Flight Studio "
            "(Kernprüfung plus optionale Fähigkeiten CUDA/FFmpeg)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    args = parser.parse_args(argv)

    report = run_health_check()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
