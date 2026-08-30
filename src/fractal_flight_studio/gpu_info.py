from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import os
import shutil
import subprocess
from typing import Callable


@dataclass(frozen=True, slots=True)
class CudaStatus:
    available: bool
    device_name: str | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    numba_version: str | None = None
    numba_cuda_version: str | None = None
    reason: str | None = None
    nvidia_smi_found: bool = False

    @property
    def summary(self) -> str:
        if self.available:
            device = self.device_name or "NVIDIA CUDA device"
            extras: list[str] = []
            if self.compute_capability:
                extras.append(f"CC {self.compute_capability}")
            if self.driver_version:
                extras.append(f"Treiber {self.driver_version}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            return f"CUDA verfügbar: {device}{suffix}"
        return f"CUDA nicht verfügbar: {self.reason or 'unbekannter Grund'}"

    def report(self) -> str:
        lines = [self.summary]
        lines.append(f"nvidia-smi: {'gefunden' if self.nvidia_smi_found else 'nicht gefunden'}")
        lines.append(f"Numba: {self.numba_version or 'nicht installiert'}")
        lines.append(f"numba-cuda: {self.numba_cuda_version or 'nicht installiert'}")
        return "\n".join(lines)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _candidate_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if found:
        return found
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        candidate = system_root / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _query_nvidia_smi(
    executable: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str | None, str | None, str | None]:
    executable = executable or _candidate_nvidia_smi()
    if not executable:
        return None, None, None
    try:
        completed = runner(
            [
                executable,
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = completed.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        name = parts[0] if parts else None
        driver = parts[1] if len(parts) > 1 else None
        capability = parts[2] if len(parts) > 2 else None
        return name, driver, capability
    except (OSError, subprocess.SubprocessError, IndexError):
        return None, None, None


def inspect_cuda() -> CudaStatus:
    numba_version = _distribution_version("numba")
    numba_cuda_version = _distribution_version("numba-cuda")
    smi_path = _candidate_nvidia_smi()
    smi_name, driver_version, smi_capability = _query_nvidia_smi(smi_path)

    try:
        from numba import cuda
    except Exception as exc:
        return CudaStatus(
            available=False,
            device_name=smi_name,
            compute_capability=smi_capability,
            driver_version=driver_version,
            numba_version=numba_version,
            numba_cuda_version=numba_cuda_version,
            reason=f"CUDA-Modul konnte nicht importiert werden: {exc}",
            nvidia_smi_found=smi_path is not None,
        )

    try:
        if not cuda.is_available():
            reason = "Numba meldet keinen verwendbaren CUDA-Treiber"
            try:
                # This intentionally asks the driver layer for the detailed reason.
                cuda.cudadrv.driver.driver.ensure_initialized()
            except Exception as exc:
                detail = " ".join(str(exc).split())
                if detail:
                    reason = detail
            if smi_name and numba_cuda_version is None:
                reason += "; installiere die CUDA-Abhängigkeiten mit scripts\\enable_cuda.ps1"
            return CudaStatus(
                available=False,
                device_name=smi_name,
                compute_capability=smi_capability,
                driver_version=driver_version,
                numba_version=numba_version,
                numba_cuda_version=numba_cuda_version,
                reason=reason,
                nvidia_smi_found=smi_path is not None,
            )

        devices = list(cuda.gpus)
        device_name = smi_name
        capability = smi_capability
        if devices:
            device = devices[0]
            raw_name = getattr(device, "name", None)
            if isinstance(raw_name, bytes):
                raw_name = raw_name.decode(errors="replace")
            if raw_name:
                device_name = str(raw_name)
            raw_cc = getattr(device, "compute_capability", None)
            if raw_cc and len(raw_cc) >= 2:
                capability = f"{raw_cc[0]}.{raw_cc[1]}"

        return CudaStatus(
            available=True,
            device_name=device_name,
            compute_capability=capability,
            driver_version=driver_version,
            numba_version=numba_version,
            numba_cuda_version=numba_cuda_version,
            nvidia_smi_found=smi_path is not None,
        )
    except Exception as exc:
        return CudaStatus(
            available=False,
            device_name=smi_name,
            compute_capability=smi_capability,
            driver_version=driver_version,
            numba_version=numba_version,
            numba_cuda_version=numba_cuda_version,
            reason=" ".join(str(exc).split()) or type(exc).__name__,
            nvidia_smi_found=smi_path is not None,
        )
