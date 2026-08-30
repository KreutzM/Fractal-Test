$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "Keine Projektumgebung gefunden. Starte zuerst scripts\run_windows.ps1."
}

Write-Host "Installiere NVIDIA CUDA-12-Abhängigkeiten für Fractal Flight Studio..."
& $VenvPython -m pip install -e "${ProjectRoot}[cuda12]"
if ($LASTEXITCODE -ne 0) { throw "CUDA-Installation fehlgeschlagen." }

Write-Host "Prüfe CUDA..."
& $VenvPython -m fractal_flight_studio.doctor
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CUDA wurde installiert, ist aber noch nicht verwendbar. Prüfe den NVIDIA-Treiber und die obige Diagnose."
}
