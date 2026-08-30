$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

. (Join-Path $PSScriptRoot "python_runtime.ps1")

if (-not (Test-CompatibleVenvPython -Path $VenvPython)) {
    if (Test-Path -LiteralPath $VenvDir) {
        Write-Host "Removing incomplete virtual environment..."
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }

    $Python = Resolve-CompatiblePython
    $PythonExe = [string]$Python.Executable
    $PythonArgs = @($Python.Arguments)

    Write-Host "Creating virtual environment with $PythonExe (Python $($Python.Version))..."
    & $PythonExe @PythonArgs -m venv $VenvDir

    if (($LASTEXITCODE -ne 0) -or (-not (Test-CompatibleVenvPython -Path $VenvPython))) {
        throw "Creating the virtual environment failed. Verify that '$PythonExe' is a complete Python installation with the venv module."
    }
}

Write-Host "Installing/updating Fractal Flight Studio..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Updating pip failed." }

$NvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $NvidiaSmi) {
    $KnownNvidiaSmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    if (Test-Path -LiteralPath $KnownNvidiaSmi -PathType Leaf) {
        $NvidiaSmi = $KnownNvidiaSmi
    }
}

if ($NvidiaSmi -and ($env:FRACTAL_SKIP_CUDA -ne "1")) {
    Write-Host "NVIDIA-GPU erkannt; installiere CUDA-12-Unterstützung..."
    & $VenvPython -m pip install -e "${ProjectRoot}[cuda12]"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "CUDA-Abhängigkeiten konnten nicht installiert werden. Installiere die CPU-Basis und starte trotzdem."
        & $VenvPython -m pip install -e $ProjectRoot
    }
} else {
    & $VenvPython -m pip install -e $ProjectRoot
}
if ($LASTEXITCODE -ne 0) { throw "Installing Fractal Flight Studio failed." }

Write-Host "Hardwarediagnose:"
& $VenvPython -m fractal_flight_studio.doctor
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CUDA ist nicht verfügbar; die App verwendet den CPU-Renderer."
}

Write-Host "Starting Fractal Flight Studio..."
& $VenvPython -m fractal_flight_studio.flight_app
if ($LASTEXITCODE -ne 0) { throw "Fractal Flight Studio exited with code $LASTEXITCODE." }
