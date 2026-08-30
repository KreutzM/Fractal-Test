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

Push-Location $ProjectRoot
try {
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Updating pip failed." }

    & $VenvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Installing build dependencies failed." }

    & $VenvPython -m PyInstaller --noconfirm --clean --windowed `
        --name FractalFlightStudio `
        --collect-all numba `
        --collect-all llvmlite `
        src\fractal_flight_studio\flight_app.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

    Write-Host "Build created in dist\FractalFlightStudio"
}
finally {
    Pop-Location
}
