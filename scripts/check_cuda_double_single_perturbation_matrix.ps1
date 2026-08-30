param(
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$Repeats = 5,
    [double]$WarmupSeconds = 1.0,
    [string]$Output = "double-single-perturbation-matrix.json"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment not found. Run .\scripts\run_windows.ps1 once first."
}
Push-Location $ProjectRoot
try {
    & $Python .\scripts\check_cuda_double_single_perturbation_matrix_entry.py `
        --width $Width `
        --height $Height `
        --repeats $Repeats `
        --warmup-seconds $WarmupSeconds `
        --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Double-single perturbation matrix failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
