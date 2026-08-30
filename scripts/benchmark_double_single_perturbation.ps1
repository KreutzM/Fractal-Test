param(
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$Repeats = 7,
    [double]$WarmupSeconds = 1.0,
    [double]$BatchTargetSeconds = 0.25,
    [string]$Output = "double-single-perturbation-results.json"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment not found. Run .\scripts\run_windows.ps1 once first."
}
Push-Location $ProjectRoot
try {
    & $Python .\scripts\benchmark_double_single_perturbation.py `
        --width $Width `
        --height $Height `
        --repeats $Repeats `
        --warmup-seconds $WarmupSeconds `
        --batch-target-seconds $BatchTargetSeconds `
        --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Double-single perturbation benchmark failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
