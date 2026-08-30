$ErrorActionPreference = "Stop"

$NvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.Source -and (Test-Path $_.Source) } |
    Select-Object -First 1 -ExpandProperty Source

if (-not $NvidiaSmi) {
    $Candidates = @(
        "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "$env:WINDIR\System32\nvidia-smi.exe"
    )
    $NvidiaSmi = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $NvidiaSmi) {
    throw "nvidia-smi.exe wurde nicht gefunden."
}

Write-Host "GPU-Monitor läuft. Abbruch mit Strg+C."
& $NvidiaSmi `
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,clocks.current.sm,power.draw `
    --format=csv `
    -l 1
