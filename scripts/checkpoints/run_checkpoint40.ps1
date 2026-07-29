param(
    [switch]$RebuildTemporalData
)


$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint38Script = Join-Path $PSScriptRoot "run_checkpoint38.ps1"
$TemporalData = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "retention_multi_snapshot.csv"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

if ($RebuildTemporalData -or -not (Test-Path $TemporalData)) {
    if (-not (Test-Path $Checkpoint38Script)) {
        throw "Checkpoint 38 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Build and validate temporal datasets"
    Write-Host "=================================================="

    & $Checkpoint38Script
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Create temporal model splits"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\create_model_splits.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 40 model split creation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 40 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
