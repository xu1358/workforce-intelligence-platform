param(
    [switch]$RebuildCalibration
)


$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint42Script = Join-Path $PSScriptRoot "run_checkpoint42.ps1"
$SelectionFile = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "retention_calibration_selection.csv"
$PredictionFile = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "retention_calibration_predictions.csv"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

if (
    $RebuildCalibration `
    -or -not (Test-Path $SelectionFile) `
    -or -not (Test-Path $PredictionFile)
) {
    if (-not (Test-Path $Checkpoint42Script)) {
        throw "Checkpoint 42 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Rebuild retention probability calibration"
    Write-Host "=================================================="

    & $Checkpoint42Script
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Evaluate retention ranking"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\evaluate_retention_ranking.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 43 ranking analysis failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 43 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
