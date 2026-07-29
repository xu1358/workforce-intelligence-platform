param(
    [switch]$RebuildComparison
)


$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint41Script = Join-Path $PSScriptRoot "run_checkpoint41.ps1"
$SelectionFile = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "model_selection_decision_v2.csv"
$PredictionFile = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "retention_validation_predictions_v2.csv"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

if (
    $RebuildComparison `
    -or -not (Test-Path $SelectionFile) `
    -or -not (Test-Path $PredictionFile)
) {
    if (-not (Test-Path $Checkpoint41Script)) {
        throw "Checkpoint 41 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Rebuild Version 2 model comparison"
    Write-Host "=================================================="

    & $Checkpoint41Script
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Calibrate selected retention model"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\calibrate_retention_model.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 42 calibration analysis failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 42 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
