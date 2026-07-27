param(
    [switch]$RebuildRanking
)


$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint43Script = Join-Path $PSScriptRoot "run_checkpoint43.ps1"
$RankingSummary = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "retention_ranking_summary.csv"
$CalibrationPredictions = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "retention_calibration_predictions.csv"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

if (
    $RebuildRanking `
    -or -not (Test-Path $RankingSummary) `
    -or -not (Test-Path $CalibrationPredictions)
) {
    if (-not (Test-Path $Checkpoint43Script)) {
        throw "Checkpoint 43 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Rebuild retention ranking inputs"
    Write-Host "=================================================="

    & $Checkpoint43Script
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Analyze model fairness and subgroups"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\analyze_model_fairness.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 44 fairness analysis failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 44 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
