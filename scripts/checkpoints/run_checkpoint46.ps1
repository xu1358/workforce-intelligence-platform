param(
    [switch]$RebuildInputs
)


$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint45Script = Join-Path $PSScriptRoot "run_checkpoint45.ps1"
$ProcessedDir = Join-Path $ProjectRoot "data\processed"

$RequiredInputs = @(
    "retention_multi_snapshot.csv",
    "current_active_scoring_population.csv",
    "model_split_assignments.csv",
    "retention_calibration_predictions.csv",
    "retention_calibration_selection.csv",
    "model_selection_decision_v2.csv",
    "retention_cost_scenarios.csv"
)

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

$MissingInputs = @(
    $RequiredInputs |
        Where-Object {
            -not (
                Test-Path (
                    Join-Path $ProcessedDir $_
                )
            )
        }
)

if ($RebuildInputs -or $MissingInputs.Count -gt 0) {
    if (-not (Test-Path $Checkpoint45Script)) {
        throw "Checkpoint 45 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Rebuild policy-analysis inputs"
    Write-Host "=================================================="

    & $Checkpoint45Script -RebuildInputs
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Optimize and evaluate retention policy"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\optimize_retention_policy.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 46 retention policy analysis failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 46 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
