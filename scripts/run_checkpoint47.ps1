param(
    [switch]$RebuildInputs
)


$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint46Script = Join-Path $PSScriptRoot "run_checkpoint46.ps1"
$ProcessedDir = Join-Path $ProjectRoot "data\processed"
$RawDir = Join-Path $ProjectRoot "data\raw"

$RequiredProcessedInputs = @(
    "current_retention_policy_scores.csv",
    "current_retention_policy_summary.csv",
    "current_retention_policy_group_summary.csv",
    "retention_policy_decision.csv",
    "retention_final_test_model_metrics.csv",
    "retention_calibration_selection.csv",
    "model_selection_decision_v2.csv"
)

$RequiredRawInputs = @(
    "employees.csv",
    "locations.csv"
)

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

$MissingProcessedInputs = @(
    $RequiredProcessedInputs |
        Where-Object {
            -not (
                Test-Path (
                    Join-Path $ProcessedDir $_
                )
            )
        }
)

$MissingRawInputs = @(
    $RequiredRawInputs |
        Where-Object {
            -not (
                Test-Path (
                    Join-Path $RawDir $_
                )
            )
        }
)

if (
    $RebuildInputs -or
    $MissingProcessedInputs.Count -gt 0 -or
    $MissingRawInputs.Count -gt 0
) {
    if (-not (Test-Path $Checkpoint46Script)) {
        throw "Checkpoint 46 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Rebuild current policy inputs"
    Write-Host "=================================================="

    & $Checkpoint46Script -RebuildInputs
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Build current-state dashboard data"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\build_dashboard_current_state.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 47 current-state dashboard build failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 47 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
