param(
    [switch]$RebuildInputs
)


$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint47Script = Join-Path $PSScriptRoot "run_checkpoint47.ps1"
$ProcessedDir = Join-Path $ProjectRoot "data\processed"

$RequiredInputs = @(
    "current_active_scoring_population.csv",
    "current_retention_policy_scores.csv",
    "current_retention_policy_summary.csv",
    "retention_policy_decision.csv"
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
    if (-not (Test-Path $Checkpoint47Script)) {
        throw "Checkpoint 47 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Rebuild current dashboard and policy inputs"
    Write-Host "=================================================="

    & $Checkpoint47Script -RebuildInputs
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Analyze manufacturing workforce stability"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\analyze_workforce_stability.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 48 workforce-stability analysis failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 48 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
