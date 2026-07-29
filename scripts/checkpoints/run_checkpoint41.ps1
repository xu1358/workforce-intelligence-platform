param(
    [switch]$RebuildSplits
)


$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint40Script = Join-Path $PSScriptRoot "run_checkpoint40.ps1"
$SplitAssignments = Join-Path (
    Join-Path $ProjectRoot "data\processed"
) "model_split_assignments.csv"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

if ($RebuildSplits -or -not (Test-Path $SplitAssignments)) {
    if (-not (Test-Path $Checkpoint40Script)) {
        throw "Checkpoint 40 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Create temporal model splits"
    Write-Host "=================================================="

    & $Checkpoint40Script
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Compare Version 2 retention models"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\compare_retention_models_v2.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 41 model comparison failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 41 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
