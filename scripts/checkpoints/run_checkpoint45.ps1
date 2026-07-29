param(
    [switch]$RebuildInputs
)


$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint40Script = Join-Path $PSScriptRoot "run_checkpoint40.ps1"
$ProcessedDir = Join-Path $ProjectRoot "data\processed"
$TemporalData = Join-Path (
    $ProcessedDir
) "retention_multi_snapshot.csv"
$CurrentData = Join-Path (
    $ProcessedDir
) "current_active_scoring_population.csv"
$Assignments = Join-Path (
    $ProcessedDir
) "model_split_assignments.csv"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

if (
    $RebuildInputs `
    -or -not (Test-Path $TemporalData) `
    -or -not (Test-Path $CurrentData) `
    -or -not (Test-Path $Assignments)
) {
    if (-not (Test-Path $Checkpoint40Script)) {
        throw "Checkpoint 40 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Rebuild temporal cost-model inputs"
    Write-Host "=================================================="

    & $Checkpoint40Script -RebuildTemporalData
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Calculate retention economics"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\calculate_retention_economics.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 45 retention cost model failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 45 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
