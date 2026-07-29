param(
    [switch]$RebuildTemporalData
)


$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
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
Write-Host "Diagnose feature redundancy and stability"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\diagnose_feature_redundancy.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 39 feature diagnostics failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 39 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
