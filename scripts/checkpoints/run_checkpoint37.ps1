$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint36Script = Join-Path $PSScriptRoot "run_checkpoint36.ps1"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $Checkpoint36Script)) {
    throw "Checkpoint 36 generation script was not found."
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "=================================================="
Write-Host "Regenerate Version 2 data"
Write-Host "=================================================="

& $Checkpoint36Script

Write-Host ""
Write-Host "=================================================="
Write-Host "Validate and compare Version 2 data"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\validate_v2_attrition_data.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 37 validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 37 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
