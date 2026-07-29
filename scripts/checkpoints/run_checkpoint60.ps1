$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Validator = Join-Path $ProjectRoot "src\validate_hazard_configuration.py"
$Runner = Join-Path $ScriptsDirectory "run_project.py"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $Validator)) {
    throw "Hazard-configuration validator was not found: $Validator"
}

if (-not (Test-Path $Runner)) {
    throw "Cross-platform project runner was not found: $Runner"
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "=================================================="
Write-Host "Validate executable hazard configuration"
Write-Host "=================================================="

& $Python $Validator

if ($LASTEXITCODE -ne 0) {
    throw "Hazard-configuration validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform quality gates"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 60 quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 60 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
