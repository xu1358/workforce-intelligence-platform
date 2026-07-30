$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Validator = Join-Path $ProjectRoot "src\validate_generator_signal_governance.py"
$Runner = Join-Path $ScriptsDirectory "run_project.py"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $Validator)) {
    throw "Generator-signal validator was not found: $Validator"
}

if (-not (Test-Path $Runner)) {
    throw "Cross-platform project runner was not found: $Runner"
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "=================================================="
Write-Host "Validate generator-signal governance"
Write-Host "=================================================="

& $Python $Validator

if ($LASTEXITCODE -ne 0) {
    throw "Generator-signal governance validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform quality gates"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 61 quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 61 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
