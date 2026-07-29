$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $ScriptsDirectory "run_project.py"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $Runner)) {
    throw "Cross-platform project runner was not found: $Runner"
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "=================================================="
Write-Host "Organize historical checkpoint runners"
Write-Host "=================================================="

& $Python $Runner organize-checkpoints

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint runner organization failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Validate cross-platform runner"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Cross-platform quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 59 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
