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
Write-Host "Execute and compare SQL and Python temporal logic"
Write-Host "=================================================="

& $Python $Runner postgres

if ($LASTEXITCODE -ne 0) {
    throw "SQL/Python temporal equivalence validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform quality gates"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 63 quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 63 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
