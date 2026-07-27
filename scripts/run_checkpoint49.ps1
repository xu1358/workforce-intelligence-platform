$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Tests = Join-Path $ProjectRoot "tests"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $Tests)) {
    throw "Automated test directory was not found: $Tests"
}

Set-Location $ProjectRoot

& $Python -c "import pytest"

if ($LASTEXITCODE -ne 0) {
    throw (
        "pytest is not installed. Run: " +
        "python -m pip install -r requirements.txt"
    )
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Compile source and automated tests"
Write-Host "=================================================="

& $Python -m compileall src tests

if ($LASTEXITCODE -ne 0) {
    throw "Python syntax validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run automated test suite"
Write-Host "=================================================="

& $Python -m pytest -q

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 49 automated tests failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 49 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
