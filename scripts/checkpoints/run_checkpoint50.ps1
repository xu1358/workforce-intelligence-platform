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

& $Python -m ruff --version

if ($LASTEXITCODE -ne 0) {
    throw (
        "Ruff is not installed. Run: " +
        "python -m pip install --no-build-isolation --require-hashes -r requirements-lock.txt"
    )
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Compile source, dashboard, and tests"
Write-Host "=================================================="

& $Python -m compileall src dashboard tests

if ($LASTEXITCODE -ne 0) {
    throw "Python syntax validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Lint Python with Ruff"
Write-Host "=================================================="

& $Python -m ruff check src dashboard tests

if ($LASTEXITCODE -ne 0) {
    throw "Ruff lint validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Check automated-test formatting"
Write-Host "=================================================="

& $Python -m ruff format --check tests

if ($LASTEXITCODE -ne 0) {
    throw "Ruff formatting validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run automated test suite"
Write-Host "=================================================="

& $Python -m pytest -q

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 50 automated tests failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 50 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
