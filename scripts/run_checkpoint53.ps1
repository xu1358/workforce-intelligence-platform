$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

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
Write-Host "Download and verify IBM benchmark data"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\download_ibm_attrition_benchmark.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "IBM benchmark data download or verification failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run isolated IBM attrition benchmark"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\benchmark_ibm_attrition.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "IBM external benchmark failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run automated test suite"
Write-Host "=================================================="

& $Python -m pytest -q

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 53 automated tests failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 53 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
