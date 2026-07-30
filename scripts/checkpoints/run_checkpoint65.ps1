$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $ScriptsDirectory "run_project.py"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "=================================================="
Write-Host "Format Checkpoint 65 automated tests"
Write-Host "=================================================="

& $Python -m ruff format `
    "tests\test_temporal_prior_shift.py" `
    "tests\test_dependency_reproducibility.py" `
    "tests\test_cross_platform_runner.py" `
    "tests\test_portfolio_notebooks.py"

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 65 test formatting failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Audit temporal base-rate and prior shift"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\audit_temporal_prior_shift.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Temporal prior-shift audit failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Execute temporal prior-shift notebook"
Write-Host "=================================================="

& $Python -m jupyter nbconvert `
    --to notebook `
    --execute `
    --inplace `
    --ExecutePreprocessor.timeout=900 `
    --ExecutePreprocessor.kernel_name=python3 `
    "notebooks\36_temporal_prior_shift.ipynb"

if ($LASTEXITCODE -ne 0) {
    throw "Temporal prior-shift notebook execution failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform quality gates"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 65 quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 65 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
