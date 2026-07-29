$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

$RequiredInputs = @(
    "data\processed\retention_multi_snapshot.csv",
    "data\processed\retention_calibration_predictions.csv",
    "data\processed\retention_final_test_predictions.csv",
    "data\processed\current_retention_policy_scores.csv",
    "data\processed\retention_policy_comparison.csv"
)

$MissingInputs = @(
    $RequiredInputs |
        Where-Object { -not (Test-Path (Join-Path $ProjectRoot $_)) }
)

if ($MissingInputs.Count -gt 0) {
    throw (
        "Checkpoint 56 requires completed Checkpoints 42 and 46. " +
        "Missing inputs: " + ($MissingInputs -join ", ")
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
Write-Host "Audit retention policy allocation equity"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\audit_retention_policy_equity.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Retention policy allocation-equity audit failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run automated test suite"
Write-Host "=================================================="

& $Python -m pytest -q

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 56 automated tests failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 56 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
