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
Write-Host "Format Checkpoint 64 automated tests"
Write-Host "=================================================="

& $Python -m ruff format `
    "tests\test_deployed_policy_fairness.py" `
    "tests\test_cross_platform_runner.py"

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 64 test formatting failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Audit exact deployed-policy subgroup fairness"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\audit_retention_policy_fairness.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Deployed-policy fairness audit failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Execute deployed-policy fairness notebook"
Write-Host "=================================================="

& $Python -m jupyter nbconvert `
    --to notebook `
    --execute `
    --inplace `
    --ExecutePreprocessor.timeout=900 `
    --ExecutePreprocessor.kernel_name=python3 `
    "notebooks\35_deployed_policy_fairness.ipynb"

if ($LASTEXITCODE -ne 0) {
    throw "Deployed-policy fairness notebook execution failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform quality gates"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 64 quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 64 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
