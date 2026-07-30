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
Write-Host "Format Checkpoint 66 automated tests"
Write-Host "=================================================="

& $Python -m ruff format `
    "tests\test_v2_exploratory_analysis.py" `
    "tests\test_cross_platform_runner.py" `
    "tests\test_portfolio_notebooks.py"

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 66 test formatting failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run authoritative Version 2 exploratory analysis"
Write-Host "=================================================="

& $Python (Join-Path $ProjectRoot "src\analyze_v2_exploratory_data.py")

if ($LASTEXITCODE -ne 0) {
    throw "Version 2 exploratory analysis failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Audit Version 2 department and location history"
Write-Host "=================================================="

& $Python (Join-Path $ProjectRoot "src\audit_v2_department_history.py")

if ($LASTEXITCODE -ne 0) {
    throw "Version 2 department-history audit failed."
}

foreach ($Notebook in @(
    "notebooks\37_v2_exploratory_analysis.ipynb",
    "notebooks\38_v2_department_history.ipynb"
)) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Execute $Notebook"
    Write-Host "=================================================="

    & $Python -m jupyter nbconvert `
        --to notebook `
        --execute `
        --inplace `
        --ExecutePreprocessor.timeout=900 `
        --ExecutePreprocessor.kernel_name=python3 `
        $Notebook

    if ($LASTEXITCODE -ne 0) {
        throw "Notebook execution failed: $Notebook"
    }
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform quality gates"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 66 quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 66 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
