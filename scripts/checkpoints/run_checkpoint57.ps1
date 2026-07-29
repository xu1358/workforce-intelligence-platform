$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DependencyValidator = Join-Path `
    $ProjectRoot `
    "src\validate_dependency_environment.py"
$LockFile = Join-Path $ProjectRoot "requirements-lock.txt"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $DependencyValidator)) {
    throw "Dependency validator was not found: $DependencyValidator"
}

if (-not (Test-Path $LockFile)) {
    throw "Dependency lock was not found: $LockFile"
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "=================================================="
Write-Host "Validate locked dependency environment"
Write-Host "=================================================="

& $Python $DependencyValidator

if ($LASTEXITCODE -ne 0) {
    throw (
        "The virtual environment does not match requirements-lock.txt. " +
        "Install the committed lock before rerunning Checkpoint 57."
    )
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Check dependency compatibility"
Write-Host "=================================================="

& $Python -m pip check

if ($LASTEXITCODE -ne 0) {
    throw "pip reported an incompatible or missing dependency."
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
    throw "Checkpoint 57 automated tests failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 57 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
