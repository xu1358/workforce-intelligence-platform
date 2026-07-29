$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Validator = Join-Path $ProjectRoot "src\validate_markdown_docs.py"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $Validator)) {
    throw "Markdown validator was not found: $Validator"
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
Write-Host "Validate Markdown and Mermaid rendering"
Write-Host "=================================================="

& $Python $Validator

if ($LASTEXITCODE -ne 0) {
    throw "Markdown rendering validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run automated test suite"
Write-Host "=================================================="

& $Python -m pytest -q

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 58 automated tests failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 58 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
