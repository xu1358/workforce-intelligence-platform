$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $ScriptsDirectory "run_project.py"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

$LegacyScreenshots = @(
    "docs\screenshots\01_overview.png",
    "docs\screenshots\02(1)_workforce.png",
    "docs\screenshots\02_workforce.png",
    "docs\screenshots\03(1)_recruiting.png",
    "docs\screenshots\03_recruiting.png",
    "docs\screenshots\04_retention_risk.png",
    "docs\screenshots\05_risk_table.png",
    "docs\screenshots\06(1)_model_performance.png",
    "docs\screenshots\06_model_performance.png"
)

Write-Host ""
Write-Host "=================================================="
Write-Host "Remove stale Version 1 dashboard screenshots"
Write-Host "=================================================="

foreach ($RelativePath in $LegacyScreenshots) {
    $ScreenshotPath = Join-Path $ProjectRoot $RelativePath

    if (Test-Path -LiteralPath $ScreenshotPath) {
        Remove-Item -LiteralPath $ScreenshotPath -Force
        Write-Host "Removed $RelativePath"
    }
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Format Checkpoint 67 automated tests"
Write-Host "=================================================="

& $Python -m ruff format `
    "tests\test_dashboard_evidence.py" `
    "tests\test_cross_platform_runner.py" `
    "tests\test_quality_configuration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 67 test formatting failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Validate current dashboard evidence"
Write-Host "=================================================="

& $Python (Join-Path $ProjectRoot "src\validate_dashboard_evidence.py")

if ($LASTEXITCODE -ne 0) {
    throw "Dashboard evidence validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform quality gates"
Write-Host "=================================================="

& $Python $Runner quality

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 67 quality validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 67 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
