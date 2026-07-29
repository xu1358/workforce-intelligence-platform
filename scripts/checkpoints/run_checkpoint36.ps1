$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

$Steps = @(
    @("Generate reference data", "src\generate_reference_data.py"),
    @("Generate potential workforce", "src\generate_full_workforce.py"),
    @("Generate potential compensation", "src\generate_compensation_history.py"),
    @("Generate potential performance", "src\generate_performance_reviews.py"),
    @("Generate potential training", "src\generate_training_records.py"),
    @("Generate potential employee events", "src\generate_employee_events.py"),
    @("Apply monthly attrition hazard", "src\generate_attrition_outcomes.py")
)

foreach ($Step in $Steps) {
    $StepName = $Step[0]
    $RelativePath = $Step[1]

    Write-Host ""
    Write-Host "=================================================="
    Write-Host $StepName
    Write-Host "=================================================="

    & $Python (Join-Path $ProjectRoot $RelativePath)

    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint 36 stopped at: $StepName"
    }
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 36 GENERATION COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
