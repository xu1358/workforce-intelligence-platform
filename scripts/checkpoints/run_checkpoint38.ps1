param(
    [switch]$RegenerateV2Data
)


$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint37Script = Join-Path $PSScriptRoot "run_checkpoint37.ps1"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

if ($RegenerateV2Data) {
    if (-not (Test-Path $Checkpoint37Script)) {
        throw "Checkpoint 37 runner was not found."
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Regenerate and validate Version 2 data"
    Write-Host "=================================================="

    & $Checkpoint37Script
}

$RequiredRawFiles = @(
    "employees.csv",
    "departments.csv",
    "locations.csv",
    "job_roles.csv",
    "compensation_history.csv",
    "performance_reviews.csv",
    "training_records.csv",
    "employee_events.csv"
)

foreach ($FileName in $RequiredRawFiles) {
    $FullPath = Join-Path $ProjectRoot "data\raw\$FileName"

    if (-not (Test-Path $FullPath)) {
        throw (
            "Missing required Version 2 data file: " +
            "$FileName. Run Checkpoint 37 first."
        )
    }
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Build multi-snapshot temporal datasets"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot `
        "src\build_multi_snapshot_retention_dataset.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 38 temporal dataset construction failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "CHECKPOINT 38 COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
