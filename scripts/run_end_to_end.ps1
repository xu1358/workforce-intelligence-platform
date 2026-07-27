param(
    [switch]$SkipDataGeneration,
    [switch]$RunLegacyV1Outputs
)


$ErrorActionPreference = "Stop"


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

$ProjectRoot = Split-Path -Parent $PSScriptRoot

$Python = Join-Path `
    $ProjectRoot `
    ".venv\Scripts\python.exe"


if (-not (Test-Path $Python)) {
    throw (
        "Project virtual environment was not found at: " +
        $Python
    )
}


Set-Location $ProjectRoot


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

function Invoke-PythonScript {

    param(
        [string]$StepName,
        [string]$RelativePath
    )

    $FullPath = Join-Path `
        $ProjectRoot `
        $RelativePath

    if (-not (Test-Path $FullPath)) {
        throw (
            "Required script was not found: " +
            $RelativePath
        )
    }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host $StepName
    Write-Host "=================================================="

    & $Python $FullPath

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Pipeline stopped because this step failed: " +
            $StepName
        )
    }
}


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

$StartTime = Get-Date


Write-Host ""
Write-Host "Workforce Intelligence Platform"
Write-Host "End-to-End Validation"
Write-Host ""
Write-Host "Project root:"
Write-Host $ProjectRoot


# ---------------------------------------------------------
# Python syntax validation
# ---------------------------------------------------------

Write-Host ""
Write-Host "=================================================="
Write-Host "Validate Python syntax"
Write-Host "=================================================="

& $Python -m compileall src dashboard

if ($LASTEXITCODE -ne 0) {
    throw "Python syntax validation failed."
}


# ---------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------

if (-not $SkipDataGeneration) {

    Invoke-PythonScript `
        "Generate reference data" `
        "src\generate_reference_data.py"

    Invoke-PythonScript `
        "Generate full workforce" `
        "src\generate_full_workforce.py"

    Invoke-PythonScript `
        "Generate compensation history" `
        "src\generate_compensation_history.py"

    Invoke-PythonScript `
        "Generate performance reviews" `
        "src\generate_performance_reviews.py"

    Invoke-PythonScript `
        "Generate training records" `
        "src\generate_training_records.py"

    Invoke-PythonScript `
        "Generate employee events" `
        "src\generate_employee_events.py"

    Invoke-PythonScript `
        "Apply attrition hazard and censor histories" `
        "src\generate_attrition_outcomes.py"

    Invoke-PythonScript `
        "Validate Version 2 attrition data" `
        "src\validate_v2_attrition_data.py"

    Invoke-PythonScript `
        "Generate candidates" `
        "src\generate_candidates.py"

    Invoke-PythonScript `
        "Generate job requisitions" `
        "src\generate_job_requisitions.py"

    Invoke-PythonScript `
        "Generate applications" `
        "src\generate_applications.py"
}

else {

    Write-Host ""
    Write-Host "Synthetic data generation was skipped."
}


Invoke-PythonScript `
    "Build multi-snapshot temporal datasets" `
    "src\build_multi_snapshot_retention_dataset.py"

Invoke-PythonScript `
    "Diagnose feature redundancy and stability" `
    "src\diagnose_feature_redundancy.py"

Invoke-PythonScript `
    "Create temporal model splits" `
    "src\create_model_splits.py"

Invoke-PythonScript `
    "Compare Version 2 retention models" `
    "src\compare_retention_models_v2.py"

Invoke-PythonScript `
    "Calibrate selected retention model" `
    "src\calibrate_retention_model.py"

Invoke-PythonScript `
    "Evaluate retention ranking" `
    "src\evaluate_retention_ranking.py"

Invoke-PythonScript `
    "Analyze model fairness and subgroups" `
    "src\analyze_model_fairness.py"

Invoke-PythonScript `
    "Calculate retention economics" `
    "src\calculate_retention_economics.py"

Invoke-PythonScript `
    "Optimize and evaluate retention policy" `
    "src\optimize_retention_policy.py"


# ---------------------------------------------------------
# PostgreSQL pipeline
# ---------------------------------------------------------

Invoke-PythonScript `
    "Test PostgreSQL connection" `
    "src\test_database_connection.py"

Invoke-PythonScript `
    "Create PostgreSQL schema" `
    "src\create_database_schema.py"

Invoke-PythonScript `
    "Validate PostgreSQL schema" `
    "src\validate_database_schema.py"

Invoke-PythonScript `
    "Load CSV data into PostgreSQL" `
    "src\load_postgresql_data.py"

Invoke-PythonScript `
    "Validate PostgreSQL data load" `
    "src\validate_database_load.py"


# ---------------------------------------------------------
# Machine-learning pipeline
# ---------------------------------------------------------

if ($RunLegacyV1Outputs) {

    Write-Host ""
    Write-Host (
        "WARNING: Running legacy Version 1 model and dashboard outputs. " +
        "These outputs must not be used for Version 2 decisions."
    )

    Invoke-PythonScript `
        "Build legacy retention modeling dataset" `
        "src\build_retention_dataset.py"

    Invoke-PythonScript `
        "Train legacy baseline retention model" `
        "src\train_baseline_retention_model.py"

    Invoke-PythonScript `
        "Compare legacy retention models" `
        "src\compare_retention_models.py"

    Invoke-PythonScript `
        "Analyze legacy selected retention model" `
        "src\analyze_retention_model.py"


    # -----------------------------------------------------
    # Dashboard pipeline
    # -----------------------------------------------------

    Invoke-PythonScript `
        "Build legacy dashboard data layer" `
        "src\build_dashboard_data.py"


    # -----------------------------------------------------
    # Legacy final output validation
    # -----------------------------------------------------

    Invoke-PythonScript `
        "Validate legacy final project outputs" `
        "src\validate_project_outputs.py"
}

else {

    Write-Host ""
    Write-Host "Legacy Version 1 modeling and dashboard steps were skipped."
    Write-Host (
        "This protects the reserved Version 2 test period from " +
        "legacy threshold selection."
    )
}

# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------

$EndTime = Get-Date

$ElapsedTime = $EndTime - $StartTime

Write-Host ""
Write-Host "=================================================="
Write-Host "END-TO-END VALIDATION COMPLETED SUCCESSFULLY"
Write-Host "=================================================="

Write-Host (
    "Elapsed time: " +
    $ElapsedTime.ToString()
)
