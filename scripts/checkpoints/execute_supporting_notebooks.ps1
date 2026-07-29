$ErrorActionPreference = "Stop"

$ScriptsDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptsDirectory
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

Set-Location $ProjectRoot

$NotebookFiles = @(
    "notebooks\20_v2_data_validation.ipynb",
    "notebooks\21_temporal_dataset_validation.ipynb",
    "notebooks\22_feature_diagnostics.ipynb",
    "notebooks\23_model_comparison_v2.ipynb",
    "notebooks\24_calibration_analysis.ipynb",
    "notebooks\25_ranking_analysis.ipynb",
    "notebooks\26_fairness_analysis.ipynb",
    "notebooks\27_retention_cost_model.ipynb",
    "notebooks\28_retention_policy_analysis.ipynb",
    "notebooks\29_dashboard_current_state_validation.ipynb",
    "notebooks\30_manufacturing_workforce_stability.ipynb"
)

Write-Host ""
Write-Host "=================================================="
Write-Host "Execute Version 2 supporting notebooks"
Write-Host "=================================================="

foreach ($Notebook in $NotebookFiles) {
    Write-Host ""
    Write-Host ("Executing " + $Notebook)

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
Write-Host "Validate saved notebook evidence"
Write-Host "=================================================="

& $Python (
    Join-Path $ProjectRoot "src\validate_portfolio_notebooks.py"
)

if ($LASTEXITCODE -ne 0) {
    throw "Supporting notebook validation failed."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "SUPPORTING NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY"
Write-Host "=================================================="
