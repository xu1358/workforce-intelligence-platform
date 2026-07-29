param(
    [switch]$SkipDataGeneration,
    [switch]$RunLegacyV1Outputs,
    [switch]$SkipExternalBenchmark,
    [switch]$SkipPostgres,
    [switch]$SkipQuality
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run_project.py"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

if (-not (Test-Path $Runner)) {
    throw "Cross-platform project runner was not found: $Runner"
}

Set-Location $ProjectRoot

$RunnerArguments = @(
    $Runner,
    "pipeline"
)

if ($SkipDataGeneration) {
    $RunnerArguments += "--skip-data-generation"
}

if ($RunLegacyV1Outputs) {
    $RunnerArguments += "--include-legacy-v1"
}

if ($SkipExternalBenchmark) {
    $RunnerArguments += "--skip-external-benchmark"
}

if ($SkipPostgres) {
    $RunnerArguments += "--skip-postgres"
}

if ($SkipQuality) {
    $RunnerArguments += "--skip-quality"
}

Write-Host ""
Write-Host "=================================================="
Write-Host "Run cross-platform Version 2 pipeline"
Write-Host "=================================================="

& $Python @RunnerArguments

if ($LASTEXITCODE -ne 0) {
    throw "The cross-platform project pipeline failed."
}
