# Code Quality and Continuous Integration

## Purpose

Checkpoint 50 adds a repeatable engineering-quality gate to the Workforce
Intelligence Platform.

Checkpoint 49 answered:

> Do the important calculations and safeguards still behave correctly?

Checkpoint 50 adds:

> Does the Python code satisfy committed quality rules, and will GitHub check
> those rules automatically after repository changes?

The answer is now yes for the configured scope.

## Quality Gates

The local and GitHub workflows run the same four gates in the same order:

1. Compile Python source, dashboard, and tests.
2. Lint Python with Ruff.
3. Verify formatting for the automated-test suite.
4. Run all pytest tests.

A failure in any gate stops the workflow.

## Ruff

Ruff is a fast Python linter and formatter. Its committed configuration is
stored in `pyproject.toml`.

### Target Runtime

The project targets Python 3.12:

```toml
target-version = "py312"
```

This matches the local project environment and the GitHub Actions runner.

### Enforced Lint Rules

The repository currently enforces:

| Rule family | Purpose |
| --- | --- |
| `E4` | Import-related Python style errors |
| `E7` | Invalid or misleading statement structure |
| `E9` | Syntax and runtime parsing errors |
| `F` | Undefined names, unused imports, and unused local variables |

These rules run across:

- `src`
- `dashboard`
- `tests`

They focus on defects that can hide real mistakes instead of introducing a
large cosmetic rewrite.

### Intentional E402 Exceptions

Five plotting scripts set environment variables or the noninteractive
Matplotlib backend before importing plotting modules:

- `src/analyze_employee_survival.py`
- `src/analyze_model_fairness.py`
- `src/analyze_retention_explanations.py`
- `src/evaluate_retention_ranking.py`
- `src/optimize_retention_policy.py`

Those imports intentionally occur after environment setup. `E402` is therefore
ignored only for those named files instead of being disabled for the entire
repository.

### Progressive Formatting

The new automated-test suite is fully formatted and checked with:

```powershell
python -m ruff format --check tests
```

Repository-wide formatting is not applied in this checkpoint. A full Ruff
format pass would rewrite dozens of established analytical files and mix
large cosmetic changes with functional history.

The progressive policy is:

- Enforce correctness-oriented lint rules across all Python now.
- Enforce formatting on new test code now.
- Avoid unrelated bulk rewrites of validated analytical scripts.
- Extend formatting coverage deliberately when files are substantially
  revised.

This keeps Git history reviewable while preventing new test-format drift.

## Small Source Cleanup

Ruff identified two unused elements:

- An unused `date` import in `src/attrition_hazard.py`
- An unused `involuntary_count` local variable in
  `src/validate_v2_attrition_data.py`

Both were removed. Neither value affected an output, branch, saved artifact,
or model calculation.

## Local Checkpoint Runner

Run the complete local quality gate with:

```powershell
.\scripts\checkpoints\run_checkpoint50.ps1
```

The runner:

1. Confirms the project virtual environment exists.
2. Confirms Ruff is installed.
3. Compiles `src`, `dashboard`, and `tests`.
4. Runs Ruff linting.
5. checks test-suite formatting.
6. Runs pytest.
7. Stops immediately after any failure.

The individual commands are:

```powershell
python -m compileall src dashboard tests
python -m ruff check src dashboard tests
python -m ruff format --check tests
python -m pytest -q
```

To format the tests after an intentional test edit:

```powershell
python -m ruff format tests
```

Formatting changes files. The `--check` command does not.

## GitHub Actions

The workflow is stored at:

```text
.github/workflows/python-quality.yml
```

It runs automatically on:

- Pushes to `master`
- Pushes to `revision-v2`
- Pull requests targeting either branch
- Manual workflow dispatch

The job uses:

- A GitHub-hosted Ubuntu runner
- Python 3.12.4
- pip 26.0.1
- A 113-distribution SHA-256-verified dependency lock
- Cached pip dependencies
- Read-only repository-content permission
- A 15-minute timeout
- Concurrency cancellation for superseded runs

If a newer commit is pushed while an older run for the same branch is still
executing, GitHub may cancel the outdated run and keep the newest one.

## GitHub Workflow Steps

The remote job performs:

1. Repository checkout
2. Python 3.12.4 setup
3. pip 26.0.1 installation
4. Hashed dependency-lock installation
5. `pip check` and installed-version validation
6. Markdown and Mermaid rendering validation
7. Python compilation
8. Ruff linting
9. Ruff formatting verification
10. Pytest execution

The workflow does not:

- Connect to PostgreSQL
- Read local `.env` credentials
- Regenerate synthetic data
- Retrain the model
- Access saved final-test outcomes
- Launch Streamlit
- Deploy the dashboard

The workflow is intentionally fast and deterministic.

## Viewing a Workflow Run

After the workflow file is committed and pushed:

1. Open the GitHub repository.
2. Select the **Actions** tab.
3. Select **Python quality**.
4. Open the newest run for `revision-v2`.

A green check means every gate passed.

A red X means at least one gate failed. Open the failed step to see the exact
command and diagnostic message.

The first workflow run can take longer because GitHub must install and cache
the dependencies. Later runs can reuse that cache.

## Test Coverage Added in This Checkpoint

Checkpoint 50 adds configuration tests that verify:

- Ruff targets Python 3.12.
- The committed lint families remain enabled.
- Intentional import-order exceptions stay narrowly scoped.
- GitHub runs on push, pull request, and manual dispatch.
- Official GitHub actions use explicit major versions.
- Local and remote workflows run the same four quality gates.
- Both pytest and Ruff remain declared dependencies.

Checkpoints 51 and 52 extend the same suite with curated-notebook and
recruiter-facing README contracts. Checkpoint 53 adds isolated
external-benchmark source, schema, feature, metric, and governance contracts.
Checkpoint 54 adds exact linear-explanation, raw-feature grouping,
importance-stability, aggregate-privacy, and governance contracts. The
Checkpoint 55 adds censoring, hire-history reconstruction, Kaplan–Meier,
Cox-encoding, and survival-isolation contracts. The complete automated suite
also includes Checkpoint 56 salary-allocation, policy-sensitivity, and
no-retuning governance contracts. Checkpoint 57 adds direct-pin,
transitive-lock, artifact-hash, fingerprint, installed-version, and CI
environment contracts. Checkpoint 58 adds repository-wide CommonMark fence
validation and a specific rendering contract for the data-model Mermaid ER
diagram. Checkpoint 59 adds a cross-platform pipeline-resolution smoke test
and verifies that the Windows wrapper delegates to the canonical Python
runner. Checkpoint 60 makes the attrition-hazard simulation mapping an
executable contract and adds regression coverage for every setting. It also
checks that archived PowerShell runners resolve the repository root from
their new directory. Checkpoint 61 prevents generator-signal acceptance
language from being confused with model selection, validates the chronology
through the unchanged final-test result, and keeps private first-person notes
outside the public repository. Checkpoint 62 adds database-view, source-parity,
pipeline-order, least-privilege projection, and frozen-output fingerprint
contracts for the primary PostgreSQL-backed Version 2 path.

## End-to-End Integration

`scripts/run_project.py pipeline` performs:

1. Compilation
2. Ruff linting
3. Ruff formatting verification
4. Automated testing

before entering expensive data-generation and analytical stages.

This creates a fail-fast boundary: the full pipeline stops early when basic
engineering rules are already broken. `scripts/run_end_to_end.ps1` remains a
thin Windows wrapper around this canonical Python command.

## Files Added or Updated

| File | Purpose |
| --- | --- |
| `.github/workflows/python-quality.yml` | Runs quality checks automatically on GitHub |
| `pyproject.toml` | Stores Ruff lint and formatting configuration |
| `requirements.txt` | Pins every direct dependency exactly |
| `requirements-lock.txt` | Pins and hashes the complete transitive environment |
| `config/dependency_reproducibility.yaml` | Defines Python, pip, count, version, and fingerprint contracts |
| `src/validate_dependency_environment.py` | Compares manifests, hashes, CI, and installed versions |
| `scripts/checkpoints/run_checkpoint57.ps1` | Reproduces dependency and code-quality validation locally |
| `tests/test_dependency_reproducibility.py` | Prevents dependency and CI drift |
| `config/markdown_integrity.yaml` | Defines Markdown scope and Mermaid relationships |
| `src/validate_markdown_docs.py` | Detects unclosed fences and validates the ER diagram |
| `scripts/checkpoints/run_checkpoint58.ps1` | Runs Markdown rendering and regression validation |
| `tests/test_markdown_integrity.py` | Prevents broken Markdown and Mermaid rendering |
| `config/cross_platform_runner.yaml` | Defines portable commands, stages, and archive layout |
| `scripts/run_project.py` | Runs quality, validation, pipeline, dashboard, and notebooks across platforms |
| `src/validate_cross_platform_runner.py` | Reports command, stage, path, CI, and governance checks |
| `tests/test_cross_platform_runner.py` | Tests stage parity, skip flags, paths, and script organization |
| `config/v2_postgresql_source.yaml` | Defines Version 2 views, source parity, output fingerprints, and governance |
| `src/v2_data_access.py` | Implements CSV and PostgreSQL source backends |
| `src/validate_v2_postgresql_integration.py` | Reconciles database sources and temporal outputs |
| `tests/test_v2_postgresql_integration.py` | Tests view, type, runner, and fallback contracts |
| `scripts/checkpoints/run_checkpoint62.ps1` | Runs live PostgreSQL integration and complete quality gates |
| `scripts/checkpoints/run_checkpoint50.ps1` | Reproduces all quality gates locally |
| `scripts/run_end_to_end.ps1` | Maps Windows switches to the canonical Python pipeline |
| `tests/test_quality_configuration.py` | Tests local and GitHub quality contracts |
| `tests/test_retention_policy_equity.py` | Tests pay-band allocation and sensitivity contracts |
| `tests/*.py` | Applies consistent Ruff formatting to the test suite |
| `src/attrition_hazard.py` | Removes one unused import |
| `src/validate_v2_attrition_data.py` | Removes one unused local variable |
| `docs/code_quality_and_ci.md` | Documents scope, commands, workflow, and limitations |
| `docs/testing_strategy.md` | Records that automated CI is now implemented |

## Limitations

The quality workflow does not prove analytical correctness by itself.

It complements, but does not replace:

- The complete automated regression suite
- Dataset validation checkpoints
- Temporal leakage audits
- Model calibration and out-of-time evaluation
- Fairness diagnostics
- Cost and policy sensitivity analysis
- The complete end-to-end pipeline

The workflow also does not currently enforce repository-wide Ruff formatting.
That is a deliberate scope decision to avoid a large cosmetic rewrite of
validated analytical history.

The dependency lock is intentionally conservative. Updates are reviewed,
regenerated, fingerprinted, and tested rather than accepted automatically.
See `docs/dependency_reproducibility.md`.
