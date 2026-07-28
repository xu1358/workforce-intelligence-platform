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

Three plotting scripts set environment variables or the noninteractive
Matplotlib backend before importing plotting modules:

- `src/analyze_model_fairness.py`
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
.\scripts\run_checkpoint50.ps1
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
- Python 3.12
- Cached pip dependencies
- Read-only repository-content permission
- A 15-minute timeout
- Concurrency cancellation for superseded runs

If a newer commit is pushed while an older run for the same branch is still
executing, GitHub may cancel the outdated run and keep the newest one.

## GitHub Workflow Steps

The remote job performs:

1. Repository checkout
2. Python 3.12 setup
3. Dependency installation
4. Python compilation
5. Ruff linting
6. Ruff formatting verification
7. Pytest execution

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

The complete automated suite now contains 76 tests.

## End-to-End Integration

`scripts/run_end_to_end.ps1` now performs:

1. Compilation
2. Ruff linting
3. Ruff formatting verification
4. Automated testing

before entering expensive data-generation and analytical stages.

This creates a fail-fast boundary: the full pipeline stops early when basic
engineering rules are already broken.

## Files Added or Updated

| File | Purpose |
| --- | --- |
| `.github/workflows/python-quality.yml` | Runs quality checks automatically on GitHub |
| `pyproject.toml` | Stores Ruff lint and formatting configuration |
| `requirements.txt` | Adds Ruff as a reproducible dependency |
| `scripts/run_checkpoint50.ps1` | Reproduces all quality gates locally |
| `scripts/run_end_to_end.ps1` | Adds fail-fast quality checks to the full pipeline |
| `tests/test_quality_configuration.py` | Tests local and GitHub quality contracts |
| `tests/*.py` | Applies consistent Ruff formatting to the test suite |
| `src/attrition_hazard.py` | Removes one unused import |
| `src/validate_v2_attrition_data.py` | Removes one unused local variable |
| `docs/code_quality_and_ci.md` | Documents scope, commands, workflow, and limitations |
| `docs/testing_strategy.md` | Records that automated CI is now implemented |

## Limitations

The quality workflow does not prove analytical correctness by itself.

It complements, but does not replace:

- The 76 automated tests
- Dataset validation checkpoints
- Temporal leakage audits
- Model calibration and out-of-time evaluation
- Fairness diagnostics
- Cost and policy sensitivity analysis
- The complete end-to-end pipeline

The workflow also does not currently enforce repository-wide Ruff formatting.
That is a deliberate scope decision to avoid a large cosmetic rewrite of
validated analytical history.
