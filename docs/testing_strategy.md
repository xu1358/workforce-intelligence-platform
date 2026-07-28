# Automated Testing Strategy

## Purpose

Checkpoint 49 adds a deterministic automated test suite for the Version 2
workforce-intelligence pipeline.

The suite verifies small, high-risk pieces of analytical logic without
regenerating the full 10,000-employee synthetic workforce, connecting to
PostgreSQL, retraining models, or accessing unknown current outcomes.

The tests are designed to answer:

> If an important formula, temporal boundary, configuration contract, or
> governance rule changes unexpectedly, will the project detect it quickly?

The answer after this checkpoint is yes for the covered contracts.

## What the Tests Cover

### 1. Configuration Contracts

The configuration tests verify that:

- Monthly baseline attrition probabilities form a valid distribution.
- The current as-of date is consistently `2026-06-30`.
- Historical prediction windows are chronological and nonoverlapping.
- Feature-development periods align with train and validation periods.
- The reserved feature holdout aligns with the final test snapshot.
- Model-eligible and protected hierarchy levels remain consistent.
- The economic model retains all 27 scenario combinations.
- The frozen policy retains its 700-person capacity.
- Policy freeze and once-only final-test rules remain enabled.
- Current decisions require human review and prohibit automatic action.

These tests protect agreements that span several YAML files. A single
configuration file may look valid by itself while still conflicting with
another stage of the pipeline.

### 2. Attrition-Hazard Logic

The hazard tests verify:

- Configuration validation.
- Inclusive monthly date generation.
- Nonnegative calendar-month recency.
- Strict feature cutoffs.
- Trailing-window boundaries.
- Feature clamping and supported transforms.
- Historical location reconstruction.
- Performance defaults and trends.
- Compensation growth and promotion recency.
- Training and manager-change windows.
- Peer-relative salary positioning.
- Competing-risk probability bounds.
- Termination-date sampling boundaries.

The tests use small examples with known answers. They do not recalibrate the
hazard or sample a new workforce.

### 3. Temporal Feature Engineering

The temporal tests verify:

- First and latest records are selected on or before a snapshot.
- Month differences use calendar months.
- Missing source dates remain distinguishable from real dates.
- Future location transfers cannot alter historical features.
- Salary position uses the correct job-family and job-level peer group.

These checks directly protect against temporal leakage.

### 4. Feature Policy and Preprocessing

The feature-policy tests verify:

- All 21 selected raw features remain in their committed order.
- Job level and promotion recommendation are normalized consistently.
- Promotion recency becomes missing when no prior promotion exists.
- The reference-category preprocessor produces 35 encoded features.
- Encoded feature names remain unique.
- Unknown categories can be scored without changing feature width.
- Every modeling-dataset column belongs to exactly one policy group.
- A missing policy file fails with an actionable error.

The fixture deliberately covers every configured categorical level.

### 5. Model-Split Safety

The split tests verify:

- Train, validation, and final-test dates remain chronological.
- Every snapshot maps to the expected prediction-window end.
- Protected hierarchy rows remain outside the model population.
- Final-test targets are masked during split creation.
- Repeated employee histories stay inside one grouped fold.
- Training and validation folds remain employee-disjoint.
- Final-test rows receive no development fold.
- Test positive counts and rates are redacted from summaries.
- Saved split assignments contain no target column.

These checks protect the project’s once-only out-of-time evaluation design.

### 6. Economics and Policy

The economics and policy tests verify:

- All nine named assumption rows and all 27 scenario combinations.
- Replacement-cost, avoidable-value, break-even, and coverage formulas.
- Department salary summaries.
- Employee expected-net-value calculations.
- Stable employee-ID tie-breaking.
- A cost threshold can correctly select nobody.
- Frozen policy hashes are deterministic.
- Capacity, budget, and positive-value constraints are enforced.
- Expected and outcome-aligned policy metrics reconcile to hand calculations.

No test changes the frozen policy or reads the final-test outcome file.

### 7. Workforce-Stability Planning

The workforce-stability tests verify:

- Current score coverage exactly matches eligible active employees.
- Protected hierarchy employees remain in workforce headcount.
- Frozen governance assumptions are retained.
- Expected departures equal summed probabilities.
- Expected prevention uses the configured success probability.
- Backfill, intervention spending, avoided cost, and net value reconcile.
- Department and Manufacturing role summaries agree.
- Scenario caveats remain available for the dashboard.

### 8. External-Benchmark Contracts

Checkpoint 53 adds tests that verify:

- The IBM source is pinned to a 40-character commit and SHA-256 checksum.
- The dataset is explicitly fictional and its ODbL/DbCL license is recorded.
- The committed 1,470-row, 35-column schema and target counts are enforced.
- Target, identifier, constant, opaque-rate, and gender fields remain outside
  the benchmark model.
- Top-decile tie-breaking and ranking formulas are deterministic.
- Every configured age boundary receives one descriptive band.
- Primary and IBM performance rows are always marked not directly comparable.
- The benchmark cannot change the primary model, final test, policy,
  dashboard, or employee review plan.

These tests use in-memory data frames. They do not download the external CSV,
fit the 50 benchmark folds, or require network access in GitHub Actions.

## Test Isolation

The suite uses small data frames created inside the tests.

It does not require:

- Files under `data/raw`
- Files under `data/processed`
- A running PostgreSQL server
- Saved machine-learning model artifacts
- Network access
- Streamlit
- Regeneration of the synthetic workforce

This isolation keeps the suite fast and makes failures easier to diagnose.

## Running Checkpoint 49

Install the updated requirements once:

```powershell
python -m pip install -r requirements.txt
```

Then run:

```powershell
.\scripts\run_checkpoint49.ps1
```

The runner:

1. Confirms that the project virtual environment exists.
2. Confirms that `pytest` is installed.
3. Compiles `src` and `tests`.
4. Runs the complete test suite.
5. Stops immediately if any test fails.

The tests can also be run directly:

```powershell
python -m pytest -q
```

To run one test file:

```powershell
python -m pytest tests\test_model_splits.py -q
```

To run one named test:

```powershell
python -m pytest tests\test_model_splits.py::test_reserved_test_has_no_group_fold -q
```

## Reading the Result

A successful run ends with a summary similar to:

```text
70 passed
```

If a test fails, `pytest` reports:

- The test name
- The assertion that failed
- The observed value
- The source line

The checkpoint runner will not print its success message after a failed test.

## End-to-End Integration

`scripts/run_end_to_end.ps1` now compiles the test directory and runs the test
suite before expensive data generation or model work.

This provides an early stop if a core contract has already been broken.

The automated tests do not replace the end-to-end pipeline. The full pipeline
still validates:

- Large generated datasets
- Distributional calibration targets
- Model fitting and uncertainty estimates
- PostgreSQL schema and data loading
- Saved dashboard outputs

The two validation layers serve different purposes:

- Automated tests provide fast, isolated regression detection.
- The end-to-end pipeline validates complete system behavior.

## Portfolio Presentation Contracts

Checkpoints 51 and 52 extend the suite beyond analytical calculations:

- `tests/test_portfolio_notebooks.py` protects the ordered four-notebook
  reviewer path, execution state, navigation, portability, synthetic
  disclosure, and aggregate-only presentation.
- `tests/test_readme_portfolio.py` protects README sections, headline evidence,
  local links, Markdown rendering, reproduction commands, governance language,
  and removal of stale Version 1 claims.

The complete suite now contains 96 tests. The presentation and external-source
contract tests do not recalculate the primary model; they make sure the
committed portfolio tells the validated Version 2 story accurately.

## Files Added or Updated

| File | Purpose |
| --- | --- |
| `pytest.ini` | Defines test discovery and strict pytest behavior |
| `tests/conftest.py` | Provides shared paths and configuration fixtures |
| `tests/test_configuration_contracts.py` | Verifies cross-file configuration agreements |
| `tests/test_attrition_hazard.py` | Tests monthly hazard helper logic |
| `tests/test_temporal_features.py` | Tests leakage-safe temporal features |
| `tests/test_feature_policy.py` | Tests feature normalization and encoding |
| `tests/test_model_splits.py` | Tests chronological and grouped assignments |
| `tests/test_economics_and_policy.py` | Tests economic and intervention formulas |
| `tests/test_workforce_stability.py` | Tests current planning calculations |
| `tests/test_portfolio_notebooks.py` | Tests the curated notebook sequence |
| `tests/test_readme_portfolio.py` | Tests README evidence and navigation contracts |
| `tests/test_ibm_attrition_benchmark.py` | Tests source, schema, feature, metric, and isolation contracts |
| `scripts/run_checkpoint49.ps1` | Runs the isolated Checkpoint 49 suite |
| `scripts/run_checkpoint53.ps1` | Runs the verified external benchmark and complete suite |
| `scripts/run_end_to_end.ps1` | Runs tests before the complete pipeline |
| `requirements.txt` | Adds pytest as a reproducible dependency |

## Limitations

Passing tests show that the covered implementation contracts behave as
specified. They do not prove:

- That synthetic assumptions are true for a real employer
- That interventions causally prevent departures
- That the model is fair in every deployment context
- That future workforce behavior will match expected values exactly
- That every possible software defect has been tested

Checkpoint 50 adds Ruff code-quality checks and GitHub Actions so these tests
run automatically on repository changes. See `docs/code_quality_and_ci.md`.
