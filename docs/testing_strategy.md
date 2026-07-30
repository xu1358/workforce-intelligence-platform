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
- Complete materialization of all six simulation controls.
- Feature-lag and first-month-risk branch behavior.
- Cause-model dispatch and unsupported-model rejection.
- Mandatory administrative censoring at the as-of date.

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

### 9. Model-Explanation Contracts

Checkpoint 54 adds tests that verify:

- The explanation policy keeps Logistic Regression and sigmoid calibration
  fixed.
- Explanations use the base-model log-odds scale.
- Employee-level explanation export and automatic action remain prohibited.
- Exact linear SHAP values reconstruct the fitted decision function.
- The historical background centers mean feature contributions.
- Encoded one-hot contributions group back to the configured raw feature.
- Importance ordering and top-k comparisons use deterministic tie-breaking.
- Spearman rank correlation and Jaccard overlap helpers reconcile.
- Aggregate importance shares sum to one.
- Probability-quartile summaries contain no employee identifiers or targets.

The unit tests fit only a tiny in-memory linear classifier. They do not refit
the full current workforce model or load processed employee-level data.

### 10. Survival-Analysis Contracts

Checkpoint 55 adds tests that verify:

- The survival extension stays isolated from the classifier, final test,
  policy, dashboard, and automatic action.
- Hire-event notes reconstruct original department, location, and role IDs.
- Malformed hire notes fail instead of falling back to current organization
  fields.
- Known terminations become events and active employees are right-censored.
- Inclusive tenure durations remain positive and reconcile to calendar dates.
- Salary position uses the original hire salary and hire-role band.
- Kaplan–Meier survival stays bounded and nonincreasing.
- Configured horizon summaries remain ordered and bounded.
- Cox reference categories are omitted and numeric features are standardized.

The tests use three small in-memory employee histories. They do not read the
generated 10,000-employee cohort or fit the full survival model.

### 11. Policy Allocation-Equity Contracts

Checkpoint 56 adds tests that verify:

- Direct salary bands and within-job-level salary quintiles are configured.
- Equal probabilities mechanically produce higher expected value when
  replacement cost is salary based.
- Ranking ties use employee ID deterministically.
- Every sensitivity policy respects the same capacity and budget.
- Salary-dependent and salary-neutral rankings can produce different
  selections.
- Every employee receives absolute, relative, within-level, and risk groups.
- Outcome access, post-test retuning, and automatic action remain prohibited.
- Notebook 34 contains executed aggregate outputs.
- The policy-equity document reports the committed salary findings.

The unit tests use a five-row in-memory population. They do not load employee
outcomes or recalculate the complete current allocation.

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
python -m pip install "pip==26.0.1" "setuptools==83.0.0" "wheel==0.47.0"
python -m pip install --no-build-isolation --require-hashes -r requirements-lock.txt
```

Then run:

```powershell
.\scripts\checkpoints\run_checkpoint49.ps1
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

`scripts/run_project.py pipeline` now compiles the test directory and runs the
test suite before expensive data generation or model work. The Windows
PowerShell wrapper delegates to that same command.

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

The presentation, external-source, explanation, survival, and allocation-equity
contract tests do not recalculate the complete primary pipeline; they make sure
the committed Version 2 story and its safeguards remain reproducible.

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
| `tests/test_model_explanations.py` | Tests exact linear explanations, grouping, stability, and governance |
| `tests/test_survival_analysis.py` | Tests censoring, hire reconstruction, Kaplan–Meier behavior, Cox encoding, and isolation |
| `tests/test_retention_policy_equity.py` | Tests salary mechanisms, pay groups, sensitivity policies, and governance |
| `tests/test_dependency_reproducibility.py` | Tests direct pins, transitive hashes, fingerprints, CI, and update documentation |
| `tests/test_markdown_integrity.py` | Tests CommonMark fence balance and Mermaid ER rendering |
| `tests/test_cross_platform_runner.py` | Tests portable commands, stage parity, archive layout, and wrapper delegation |
| `tests/test_hazard_configuration_contract.py` | Tests complete simulation-key wiring, behavior, and governance |
| `tests/test_generator_signal_governance.py` | Tests generator/model separation, chronology, public evidence, and private-note exclusion |
| `tests/test_v2_postgresql_integration.py` | Tests V2 analytical views, least-privilege projections, type normalization, pipeline order, and CSV fallback |
| `scripts/checkpoints/run_checkpoint49.ps1` | Runs the isolated Checkpoint 49 suite |
| `scripts/checkpoints/run_checkpoint53.ps1` | Runs the verified external benchmark and complete suite |
| `scripts/checkpoints/run_checkpoint54.ps1` | Runs aggregate explanation and grouped stability validation |
| `scripts/checkpoints/run_checkpoint55.ps1` | Runs censoring-aware survival analysis and complete validation |
| `scripts/checkpoints/run_checkpoint56.ps1` | Runs the post-policy salary-allocation equity audit |
| `scripts/checkpoints/run_checkpoint57.ps1` | Validates the complete locked dependency environment |
| `scripts/checkpoints/run_checkpoint58.ps1` | Validates reviewer-facing Markdown rendering |
| `scripts/run_project.py` | Provides the canonical cross-platform execution interface |
| `src/validate_cross_platform_runner.py` | Saves aggregate runner and repository-layout validation |
| `scripts/checkpoints/run_checkpoint59.ps1` | Applies and validates the runner reorganization on Windows |
| `scripts/checkpoints/run_checkpoint60.ps1` | Validates executable hazard settings and the complete quality suite |
| `scripts/checkpoints/run_checkpoint61.ps1` | Validates generator-signal governance and the complete quality suite |
| `scripts/checkpoints/run_checkpoint62.ps1` | Builds Version 2 from PostgreSQL, validates source/output parity, and runs the complete quality suite |
| `scripts/run_end_to_end.ps1` | Maps Windows switches to the canonical Python pipeline |
| `requirements.txt` | Pins every direct dependency |
| `requirements-lock.txt` | Pins and hashes every resolved distribution |

## Dependency Reproducibility Contracts

Checkpoint 57 makes the test environment itself testable. The suite verifies:

- Python 3.12 and the exact GitHub Actions interpreter;
- the exact pip bootstrap version;
- 19 exact direct dependency pins;
- 113 exact direct and transitive distributions;
- SHA-256 hashes for every locked distribution;
- SHA-256 fingerprints for both dependency files;
- agreement between direct pins and the generated lock;
- locked installation and `pip check` in GitHub Actions; and
- fail-fast validation in the end-to-end pipeline.

Run the complete local environment contract with:

```powershell
.\scripts\checkpoints\run_checkpoint57.ps1
```

## Markdown Rendering Contracts

Checkpoint 58 audits every reviewer-facing Markdown file under the root,
`docs/`, and the notebook index. It detects unfinished backtick or tilde
fences using CommonMark-compatible closing rules.

The suite also checks that `docs/data_model.md` contains one closed Mermaid
`erDiagram` with the committed core relationships. GitHub Actions and the
end-to-end runner execute the same validation.

Run the complete local rendering contract with:

```powershell
.\scripts\checkpoints\run_checkpoint58.ps1
```

## Cross-Platform Runner Contracts

Checkpoint 59 verifies that the finished project exposes one Python command
surface on Windows, macOS, and Linux. Tests compare the default pipeline stage
sequence with the committed Version 2 workflow, exercise skip-option
composition, reject shell-specific command construction, validate portable
notebook execution, and require checkpoint journals to remain archived outside
the scripts root.

Run the primary engineering contract with:

```bash
python scripts/run_project.py quality
```

## Attrition-Hazard Configuration Contracts

Checkpoint 60 treats the `simulation` mapping in
`config/attrition_hazard_config.yaml` as executable input. Tests require all
six keys, materialize them as typed settings, exercise alternative lag and
hire-month branches, reject unknown cause models, and enforce the current
administrative-censoring boundary.

The committed values reproduce the prior hard-coded semantics, so these
checks prevent silent configuration drift without retuning the synthetic
generator.

## Generator-Signal Governance Contracts

Checkpoint 61 renames the old model-sounding ROC-AUC target as a
development-only generator diagnostic. Tests verify its lower and upper
bounds, disclose both the initial `0.54` and accepted `0.6298` diagnostics,
and require the later `0.6061` final-test result to remain evidence of no
post-test retuning.

The public methodology is tested for essential chronology and the absence of
private first-person response markers.

## Version 2 PostgreSQL Integration Contracts

Checkpoint 62 moves PostgreSQL ahead of the temporal builder in the default
pipeline. Tests require eight explicit analytical views, exclude direct names
and unused identifiers from the query boundary, normalize database and CSV
types identically, and ensure `--skip-postgres` is the only route to the CSV
fallback.

The live integration validator compares row counts, column contracts, primary
keys, and canonical content hashes across all eight sources. It then requires
the PostgreSQL-built historical and current temporal files to match the frozen
row counts, widths, and SHA-256 fingerprints before downstream analysis can
continue.

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
