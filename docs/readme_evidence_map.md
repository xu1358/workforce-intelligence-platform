# README Evidence Map

## Purpose

The main README is a portfolio summary, not a second source of analytical
truth. This map ties its headline statements to the scripts, generated
artifacts, documentation, and curated notebooks that support them.

All values describe the deterministic synthetic Version 2 run. They are not
measurements from a real employer.

## Headline Claim Traceability

| README claim | Authoritative generated artifact | Producing checkpoint | Reviewer-facing evidence |
| --- | --- | ---: | --- |
| 7,409 current active employees and 7,305 model-eligible employees | `data/processed/dashboard/dashboard_metadata.csv` | 47 | Notebook 4 and `docs/dashboard_current_state.md` |
| 2023 train, 2024 validation, and reserved 2025 final-test populations | `data/processed/model_split_summary.csv` | 40 | Notebooks 1–2 and `docs/model_validation_strategy.md` |
| Generator diagnostic 0.6298 was development-only; final-test ROC-AUC 0.6061 was retained without retuning | `data/processed/v2_validation/diagnostic_model_summary.csv` and `data/processed/retention_final_test_model_metrics.csv` | 37 and 46 | `docs/synthetic_signal_governance.md` |
| Primary Version 2 temporal data is queried from PostgreSQL with exact CSV and frozen-output parity | `data/processed/v2_postgresql_integration/validation_checks.csv` and `source_table_parity.csv` | 62 | `docs/v2_postgresql_integration.md` |
| Independent SQL and Python temporal implementations agree across 24,082 rows and 52 columns | `data/processed/sql_temporal_equivalence/validation_checks.csv` and `column_equivalence.csv` | 63 | `docs/sql_python_temporal_equivalence.md` |
| Logistic Regression selected on validation | `data/processed/model_selection_decision_v2.csv` | 41 | Notebook 2 and `docs/model_comparison_v2.md` |
| Sigmoid calibration selected | `data/processed/retention_calibration_selection.csv` | 42 | Notebook 2 and `docs/calibration_analysis.md` |
| Final-test PR-AUC 0.1710, ROC-AUC 0.6061, and Brier 0.1048 | `data/processed/retention_final_test_model_metrics.csv` | 46 | Notebook 2 and `docs/retention_policy_analysis.md` |
| Top-decile precision 19.85%, capture 16.44%, and lift 1.64 | `data/processed/retention_final_test_model_metrics.csv` | 46 | Notebook 2 |
| Frozen 700-person, $1.75 million expected-value policy | `data/processed/retention_policy_decision.csv` | 46 | Notebook 3 and `docs/retention_policy_analysis.md` |
| Final-test policy capture 17.06% and $2.03 million outcome-aligned net value | `data/processed/retention_policy_decision.csv` | 46 | Notebook 3 |
| Current projection of 35.0 prevented departures and $2.40 million expected net value | `data/processed/current_retention_policy_summary.csv` | 46 | Notebooks 3–4 |
| Current selected mean salary $125,670 versus $95,722 eligible and 25.83 highest-to-lowest pay-quintile selection ratio | `data/processed/policy_equity/policy_equity_population_summary.csv` | 56 | Notebook 34 and `docs/retention_policy_equity.md` |
| Manufacturing expected departures 229.7 and residual backfills 221.3 | `data/processed/manufacturing_stability_summary.csv` | 48 | Notebook 4 and `docs/manufacturing_workforce_stability.md` |
| IBM benchmark contains 1,470 rows and 237 positive cases | `data/processed/ibm_benchmark/ibm_benchmark_data_profile.csv` | 53 | Notebook 31 and `docs/ibm_external_benchmark.md` |
| IBM repeated out-of-fold PR-AUC 0.6094, ROC-AUC 0.8277, and Brier 0.0975 | `data/processed/ibm_benchmark/ibm_benchmark_metrics.csv` | 53 | Notebook 31 and `docs/ibm_external_benchmark.md` |
| IBM top-decile precision 69.39%, capture 43.04%, and lift 4.30 | `data/processed/ibm_benchmark/ibm_benchmark_metrics.csv` | 53 | Notebook 31 |
| Explanation minimum rank correlation 0.8857 and median 0.9299 | `data/processed/retention_explanations/retention_explanation_pairwise_stability.csv` | 54 | Notebook 32 and `docs/model_explanations_and_stability.md` |
| Explanation minimum top-10 overlap 0.6667 | `data/processed/retention_explanations/retention_explanation_pairwise_stability.csv` | 54 | Notebook 32 |
| Kaplan–Meier 12-month retention 89.53% and 60-month retention 57.51% | `data/processed/survival_analysis/survival_horizon_summary.csv` | 55 | Notebook 33 and `docs/survival_analysis.md` |
| Cox five-fold mean concordance 0.5738 | `data/processed/survival_analysis/cox_cross_validation.csv` | 55 | Notebook 33 and `docs/survival_analysis.md` |
| 19 direct pins and 113 hashed locked distributions | `data/processed/dependency_environment/dependency_reproducibility_validation.csv` | 57 | `docs/dependency_reproducibility.md` |
| Automated test count | pytest collection through `scripts/run_project.py quality` | 59 | GitHub Actions and local terminal output |

Generated CSV files are intentionally excluded from Git. The executed curated
notebooks preserve aggregate evidence so GitHub reviewers can inspect results
without downloading employee-level synthetic records.

## Interpretation Boundaries

### Historical model evidence

The final model metrics come from the reserved 2025 snapshot predicting through
2026-06-30. The target was accessed once after the model, calibration, economic
assumptions, and intervention policy were frozen.

The earlier `0.62–0.75` ROC-AUC range belongs only to the synthetic-generator
acceptance stage. The later final-test ROC-AUC of `0.6061` was not compared
with that band for retuning or acceptance.

### Current planning projections

The 2026 current plan uses a model refitted on all observable historical
snapshots. Outcomes after 2026-06-30 are unknown. Expected departures,
prevented departures, backfills, avoided cost, and net value are projections,
not a new model test.

### Economic values

The reference policy assumes:

- replacement impact equal to 100% of base salary;
- intervention cost of $2,500 per reviewed employee; and
- 25% probability that an intervention prevents a departure.

The project evaluates 27 scenarios around these inputs. No intervention was
performed, so effectiveness and savings are not causal estimates.

### Governance

Selection means eligibility for supportive human review. It does not mean an
employee is known to leave, and it must not trigger an automatic employment
action. Financial value is a scenario-planning quantity, not a measure of a
person's value.

### Allocation-equity evidence

Checkpoint 56 evaluates the policy decision after salary-based economics are
applied. It reports absolute salary bands, population and within-job-level
salary quintiles, salary selection within risk quintiles, department salary
concentration, and three alternative ranking sensitivities.

The audit reads no target or termination columns and does not change the
frozen policy. Salary-neutral and salary-capped alternatives are diagnostic;
adopting a replacement policy requires a new holdout or prospective
evaluation.

### Deployed-policy fairness evidence

Checkpoint 44's global top-10% cutoff is retained only as a model-ranking
diagnostic. Checkpoint 64 imports the exact frozen `policy_flags()` function
from Checkpoint 46 and audits the 700-person expected-value allocation.

Only 184 validation employees overlap the probability proxy and deployed
policy. The corrected policy rates include Customer Support at 1.25% versus
Engineering at 24.02%, Hourly at 0.67% versus Salaried at 14.98%, and job
level 1 at 0.63% versus level 3 at 19.21%. The audit reads validation outcomes
only, exports aggregate evidence, and leaves the final test and policy frozen.

### External benchmark

The IBM HR Analytics data is a second fictional dataset with no event dates or
stated target horizon. Its repeated stratified cross-validation metrics are
methodological evidence only. They are not directly comparable with the
primary model's once-only future-time test and do not validate the primary
model on another employer.

The source commit, checksum, schema, license, and isolation rules are committed
in `config/ibm_attrition_benchmark.yaml`. No IBM row-level probability,
ranking, or review list is saved.

### Explanation evidence

Checkpoint 54 explains the refitted current Logistic Regression on its native
log-odds scale. The closed-form linear SHAP values exactly reconstruct the
model decision function and are aggregated from encoded columns to 21 raw
features.

Five employee-grouped refits measure importance-rank correlation, top-10
feature overlap, and highest-probability-quartile direction stability. Only
aggregate tables and figures are saved. The evidence is noncausal, does not
reopen the final test, and does not change the frozen policy.

### Survival evidence

Checkpoint 55 uses one record per synthetic employee from hire until a known
termination or right-censoring at 2026-06-30. Kaplan–Meier estimates therefore
use active employees' observed tenure without inventing future exits.

The Cox model uses only reconstructed baseline-at-hire fields. Its hazard
ratios are adjusted synthetic associations, not causal effects. Five-fold
concordance is a robustness diagnostic for this separate extension and is not
directly comparable with the primary classifier's PR-AUC or ROC-AUC. No
employee-level survival rows are saved, and the primary model, final test,
policy, dashboard, and review list remain unchanged.

## Validation Contract

`config/readme_portfolio.yaml` records:

- required README sections;
- the displayed forms of headline metrics;
- required local documentation and notebook links;
- required reproduction commands; and
- stale Version 1 statements that must not return.

Run the contract directly with:

```powershell
python src\validate_readme_portfolio.py
```

Checkpoint 53 combines source verification, the isolated benchmark,
compilation, Ruff, formatting, and the complete pytest suite:

```powershell
.\scripts\checkpoints\run_checkpoint53.ps1
```

Checkpoint 54 reproduces aggregate explanations, grouped stability evidence,
and the complete pytest suite:

```powershell
.\scripts\checkpoints\run_checkpoint54.ps1
```

Checkpoint 55 reproduces the censoring-aware survival extension and complete
pytest suite:

```powershell
.\scripts\checkpoints\run_checkpoint55.ps1
```

Checkpoint 56 reproduces the salary-allocation equity audit and complete
pytest suite:

```powershell
.\scripts\checkpoints\run_checkpoint56.ps1
```

Checkpoint 57 verifies the pinned Python, pip, direct dependencies, transitive
lock, package hashes, fingerprints, installed versions, CI contract, and
complete pytest suite:

```powershell
.\scripts\checkpoints\run_checkpoint57.ps1
```

Checkpoint 64 reproduces the exact deployed-policy subgroup audit, executes
Notebook 35, and runs the complete quality suite:

```powershell
.\scripts\checkpoints\run_checkpoint64.ps1
```
