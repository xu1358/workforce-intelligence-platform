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
| Logistic Regression selected on validation | `data/processed/model_selection_decision_v2.csv` | 41 | Notebook 2 and `docs/model_comparison_v2.md` |
| Sigmoid calibration selected | `data/processed/retention_calibration_selection.csv` | 42 | Notebook 2 and `docs/calibration_analysis.md` |
| Final-test PR-AUC 0.1710, ROC-AUC 0.6061, and Brier 0.1048 | `data/processed/retention_final_test_model_metrics.csv` | 46 | Notebook 2 and `docs/retention_policy_analysis.md` |
| Top-decile precision 19.85%, capture 16.44%, and lift 1.64 | `data/processed/retention_final_test_model_metrics.csv` | 46 | Notebook 2 |
| Frozen 700-person, $1.75 million expected-value policy | `data/processed/retention_policy_decision.csv` | 46 | Notebook 3 and `docs/retention_policy_analysis.md` |
| Final-test policy capture 17.06% and $2.03 million outcome-aligned net value | `data/processed/retention_policy_decision.csv` | 46 | Notebook 3 |
| Current projection of 35.0 prevented departures and $2.40 million expected net value | `data/processed/current_retention_policy_summary.csv` | 46 | Notebooks 3–4 |
| Manufacturing expected departures 229.7 and residual backfills 221.3 | `data/processed/manufacturing_stability_summary.csv` | 48 | Notebook 4 and `docs/manufacturing_workforce_stability.md` |
| Automated test count | pytest collection during `scripts/run_checkpoint52.ps1` | 52 | GitHub Actions and local terminal output |

Generated CSV files are intentionally excluded from Git. The executed curated
notebooks preserve aggregate evidence so GitHub reviewers can inspect results
without downloading employee-level synthetic records.

## Interpretation Boundaries

### Historical model evidence

The final model metrics come from the reserved 2025 snapshot predicting through
2026-06-30. The target was accessed once after the model, calibration, economic
assumptions, and intervention policy were frozen.

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

Checkpoint 52 combines that validation with compilation, Ruff, formatting,
and the complete pytest suite:

```powershell
.\scripts\run_checkpoint52.ps1
```
