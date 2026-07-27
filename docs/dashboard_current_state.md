# Current-State Dashboard Timeline and Safety

## Purpose

Checkpoint 47 corrects the retention dashboard timeline. The dashboard now
uses the current active workforce plan created by Checkpoint 46 instead of
presenting the legacy Version 1 modeling snapshot as a current risk list.

The dashboard is explicitly labeled **as of 2026-06-30**.

## Authoritative inputs

The current retention view reads:

- `current_retention_policy_scores.csv`
- `current_retention_policy_summary.csv`
- `current_retention_policy_group_summary.csv`
- `retention_policy_decision.csv`
- `retention_final_test_model_metrics.csv`
- `retention_calibration_selection.csv`
- `model_selection_decision_v2.csv`
- `employees.csv`
- `locations.csv`

Checkpoint 47 does not train, recalibrate, retune, or rescore the model. It
reconciles existing Checkpoint 46 outputs to workforce status and prepares
them for the dashboard.

## Current population rule

An employee is current only when all of the following are true at the as-of
date:

1. The hire date is on or before 2026-06-30.
2. The termination date is missing or later than 2026-06-30.
3. `employment_status` is `Active`.

Individual retention scoring is limited to the model-eligible levels:

- Individual Contributor
- Team Manager

Department Heads and Senior Managers remain part of workforce headcount but
are excluded from the individual review list. This preserves the Version 2
model-population policy.

## Dashboard meaning

The current view contains:

- 7,409 active synthetic employees.
- 7,305 model-eligible active employees with current scores.
- 104 protected-level active employees excluded from individual scoring.
- 700 employees selected by the frozen expected-value policy for human review.
- Zero terminated employees shown as current retention risks.

The dashboard uses two separate concepts:

- **Estimated 12-month attrition probability** is the calibrated model score.
- **Selected for human review** is the frozen Checkpoint 46 policy decision,
  which also considers replacement cost, intervention cost, expected
  effectiveness, the 700-person capacity limit, and the $1.75 million budget.

The probability bands are descriptive distribution bands. They are not
employment-action thresholds and they do not replace the selected policy.

## Model evidence versus current scoring

The performance metrics shown in the dashboard come from the once-only
out-of-time 2025 final test:

- PR-AUC: approximately 0.1710
- ROC-AUC: approximately 0.6061
- Brier score: approximately 0.1048

The current 2026 scores are different. The final model was refitted on all
observed historical snapshots and then applied to the active 2026 workforce.
Outcomes after 2026-06-30 are unknown, so current rows are planning
projections and are not a new performance test.

The UI also states that legacy Version 1 full-snapshot scores were in-sample
prioritization scores. Those scores are not used in the current workforce
view.

## Safety and governance

The dashboard enforces the following interpretation:

- All employees and outcomes are synthetic.
- The employee table is anonymized by employee ID.
- No current outcome or termination fields appear in the review table.
- Human review is required.
- Selection supports a retention conversation; it is not a causal finding.
- Automatic employment action is prohibited.

## Outputs

Checkpoint 47 writes the following dashboard files under
`data/processed/dashboard/`:

- `retention_risk_employees.csv`
- `retention_risk_summary.csv`
- `current_policy_summary.csv`
- `current_policy_by_department.csv`
- `model_performance.csv`
- `model_summary.csv`
- `dashboard_metadata.csv`

It also writes:

- `data/processed/dashboard_current_state_validation.csv`

Generated CSV outputs are reproducible and remain excluded from Git.

## Run

From the project root:

```powershell
.\scripts\run_checkpoint47.ps1
```

Use `-RebuildInputs` only when the Checkpoint 46 outputs need to be
regenerated:

```powershell
.\scripts\run_checkpoint47.ps1 -RebuildInputs
```

After the checkpoint succeeds, start the Streamlit dashboard with:

```powershell
streamlit run dashboard\app.py
```
