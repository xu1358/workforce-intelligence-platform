# Current-State Dashboard Timeline and Safety

## Purpose

Checkpoint 47 corrects the retention dashboard timeline. The dashboard now
uses the current active workforce plan created by Checkpoint 46 instead of
presenting the legacy Version 1 modeling snapshot as a current risk list.

The dashboard is explicitly labeled **as of 2026-06-30**.

## Results Snapshot

| Current-state result | Value |
|---|---:|
| Active synthetic employees | 7,409 |
| Model-eligible employees with scores | 7,305 |
| Protected-level active employees excluded from scoring | 104 |
| Calibrated probability range | 1.74%–42.59% |
| Employees selected for human review | 700 |
| Planned intervention spend | $1,750,000 |
| Projected prevented departures | 35.03 |
| Projected avoided cost | $4,146,216 |
| Projected expected net value | $2,396,216 |
| Inactive employees displayed as current risks | 0 |

Every eligible active employee received exactly one current score. The
frozen policy preserved the tested 700-person limit, and the dashboard
contains no future outcome fields or direct personal names.

The financial and prevented-departure values are planning projections based
on calibrated probabilities and the synthetic reference cost scenario. They
are not observed 2026 outcomes or guaranteed intervention effects.

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

## Current visual evidence

The live Streamlit application contains six tabs: Overview, Workforce,
Workforce Stability, Recruiting, Retention Risk, and Model Performance.

Checkpoint 67 removes nine stale Version 1 screenshots that showed a
five-tab application, obsolete workforce and risk counts, an obsolete
ROC-AUC, and a classification threshold that is not the deployed policy.
Three of those files also contained the duplicate-download marker `(1)`.

Static screenshots are intentionally omitted from the current branch until
their creation can be automated from the same locked application and data
build. The current interface is documented in the
[Version 2 dashboard guide](dashboard.md), and the original images remain
available through the `v1.0-portfolio` Git tag.

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
.\scripts\checkpoints\run_checkpoint47.ps1
```

Use `-RebuildInputs` only when the Checkpoint 46 outputs need to be
regenerated:

```powershell
.\scripts\checkpoints\run_checkpoint47.ps1 -RebuildInputs
```

After the checkpoint succeeds, start the Streamlit dashboard with:

```powershell
streamlit run dashboard\app.py
```

## Evidence and Reproduction

- Executed validation:
  [Notebook 29](../notebooks/29_dashboard_current_state_validation.ipynb)
- Curated reviewer narrative:
  [Current Manufacturing Workforce Stability Plan](../notebooks/portfolio/04_current_workforce_stability_plan.ipynb)
- Data preparation:
  [build_dashboard_current_state.py](../src/build_dashboard_current_state.py)
- Dashboard application: [app.py](../dashboard/app.py)
- Primary generated evidence:
  `data/processed/dashboard/dashboard_metadata.csv`,
  `data/processed/dashboard/current_policy_summary.csv`, and
  `data/processed/dashboard_current_state_validation.csv`
