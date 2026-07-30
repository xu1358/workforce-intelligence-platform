# Version 2 Workforce Intelligence Dashboard

## Purpose

The Streamlit dashboard is the reviewer-facing interface for the current
Version 2 workforce, retention policy, model evidence, and manufacturing
stability plan.

The live application in [`dashboard/app.py`](../dashboard/app.py) is the
authoritative interface. It reads prepared tables from
`data/processed/dashboard/` and does not fit a model while a reviewer is using
the application.

## Results Snapshot

| Current Version 2 dashboard result | Value |
| --- | ---: |
| Active synthetic employees | 7,409 |
| Model-eligible employees with scores | 7,305 |
| Protected-level active employees excluded from individual scoring | 104 |
| Employees selected for supportive human review | 700 |
| Current planning date | 2026-06-30 |
| Once-only final-test ROC-AUC shown as model evidence | 0.6061 |
| Dashboard tabs | 6 |

The current employee scores are planning projections. Outcomes after
2026-06-30 are unknown, so the current workforce is not treated as another
test set.

## Six Live Tabs

The application source defines exactly six tabs in this order.

### Overview

The Overview tab summarizes current active headcount, 2025 turnover, open
requisitions, the 700-person human-review plan, applications, hires, the
current eligible population, and active locations. It compares department
headcount with the current review plan.

### Workforce

The Workforce tab shows active headcount by department and location with both
charts and aggregate tables.

### Workforce Stability

The Workforce Stability tab presents the manufacturing planning extension
that was missing from the old five-tab gallery. It reports the active
manufacturing workforce, expected departures, employees selected for review,
expected prevented departures, residual expected backfills, department-level
value, role-level backfill demand, vacancy days, training hours, and temporary
coverage hours.

These figures are scenario estimates, not observed operational outcomes.

### Recruiting

The Recruiting tab summarizes the application funnel and hire rate by source,
then reports job-requisition counts, target headcount, and average days open.

### Retention Risk

The Retention Risk tab contains only current, active, model-eligible synthetic
employees. Sidebar controls filter department, region, probability band, and
review status.

The page distinguishes:

- estimated 12-month attrition probability;
- descriptive probability bands; and
- selection by the frozen budget-constrained expected-value policy.

Selection means eligibility for supportive human review. It is not an
automatic employment action.

### Model Performance

The Model Performance tab displays the selected Logistic Regression with
sigmoid calibration, once-only out-of-time final-test metrics, and the frozen
planning-policy contract.

The displayed ROC-AUC is 0.6061, not the earlier Version 1 dashboard value.
The page does not present a 0.50 classification cutoff as the deployed
retention policy.

## Screenshot Freshness Policy

Static screenshots are intentionally not committed on the current branch.

The previous nine files under `docs/screenshots/` were captured from the
five-tab Version 1 application. They showed obsolete headcount, risk,
performance, and threshold figures, omitted Workforce Stability, and included
three duplicate-download-style filenames containing `(1)`.

Keeping those images beside Version 2 documentation created two competing
versions of the dashboard. Checkpoint 67 removes all nine files and their
Markdown embeds.

This is a deliberate freshness control:

- the live Version 2 application is the dashboard source of truth;
- the current numbers remain traceable to generated CSV evidence;
- automated validation checks the exact six tab labels;
- automated validation rejects any image left in `docs/screenshots/`;
- automated validation rejects Markdown links to that directory; and
- the original images remain recoverable from the `v1.0-portfolio` Git tag.

A future screenshot gallery should be committed only with an automated capture
and refresh process tied to the same locked dashboard build.

## Run the Dashboard

Create and activate the locked project environment, build the project outputs,
and start Streamlit:

```bash
python scripts/run_project.py pipeline
python scripts/run_project.py dashboard
```

For a previously built project, only the second command is necessary.
Streamlit normally opens `http://localhost:8501`.

## Interpretation Boundaries

- Every employee, event, model result, and financial value is synthetic.
- Current probabilities are planning projections, not known future outcomes.
- Final-test metrics describe the reserved historical test period.
- Expected prevented departures and net value depend on assumed intervention
  effectiveness.
- Human review is required.
- Automatic employment action is prohibited.

## Evidence and Reproduction

- Current-state timeline and safety:
  [`dashboard_current_state.md`](dashboard_current_state.md)
- Manufacturing planning:
  [`manufacturing_workforce_stability.md`](manufacturing_workforce_stability.md)
- Dashboard application:
  [`dashboard/app.py`](../dashboard/app.py)
- Current data preparation:
  [`build_dashboard_current_state.py`](../src/build_dashboard_current_state.py)
- Executed dashboard validation:
  [`29_dashboard_current_state_validation.ipynb`](../notebooks/29_dashboard_current_state_validation.ipynb)
- Executed current stability plan:
  [`04_current_workforce_stability_plan.ipynb`](../notebooks/portfolio/04_current_workforce_stability_plan.ipynb)
- Screenshot and tab validator:
  [`validate_dashboard_evidence.py`](../src/validate_dashboard_evidence.py)
- Reproduce Checkpoint 67:

```powershell
.\scripts\checkpoints\run_checkpoint67.ps1
```
