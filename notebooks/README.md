# Notebook Portfolio Guide

The project contains two notebook layers:

1. **Portfolio sequence** — four executed notebooks that tell the complete
   Version 2 analytical and business story.
2. **Supporting checkpoint notebooks** — the original detailed notebooks
   `01` through `33`, retained as technical evidence and an audit trail.
3. **External benchmark** — Notebook `31`, which evaluates the same
   methodological discipline on a separate fictional IBM dataset.
4. **Explanation extension** — Notebook `32`, which documents aggregate
   current-model explanations and grouped-refit stability.

## Recommended reviewer sequence

| Order | Notebook | Main question | Approximate review time |
| --- | --- | --- | --- |
| 1 | [Data Foundation and Temporal Design](portfolio/01_data_foundation_and_temporal_design.ipynb) | Is the synthetic data valid, temporally structured, and leakage controlled? | 6–8 minutes |
| 2 | [Model Development and Final Evaluation](portfolio/02_model_development_and_final_evaluation.ipynb) | Which model was selected, how well are probabilities calibrated, and how did it perform out of time? | 7–9 minutes |
| 3 | [Fairness, Economics, and Retention Policy](portfolio/03_fairness_economics_and_policy.ipynb) | How are subgroup risk, uncertain costs, capacity, and governance incorporated? | 7–9 minutes |
| 4 | [Current Manufacturing Workforce Stability Plan](portfolio/04_current_workforce_stability_plan.ipynb) | How does the frozen policy support current backfill and workforce planning? | 6–8 minutes |

The sequence is designed to be read in order. Each notebook contains embedded
outputs, previous/next navigation, a synthetic-data notice, and a concise
interpretation section.

## What the portfolio notebooks contain

- Aggregate tables and figures only
- Executed outputs with no error cells
- Historical validation and final-test evidence
- Current planning outputs kept separate from known historical outcomes
- Explicit uncertainty, fairness, and human-review limitations

They do not expose employee-level review lists or claim that synthetic
relationships are causal.

## Supporting technical notebooks

The original notebooks in this directory remain useful when a reviewer wants
implementation detail:

- `01`–`10`: generated data and relationship validation
- `11`–`16`: Version 1 SQL, modeling, interpretation, and dashboard validation
- `17`–`19`: generator audit, EDA, and historical-assignment analysis
- `20`–`33`: Version 2 checkpoint-specific analytical validation and extensions

These supporting notebooks are not the recommended first reading path.

The reviewer-visible Version 2 notebooks `20`–`30` are committed with saved
tables and figures. Their execution state, error outputs, portability, and file
sizes are enforced by the portfolio validator and automated tests. Regenerate
them after refreshing the processed pipeline artifacts with:

```powershell
.\scripts\execute_supporting_notebooks.ps1
```

## Separate external benchmark

[Notebook 31: IBM HR Analytics External Benchmark](31_ibm_external_benchmark.ipynb)
documents the pinned source, feature exclusions, repeated out-of-fold
evaluation, calibration, ranking, coefficient stability, and descriptive
subgroup checks.

It is intentionally separate from the four-notebook primary story because the
IBM table has no dates or stated outcome horizon. Its metrics are not directly
comparable with the primary model's once-only out-of-time final test. It does
not change the primary fitted model, frozen policy, dashboard, or current
review plan.

## Separate explanation extension

[Notebook 32: Retention Explanation Stability](32_retention_explanation_stability.ipynb)
documents exact aggregate linear-SHAP contributions, five employee-grouped
stability refits, probability-quartile patterns, and interpretation limits.

It is separate from the four-notebook primary story because it adds diagnostic
transparency after the model and policy decisions are already frozen. It does
not save employee-level explanations, reopen the final test, or alter the
current human-review plan.

## Separate survival-analysis extension

[Notebook 33: Employee Survival and Time-to-Exit Analysis](33_survival_analysis.ipynb)
documents right-censoring, overall and subgroup Kaplan–Meier estimates,
baseline-at-hire Cox hazard ratios, five-fold held-out concordance, and the
reported proportional-hazards caveat.

It remains separate from the four-notebook primary story because it answers
how retention changes with tenure rather than whether an active employee will
leave in the next 12 months. It does not replace the classifier, reopen the
final test, change the frozen policy, or export employee-level survival rows.

Run the survival extension with:

```powershell
.\scripts\run_checkpoint55.ps1
```

## Reproducibility

The embedded outputs were produced from the deterministic Version 2 pipeline.
To regenerate the underlying CSV artifacts, run the relevant checkpoint
scripts or the complete end-to-end pipeline before opening the notebooks.

Validate the committed portfolio sequence with:

```powershell
python src\validate_portfolio_notebooks.py
```

The validator checks the manifest, titles, headings, execution state, error
outputs, navigation, aggregate-only presentation, synthetic-data notices, and
file sizes.

Run the external benchmark and reproduce Notebook 31's aggregate source data
with:

```powershell
.\scripts\run_checkpoint53.ps1
```

Run the primary explanation and stability extension with:

```powershell
.\scripts\run_checkpoint54.ps1
```
