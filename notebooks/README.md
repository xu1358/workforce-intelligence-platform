# Notebook Portfolio Guide

The project contains two notebook layers:

1. **Portfolio sequence** — four executed notebooks that tell the complete
   Version 2 analytical and business story.
2. **Supporting checkpoint notebooks** — the original detailed notebooks
   `01` through `30`, retained as technical evidence and an audit trail.

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
- `20`–`30`: Version 2 checkpoint-specific analytical validation

These supporting notebooks are not the recommended first reading path.

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
