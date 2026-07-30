# Version 2 Exploratory Analysis

## Purpose

This is the authoritative exploratory analysis for the Version 2 temporal
retention project. It replaces Version 1 numbers as evidence for current model,
calibration, fairness, economics, policy, and dashboard results.

The analysis is descriptive. It does not establish causal relationships.
Outcome comparisons use the training snapshot only. Validation and final-test
outcomes remain outside exploratory analysis, and current future outcomes are
unknown.

## Results Snapshot

| Population | Rows | Unique employees | Outcome use |
| --- | ---: | ---: | --- |
| Historical temporal panel | 16,673 | 7,745 | Training outcomes only for EDA |
| 2023 training snapshot | 4,303 | 4,303 | Descriptive outcome relationships |
| 2024 validation snapshot | 5,625 | 5,625 | Structure and availability only |
| 2025 final-test snapshot | 6,745 | 6,745 | Structure and availability only |
| Current active scoring population | 7,409 | 7,409 | Future outcome unknown |

The historical row count is larger than the unique-employee count because an
employee can appear at multiple snapshot dates. Those repeated observations are
intentional temporal panel structure, not duplicate records.

### Performance-history availability

In the historical temporal panel, 7,226 of 16,673 rows, or **43.34%**, have no
prior performance review. The rate declines as history accumulates:

| Snapshot | Rows | No prior review | Missing-review rate |
| --- | ---: | ---: | ---: |
| 2023 training | 4,303 | 2,916 | 67.77% |
| 2024 validation | 5,625 | 2,489 | 44.25% |
| 2025 final test | 6,745 | 1,821 | 27.00% |
| 2026 current scoring | 7,409 | 1,475 | 19.91% |

Version 2 represents this availability explicitly through `no_prior_review`.
Missing review fields are not silently treated as average performance.

### The Version 1 missingness result does not transfer

| Dataset | No-review attrition | Review-present attrition | Gap |
| --- | ---: | ---: | ---: |
| Version 1 single snapshot | 15.07% | 6.36% | +8.71 points |
| Version 2 training snapshot | 9.81% | 9.23% | +0.58 points |

The strong Version 1 association is not a Version 2 finding. In the Version 2
training snapshot, the difference is much smaller: 286 outcomes among 2,916
rows with no prior review versus 128 among 1,387 rows with prior review
history.

This comparison does not mean missingness is irrelevant. It means its meaning
depends on the temporal data-generating process and must be measured on the
dataset being modeled.

### Training-period descriptive patterns

Training-period group summaries show:

- Customer Support has the highest department attrition rate at **16.62%**
  across 325 rows.
- Supply Chain has the lowest department rate at **6.40%** across 531 rows.
- Hourly employees have a **14.48%** rate across 670 rows, compared with
  **8.73%** across 3,633 salaried rows.
- Job-level rates decline from **14.21%** at level 1 to **5.70%** at level 4.

These are synthetic, unadjusted group comparisons. Department, employment type,
job level, compensation, tenure, and review availability are related to one
another, so the figures should not be interpreted as causal effects.

## What Changed From Version 1

Notebook 18 remains as an explicitly historical Version 1 artifact. Its
7,386-row population, 8.49% attrition rate, and 24.44% review missingness do not
describe the Version 2 temporal model.

Version 2 adds:

- three chronological historical snapshots;
- repeated employees across time;
- an explicit current scoring population with unknown future outcomes;
- as-of feature construction;
- explicit missing-history flags; and
- training-only exploratory outcome analysis.

The later period outcome rates are reported only after evaluation in the
[temporal prior-shift audit](temporal_prior_shift.md). They were not used to
choose exploratory hypotheses, features, models, or calibration.

## Interpretation

The most important exploratory conclusion is not that missing performance
history predicts attrition. It is that data availability changes materially
with snapshot age. Review missingness falls from 67.77% to 27.00% across the
historical panel and to 19.91% in current scoring.

That temporal pattern supports:

1. keeping `no_prior_review` as an explicit feature state;
2. evaluating model and calibration behavior out of time;
3. monitoring feature availability after deployment; and
4. avoiding claims based on the discarded Version 1 snapshot.

## Governance Boundary

Checkpoint 66 does not:

- inspect validation or final-test outcomes for exploratory relationships;
- modify the generated workforce;
- change model features, coefficients, or calibration;
- reopen model selection;
- change policy selections or dashboard probabilities; or
- claim that descriptive differences are causal.

## Evidence and Reproduction

Run the complete checkpoint:

```powershell
.\scripts\checkpoints\run_checkpoint66.ps1
```

Run only the Version 2 EDA:

```powershell
python src\analyze_v2_exploratory_data.py
```

Reviewer-facing evidence:

- [Notebook 37: Version 2 Exploratory Analysis](../notebooks/37_v2_exploratory_analysis.ipynb)
- configuration: `config/v2_exploratory_analysis.yaml`
- implementation: `src/analyze_v2_exploratory_data.py`
- aggregate outputs: `data/processed/v2_exploratory_analysis/`

Historical context:

- [Version 1 exploratory report](exploratory_analysis.md)
- [Version 1 Notebook 18](../notebooks/18_workforce_eda.ipynb)
