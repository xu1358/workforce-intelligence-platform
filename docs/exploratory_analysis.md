# Workforce Exploratory Analysis

> **Historical Version 1 evidence only.** This report does not describe the
> temporal data used by the current model, policy, or dashboard. See the
> authoritative [Version 2 exploratory analysis](v2_exploratory_analysis.md)
> and [Notebook 37](../notebooks/37_v2_exploratory_analysis.ipynb).

## Purpose

This report summarizes exploratory analysis of the Version 1 synthetic
workforce and retention-modeling dataset. It measures the target balance,
missing-data patterns, univariate associations, and redundant features that
motivated the Version 2 temporal redesign.

The results are descriptive. They show how the synthetic data behaves; they
do not establish that any feature causes attrition.

## Results Snapshot

| Result | Value |
|---|---:|
| Full synthetic workforce | 10,000 employees |
| Retention-modeling population | 7,386 employees |
| Snapshot date | 2025-06-30 |
| Prediction end date | 2026-06-30 |
| Twelve-month attrition cases | 627 |
| Twelve-month non-attrition cases | 6,759 |
| Twelve-month attrition rate | 8.49% |
| Modeling columns | 39 |

The target is imbalanced: approximately 8.49% of modeling rows are positive
and 91.51% are negative. This is why later model evaluation emphasizes
PR-AUC, top-k precision, capture, and lift rather than accuracy alone.

## Missing-Data Findings

| Feature or feature group | Missing rows | Missing rate |
|---|---:|---:|
| Days since last change event | 5,542 | 75.03% |
| Review-related fields | 1,805 | 24.44% |
| Average training score | 124 | 1.68% |

The review-related fields include current performance rating, average
performance rating, goal completion, promotion recommendation, and days
since review. Rows without review history had a 15.07% attrition rate,
compared with 6.36% when review information was present.

That gap should not be read as evidence that a missing review causes
attrition. Missing review history is strongly connected to employee tenure
and data availability. Version 2 therefore adds explicit indicators such as
`no_prior_review` and uses only records available as of each snapshot.

## Numeric Associations

The largest standardized differences between attrition and non-attrition
rows included:

| Feature | No attrition mean | Attrition mean | Standardized difference |
|---|---:|---:|---:|
| Tenure, years | 2.29 | 1.64 | -0.51 |
| Compensation records | 2.82 | 2.18 | -0.49 |
| Salary growth | 9.68% | 6.22% | -0.46 |
| Review count | 1.44 | 0.96 | -0.45 |
| Base salary | $94,057 | $84,902 | -0.33 |
| Completed training hours | 27.45 | 22.13 | -0.33 |

These are single-feature comparisons. They do not adjust for department,
job level, tenure, or other correlated factors.

## Categorical Associations

The overall modeling attrition rate was 8.49%. Selected department results
were:

| Hire department | Employees | Attrition cases | Attrition rate | Lift over baseline |
|---|---:|---:|---:|---:|
| Manufacturing | 1,797 | 188 | 10.46% | 1.23 |
| Customer Support | 578 | 58 | 10.03% | 1.18 |
| Information Technology | 756 | 68 | 8.99% | 1.06 |
| Finance | 585 | 52 | 8.89% | 1.05 |
| Engineering | 1,508 | 106 | 7.03% | 0.83 |

These rates reflect the Version 1 synthetic generator and workforce
composition. They are not estimates for a real company.

## Redundancy Findings

The exploration exposed several pairs that contained the same or nearly the
same information:

| Feature pair | Correlation |
|---|---:|
| Promotion-compensation count and prior-promotion events | 1.000 |
| Days since compensation change and days since review | 1.000 |
| Tenure and compensation-record count | 0.976 |
| Initial salary and current base salary | 0.973 |
| Current and average performance rating | 0.962 |
| Completed training programs and hours | 0.928 |

Categorical redundancy was also substantial. For example, hire job family
and employment type had a corrected Cramér's V of 0.889, while hire
department and job family had 0.839.

These findings directly motivated Checkpoint 39's explicit feature policy,
reference-category encoding, rank checks, VIF limits, and redundant-feature
removal.

## Interpretation

The Version 1 exploration established four practical conclusions:

1. Accuracy would be misleading because attrition is uncommon.
2. Missingness carries information and must be modeled explicitly.
3. Many apparent attrition associations overlap with tenure and job
   structure.
4. Redundant variables must be removed before coefficient interpretation.

The later Version 2 model should be treated as the authoritative predictive
analysis. This report remains useful as an audit trail explaining why that
redesign was necessary.

## Limitations

- All employees and outcomes are synthetic.
- The analysis uses one Version 1 modeling snapshot.
- Univariate comparisons are not causal or fully adjusted estimates.
- Generator rules can create patterns that look stronger than real-world
  workforce relationships.
- Version 1 results must not be combined numerically with the later temporal
  final test.

## Evidence and Reproduction

- Executed analysis: [Notebook 18](../notebooks/18_workforce_eda.ipynb)
- Implementation: [run_workforce_eda.py](../src/run_workforce_eda.py)
- Generated evidence directory: `data/processed/eda/`

Reproduce the analysis from the project root:

```powershell
python src\run_workforce_eda.py
```
