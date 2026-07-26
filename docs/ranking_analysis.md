# Retention Ranking Analysis

## Purpose

Checkpoint 43 evaluates whether the selected sigmoid-calibrated Logistic
Regression places higher-risk employees near the top of its validation
ranking.

The analysis uses the 2024 validation snapshot only. The 6,641 reserved
2025 test rows remain locked, and zero test targets are accessed.

This checkpoint answers:

> If the organization can review only part of the workforce, how many
> future attrition cases are concentrated near the top of the ranking?

It does not choose the final intervention threshold or claim that every
high-ranked employee will leave.

## Validation Population

| Item | Value |
|---|---:|
| Employees | 5,521 |
| Attrition cases | 586 |
| Attrition rate | 10.61% |
| Selected calibration | Sigmoid |
| PR-AUC | 0.1638 |
| ROC-AUC | 0.6211 |

The no-skill PR-AUC baseline is approximately the attrition rate, 0.1061.
The selected model's PR-AUC is approximately 1.54 times that baseline.

This represents useful but moderate ranking signal.

## Understanding Top-k

`Top-k` means selecting the highest-ranked portion of employees.

For example, top 10% means:

1. Sort employees from highest to lowest predicted risk.
2. Select the first 10%.
3. Compare that group with known validation outcomes.

The analysis evaluates twelve workforce fractions from 1% through 100%.

## Ranking Metrics

### Precision at k

Among the selected employees, what percentage experienced attrition?

`Captured attrition cases / Selected employees`

### Capture rate at k

Among all attrition cases, what percentage appears in the selected group?

`Captured attrition cases / All attrition cases`

This is the same idea as recall, applied to a ranked review budget.

### Lift at k

How concentrated is attrition in the selected group compared with random
selection?

`Selected-group attrition rate / Overall attrition rate`

A lift of 1 means no improvement over random selection.

## Main Top-k Results

| Workforce reviewed | Employees | Cases captured | Precision | Capture | Lift |
|---:|---:|---:|---:|---:|---:|
| 1% | 56 | 14 | 25.00% | 2.39% | 2.36 |
| 5% | 277 | 58 | 20.94% | 9.90% | 1.97 |
| 10% | 553 | 109 | 19.71% | 18.60% | 1.86 |
| 20% | 1,105 | 190 | 17.19% | 32.42% | 1.62 |
| 30% | 1,657 | 259 | 15.63% | 44.20% | 1.47 |
| 50% | 2,761 | 383 | 13.87% | 65.36% | 1.31 |
| 100% | 5,521 | 586 | 10.61% | 100.00% | 1.00 |

As the review budget grows:

- More attrition cases are captured.
- Precision and lift gradually move toward the population baseline.

This is expected. The highest-ranked group has the strongest
concentration, while progressively larger groups include lower-ranked
employees.

## Top-Decile Reporting Scenario

The top 10% scenario selects 553 employees.

| Outcome | Count |
|---|---:|
| True positives | 109 |
| False positives | 444 |
| False negatives | 477 |
| True negatives | 4,491 |

The corresponding score cutoff is approximately 0.1452.

That cutoff is an observed reporting boundary needed to select exactly
10% of this validation population. It is not a recommended intervention
threshold.

Checkpoint 46 will compare thresholds using explicit intervention costs
and expected value.

## Bootstrap Uncertainty

Checkpoint 43 performs 500 stratified validation bootstraps at all twelve
review fractions, producing 6,000 results.

For the top 10%:

| Metric | Observed | Lower 95% | Upper 95% |
|---|---:|---:|---:|
| Precision | 19.71% | 16.72% | 22.97% |
| Capture | 18.60% | 15.78% | 21.67% |

These intervals show that ranking performance varies across resampled
validation populations. Portfolio claims should report uncertainty rather
than only one point estimate.

## Precision-Recall Curve

The precision-recall curve shows the tradeoff between:

- Finding more attrition cases.
- Keeping the reviewed group concentrated.

It is especially useful because attrition is uncommon. Accuracy could
look high even if a model identifies no attrition cases.

## ROC Curve

The ROC curve compares true-positive and false-positive rates across score
thresholds.

The ROC-AUC of 0.6211 indicates modest separation. It is above random
ordering but far from perfect classification.

## Cumulative Gains Curve

The gains curve compares:

- Fraction of employees reviewed.
- Fraction of all attrition cases captured.

A useful ranking rises above the random-selection diagonal.

At 10% reviewed, the model captures approximately 18.6% of attrition
cases.

## Lift Curve

The lift curve compares concentration against random selection.

Lift is strongest at the smallest review fractions and approaches 1 when
the entire workforce is reviewed.

## Score Distribution

The distribution chart compares calibrated probabilities for:

- Employees without validation-period attrition.
- Employees with validation-period attrition.

The distributions overlap substantially. The model provides prioritization
signal, not certainty about individual outcomes.

## Generated Figures

The pipeline regenerates six figures in:

`data/processed/ranking_figures/`

- `precision_recall_curve.png`
- `roc_curve.png`
- `cumulative_gains_curve.png`
- `lift_curve.png`
- `score_distribution.png`
- `top_decile_confusion_matrix.png`

They are generated artifacts and are not committed as source files.

## Generated Tables

`retention_ranking_metrics.csv`

- Observed top-k metrics and bootstrap intervals

`retention_ranking_bootstrap.csv`

- Five hundred bootstrap iterations at twelve review fractions

`retention_ranking_summary.csv`

- One-row model, validation, and top-decile summary

`retention_curve_points.csv`

- Precision-recall, ROC, cumulative-gains, and lift points

`retention_top_decile_confusion_matrix.csv`

- Reporting-only top-decile outcome counts

`retention_ranking_validation.csv`

- Nineteen automated integrity and reconciliation checks

## Limitations and Responsible Use

The data is synthetic. These results do not describe real employees or
prove that the same ranking would work in production.

Attrition prediction is not a causal explanation. A high score must not be
interpreted as evidence of poor commitment or used for an automatic
employment action.

The ranking is most appropriate for aggregate workforce planning and
carefully reviewed, supportive retention outreach.
