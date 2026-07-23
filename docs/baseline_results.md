# Workforce Intelligence Platform — Version 1 Baseline Results

## Baseline Identification

This document records the completed Version 1 project before the Version 2 revisions begin.

| Item | Value |
|---|---|
| Baseline Git branch | `master` |
| Baseline Git commit | `b4e42e1d6ea8a381ff81291ac3a52a020c456820` |
| Baseline Git tag | `v1.0-portfolio` |
| Version 2 development branch | `revision-v2` |
| Final Version 1 commit message | `Prepare project for portfolio presentation` |
| Baseline preserved on | July 22, 2026 |

## Version 1 System Scope

Version 1 includes:

- Synthetic workforce and reference-data generation
- Employee-event and historical workforce tables
- PostgreSQL database creation and validation
- SQL workforce analytics
- Retention modeling dataset construction
- Logistic Regression, Random Forest, and Gradient Boosting models
- Risk scoring and segmentation
- Interactive Streamlit dashboard
- End-to-end pipeline validation
- Project documentation and portfolio screenshots
- GitHub portfolio publication

## Test Population

The Version 1 model comparison used a test population of 1,478 records.

The test set contained:

- 125 positive attrition cases
- 1,353 negative attrition cases
- Positive-class prevalence of approximately 8.46%

Because attrition is the minority class, accuracy alone is not an appropriate model-selection metric.

## Version 1 Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.9154 | 0.0000 | 0.000 | 0.0000 | 0.5000 | 0.0846 |
| Logistic Regression | 0.5981 | 0.1376 | 0.712 | 0.2306 | 0.6968 | 0.1709 |
| Random Forest | 0.8904 | 0.1754 | 0.080 | 0.1099 | 0.7017 | 0.1662 |
| Gradient Boosting | 0.6962 | 0.1524 | 0.568 | 0.2403 | 0.6942 | 0.1645 |

These values are preserved as Version 1 results. Version 2 will regenerate the data and should not be expected to reproduce the same metrics.

## Version 1 Confusion Matrices

The positive class represents attrition.

### Logistic Regression

|  | Predicted No Attrition | Predicted Attrition |
|---|---:|---:|
| Actual No Attrition | 795 | 558 |
| Actual Attrition | 36 | 89 |

### Random Forest

|  | Predicted No Attrition | Predicted Attrition |
|---|---:|---:|
| Actual No Attrition | 1,306 | 47 |
| Actual Attrition | 115 | 10 |

### Gradient Boosting

|  | Predicted No Attrition | Predicted Attrition |
|---|---:|---:|
| Actual No Attrition | 958 | 395 |
| Actual Attrition | 54 | 71 |

## Version 1 Ranking Result

The Version 1 analysis reported a top-decile lift of approximately 2.15 times the baseline attrition rate.

This result will be recalculated after the Version 2 data generator, temporal datasets, and model-evaluation framework are completed.

## Version 1 Manufacturing Context

The Version 1 portfolio presentation reported:

- Manufacturing headcount: 1,971
- Manufacturing turnover rate: 10.46%

These are synthetic Version 1 results. They do not describe Tesla or any real employer. Version 2 will recalculate them after regenerating the workforce data.

## Version 1 Model-Selection Approach

Version 1 emphasized Logistic Regression because it provided comparatively high recall and straightforward interpretation.

The threshold-selection rule searched for a threshold satisfying a recall requirement and then maximized precision. On the Version 1 results, the selected threshold was 0.50.

Version 2 will replace this approach with:

- Separate training, validation, and test periods
- Cross-validation
- Bootstrap uncertainty estimates
- Calibration analysis
- Ranking metrics
- Explicit expected-value and intervention-cost assumptions

## Known Version 1 Limitations

The following limitations motivate the Version 2 revision:

1. Termination depended too strongly on a limited set of synthetic variables.
2. Termination dates were sampled in a way that could manufacture tenure patterns.
3. Some feature encodings were redundant or difficult to interpret.
4. Model and threshold decisions did not use a fully separate validation period.
5. Model differences were reported without bootstrap uncertainty estimates.
6. The dashboard did not clearly separate historical evaluation from current scoring.
7. Current department could be used to attribute historical events.
8. The project lacked substantive exploratory data analysis.
9. Calibration, subgroup performance, and fairness diagnostics were incomplete.
10. Manual validation had not yet been converted into a full automated test suite.
11. The intervention policy was not based on an explicit cost or capacity model.
12. The dashboard was not publicly deployed.

## Preservation Rule

The `master` branch and `v1.0-portfolio` tag represent the completed Version 1 portfolio baseline.

Version 2 development must occur on `revision-v2`. The Version 1 tag should not be moved or replaced.