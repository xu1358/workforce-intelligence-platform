# Version 2 Retention Model Comparison

## Purpose

Checkpoint 41 compares three fixed candidate models without using the
reserved 2025 test target.

The primary evaluation is chronological:

- Fit on the 2023 snapshot.
- Evaluate on the 2024 validation snapshot.
- Keep the 2025 test snapshot locked.

This provides a more realistic backtest than the Version 1 random split.

## Candidate Models

The comparison includes:

1. Logistic Regression
2. Gradient Boosting
3. Random Forest

The settings are fixed before examining validation results. Checkpoint 41
does not perform repeated hyperparameter tuning against the validation
period.

Every model uses the same 21 raw features selected in Checkpoint 39 and
the same reference-category preprocessing.

## Why PR-AUC Is Primary

The 2024 validation attrition rate is approximately 10.61%. This means a
model can achieve high accuracy by predicting that almost everyone will
remain.

Precision-recall area under the curve, or PR-AUC, focuses on how well the
model ranks the uncommon positive class.

The validation prevalence of 0.106 is the approximate no-skill PR-AUC
baseline. Values above that baseline indicate useful ranking signal.

ROC-AUC, Brier score, fixed-threshold metrics, lift, and capture rate are
also reported because no single metric describes every operational use.

## Primary Temporal Validation Results

| Model | PR-AUC | ROC-AUC | Brier | Top-decile lift | Top-decile capture |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1638 | 0.6211 | 0.2403 | 1.8570 | 18.60% |
| Gradient Boosting | 0.1615 | 0.6293 | 0.2073 | 1.7548 | 17.58% |
| Random Forest | 0.1517 | 0.6224 | 0.1719 | 1.7207 | 17.24% |

Logistic Regression has the highest observed PR-AUC and top-decile lift.
Gradient Boosting has the highest observed ROC-AUC. Random Forest has the
lowest Brier score.

These probability estimates have not yet undergone the formal calibration
analysis planned for Checkpoint 42. The current Brier comparison should
therefore be treated as evidence about uncalibrated candidates, not as the
final probability-quality result.

## The 0.50 Threshold Is Reporting Only

Checkpoint 41 reports precision, recall, and F1 at 0.50 so readers can see
how the uncalibrated models behave at a familiar cutoff.

The cutoff is not selected or recommended here.

Threshold and intervention-policy decisions must use validation evidence
after calibration. They must not be chosen from test performance.

## Employee-Grouped Robustness Results

The mean five-fold employee-grouped PR-AUC values are:

| Model | Mean grouped PR-AUC |
|---|---:|
| Logistic Regression | 0.1665 |
| Gradient Boosting | 0.1664 |
| Random Forest | 0.1567 |

All yearly rows for one employee stay in one fold. Employee overlap
between fitting and validation is zero in every fold.

These folds mix 2023 and 2024 development rows, so they are not a temporal
test. They answer a separate robustness question: whether performance
persists for entirely unseen employees.

## Bootstrap Uncertainty

Checkpoint 41 uses 500 paired stratified bootstrap samples from the 2024
validation period.

Paired means that every model is evaluated on the same resampled employees
within one iteration. This makes model differences more directly
comparable.

The PR-AUC intervals are:

| Model | Lower 95% | Observed | Upper 95% |
|---|---:|---:|---:|
| Logistic Regression | 0.1482 | 0.1638 | 0.1857 |
| Gradient Boosting | 0.1483 | 0.1615 | 0.1799 |
| Random Forest | 0.1402 | 0.1517 | 0.1674 |

Pairwise conclusions:

- Logistic Regression versus Gradient Boosting: inconclusive.
- Logistic Regression versus Random Forest: inconclusive.
- Gradient Boosting is slightly higher than Random Forest in the paired
  interval, but the lower bound is approximately zero and the practical
  difference remains small.

These results do not support a claim that one model is universally best.

## Provisional Selection

Logistic Regression is selected provisionally because:

- It has the highest observed validation PR-AUC.
- It has the highest observed top-decile lift.
- Its grouped robustness PR-AUC is competitive.
- Pairwise uncertainty does not show a clear disadvantage.
- It is easier to explain, maintain, and audit than the ensemble models.

This is a validation-stage engineering decision, not a final test result.

Checkpoint 42 will determine whether Logistic Regression probabilities
should be calibrated and whether dashboard scores can be described as
probabilities.

## Generated Artifacts

`model_comparison_v2.csv`

- Primary validation metrics
- Bootstrap intervals
- Grouped robustness summaries
- Provisional selection result

`model_grouped_cv_metrics.csv`

- One row per candidate and employee-grouped fold

`model_metric_bootstrap.csv`

- All 500 paired bootstrap iterations for every candidate

`model_pairwise_pr_auc_bootstrap.csv`

- Paired PR-AUC difference intervals

`model_selection_decision_v2.csv`

- Practical tolerance, uncertainty, complexity, and selection evidence

`retention_validation_predictions_v2.csv`

- Validation targets and probabilities for each candidate

`model_comparison_validation.csv`

- Holdout, feature-policy, metric, fold, bootstrap, and selection checks

## Limitations

The data is synthetic, so the model performance does not establish
real-world employee behavior.

The fixed candidates are useful baselines, not exhaustive hyperparameter
searches. More tuning could improve a metric, but repeated validation
tuning would also increase overfitting risk.

Associations and predictive performance must not be interpreted as causal
effects or used as automatic employment decisions.
