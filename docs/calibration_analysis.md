# Retention Probability Calibration

## Purpose

Checkpoint 42 tests whether the provisional Logistic Regression scores can
be interpreted as estimated 12-month attrition probabilities.

Checkpoint 41 established that Logistic Regression was a reasonable
validation-stage model choice. It did not establish that a score such as
`0.40` meant an employee had a 40% probability of attrition.

Calibration addresses that separate question.

## Leakage-Safe Design

The calibration process uses three time boundaries:

| Period | Use |
|---|---|
| 2023 snapshot | Fit the model and calibration mappings |
| 2024 snapshot | Compare calibration methods |
| 2025 snapshot | Remain reserved for final evaluation |

The 2023 training rows are divided into five folds. For each fold:

1. Logistic Regression fits on the other four folds.
2. It predicts the held-out fold.
3. Those held-out probabilities are saved.

Every 2023 row therefore receives a probability from a model that did not
fit that row. These out-of-fold probabilities are used to learn the
calibration mappings.

The 2024 outcomes are used only to compare the finished mappings. They are
not used to fit sigmoid or isotonic calibration.

The 6,641 reserved 2025 rows remain outside the calibration analysis, and
zero test targets are accessed.

## Methods

### Uncalibrated

The original Logistic Regression probability is retained as a baseline.
The model uses balanced class weights, which improves minority-class
learning but causes the raw scores to overstate absolute attrition risk.

### Sigmoid

Sigmoid calibration learns a smooth S-shaped transformation from
out-of-fold model scores to observed training outcomes.

It is monotonic, so employee ranking is preserved.

### Isotonic

Isotonic calibration learns a flexible nondecreasing step function.

It can represent more complex calibration patterns, but it can create tied
probabilities and is more vulnerable to sampling variation than sigmoid
calibration.

## Primary Validation Results

| Method | Brier | Log loss | ECE | Mean prediction | Observed rate |
|---|---:|---:|---:|---:|---:|
| Uncalibrated | 0.2403 | 0.6722 | 0.3662 | 47.23% | 10.61% |
| Sigmoid | 0.0932 | 0.3299 | 0.0089 | 10.02% | 10.61% |
| Isotonic | 0.0936 | 0.3303 | 0.0087 | 10.17% | 10.61% |

Lower values are better for Brier score, log loss, and expected
calibration error.

The uncalibrated model substantially overpredicts risk because its mean
score is 47.23%, while the validation attrition rate is only 10.61%.

Both calibration methods correct most of that gap.

## Temporal Calibration Transport

The validation result is not the end of the calibration story. The
model-eligible observed rate rises from 10.61% in validation to 12.09% in the
once-only final test. Over the same transition, the selected model's mean
probability rises from 10.02% to 11.02%.

| Period | Observed rate | Mean probability | Calibration gap |
| --- | ---: | ---: | ---: |
| 2024 validation | 10.61% | 10.02% | −0.59 percentage points |
| 2025 final test | 12.09% | 11.02% | −1.07 percentage points |

Observed prevalence increases 1.478 percentage points while the predicted
mean increases 1.002 percentage points. The model therefore tracks part, but
not all, of the temporal increase; its calibration gap becomes 0.476
percentage points more negative.

This pattern is consistent with **prior-probability shift risk**. It does not
prove pure label shift because feature distributions and conditional outcome
relationships may also change. The final-test rate is not used to recalibrate
the model. See the
[temporal base-rate and prior-shift audit](temporal_prior_shift.md).

## Understanding the Brier Score

The Brier score is the average squared distance between:

- The predicted probability.
- The actual outcome, zero or one.

A perfectly calibrated and perfectly accurate model would have a Brier
score of zero.

Sigmoid calibration improves the score by approximately:

`0.2403 - 0.0932 = 0.1471`

The paired bootstrap interval for this improvement is approximately
`0.1438` to `0.1507`, which remains entirely above zero.

This is strong evidence that the improvement is not a small resampling
artifact.

## Reliability Analysis

Each method's validation predictions are sorted and divided into ten
equal-frequency groups.

For every group, the analysis compares:

- Mean predicted probability.
- Observed attrition rate.

The expected calibration error, or ECE, summarizes the weighted absolute
gap across those groups.

Sigmoid calibration achieves an ECE of approximately 0.0089. This means
the average bin-level difference is less than one percentage point.

Reliability on synthetic validation data does not guarantee equivalent
performance on real employees or future time periods.

## Ranking Is a Separate Property

Calibration changes probability scale. It should not be expected to
create a stronger ranking model.

Sigmoid calibration preserves the original results:

| Metric | Uncalibrated | Sigmoid |
|---|---:|---:|
| PR-AUC | 0.1638 | 0.1638 |
| ROC-AUC | 0.6211 | 0.6211 |
| Top-decile lift | 1.8570 | 1.8570 |

Isotonic calibration creates tied scores, so its PR-AUC is slightly lower.

## Selection Decision

Sigmoid calibration is selected because:

- Its Brier score is the lowest observed value.
- Its improvement over uncalibrated scores is large and statistically
  clear in the paired bootstrap.
- Its Brier score is practically indistinguishable from isotonic
  calibration.
- It preserves ranking exactly.
- It is smoother and simpler than the isotonic step function.

This remains a validation-stage decision. The reserved test outcomes have
not been opened.

## Dashboard Recommendation

The selected sigmoid scores meet both committed display conditions:

- ECE no greater than 0.03.
- Absolute mean calibration gap no greater than 0.02.

The recommended dashboard label is:

`Estimated 12-month attrition probability`

The dashboard must still state the as-of date, prediction window, synthetic
data limitation, and noncausal nature of the score.

## Generated Artifacts

`retention_calibration_metrics.csv`

- Validation metrics and bootstrap confidence intervals

`retention_reliability_bins.csv`

- Ten reliability bins for every method

`retention_calibration_bootstrap.csv`

- Five hundred paired validation resamples for every method

`retention_calibration_differences.csv`

- Pairwise Brier-score difference intervals

`retention_calibration_selection.csv`

- Selection rules, evidence, and dashboard label

`retention_calibration_predictions.csv`

- Validation probability for every method and employee

`retention_calibration_folds.csv`

- Out-of-fold training counts and integrity checks

`retention_calibration_validation.csv`

- Nineteen automated calibration checks

## Limitations

This project uses synthetic data. A calibrated synthetic probability is
not evidence of real-world employee behavior.

Calibration may drift when the employee population, base attrition rate,
or economic environment changes. A production system would require
ongoing monitoring and scheduled recalibration. This project observes that
risk directly: the full historical base rate rises from 9.62% to 10.42% to
11.91%, and final-test aggregate underprediction is −1.07 percentage points.

The scores support aggregate planning and carefully reviewed outreach.
They must not be used as automatic employment decisions or interpreted as
causal conclusions about an individual.
