# Fairness and Subgroup Diagnostics

## Purpose

Checkpoint 44 evaluates whether the selected retention model behaves
differently across employee groups.

Checkpoint 43 established that the model provides useful overall ranking
value. Overall performance can still hide uneven behavior. A model can have
acceptable aggregate lift while selecting, missing, or miscalibrating some
groups more often than others.

This checkpoint is a diagnostic audit. It does **not**:

- Declare the model fair or unfair
- Establish discrimination or causation
- Select an intervention policy
- Authorize employment decisions
- Access the reserved 2025 test outcomes

The analysis uses the same 2024 validation population and the same
sigmoid-calibrated Logistic Regression selected in Checkpoints 41 and 42.

## Available and unavailable attributes

The available subgroup dimensions are:

- Age band
- Education level
- Region
- Employment type
- Department
- Job level
- Organizational level

Age bands are derived from the snapshot age:

```text
Under 30
30–39
40–49
50–59
60+
```

The employee schema does not contain gender, race, ethnicity, nationality,
disability status, or other protected-class fields. Gender appeared in the
original revision roadmap, but it cannot be evaluated with the current data.
The analysis does not invent or infer a missing attribute.

This absence is a limitation. It is not evidence that the model is fair with
respect to an unobserved attribute.

## Reporting policy

Checkpoint 43 used a global top-10% reporting scenario. Checkpoint 44 preserves
that exact scenario:

```text
Validation employees: 5,521
Selected employees:     553
Observed attrition:      586
Captured attrition:      109
```

The top 10% is a reporting device, not a final intervention policy.
Checkpoint 46 will compare decision policies using explicit costs and benefits.

The global ranking is applied once to the complete validation population.
Checkpoint 44 does not independently select the top 10% inside every group.
This is important because group selection rates are allowed to differ and are
therefore observable.

## Subgroup metrics

For every available group, the analysis reports:

### Population and outcome measures

- Sample size
- Positive attrition cases
- Observed attrition rate
- Mean predicted attrition probability

### Top-10% reporting measures

- Selection rate
- Precision
- True-positive rate, also called recall or capture rate
- False-positive rate
- True positives, false positives, false negatives, and true negatives

### Probability and ranking measures

- Calibration gap
- Brier score
- PR-AUC
- ROC-AUC

The calibration gap is:

```text
Mean predicted probability − observed attrition rate
```

A positive value means the group is overpredicted on average. A negative value
means the group is underpredicted on average.

Each binomial rate includes a 95% Wilson confidence interval.

## Sparse-group protection

Every group is preserved in the detailed results, but a group contributes to
max-minus-min disparity comparisons only when it has at least:

```text
100 total employees
20 positive cases
50 negative cases
10 selected employees
```

These requirements keep a very small or outcome-sparse group from determining
an extreme disparity estimate.

Two groups are reported but excluded from disparity ranges:

- Doctorate education: only 11 positive cases
- Job level 4: only 5 positive cases

Exclusion from a disparity range does not mean the group is unimportant. It
means the available validation evidence is too sparse for a stable comparison.

## Descriptive disparity measures

Checkpoint 44 uses max-minus-min differences across eligible groups:

### Demographic-parity difference

```text
Highest group selection rate − lowest group selection rate
```

This is a descriptive selection-rate gap. It is not a legal conclusion.

### Equal-opportunity difference

```text
Highest group true-positive rate − lowest group true-positive rate
```

This measures how differently known attrition cases are captured.

### False-positive-rate difference

```text
Highest group false-positive rate − lowest group false-positive rate
```

This measures how differently employees without attrition are selected.

### Equalized-odds diagnostic

The checkpoint reports the larger of:

- The true-positive-rate difference
- The false-positive-rate difference

### Calibration diagnostic

The checkpoint reports the largest absolute mean calibration gap among
eligible groups.

## Selected-model findings

The selected model produced the following descriptive gaps:

| Attribute | Selection-rate gap | Recall gap | False-positive-rate gap | Maximum absolute calibration gap |
| --- | ---: | ---: | ---: | ---: |
| Age band | 3.00% | 7.75% | 3.25% | 1.12% |
| Education level | 3.63% | 4.82% | 4.92% | 3.90% |
| Region | 15.61% | 26.30% | 14.18% | 1.93% |
| Employment type | 19.57% | 28.09% | 18.10% | 0.64% |
| Department | 43.40% | 54.19% | 41.57% | 2.81% |
| Job level | 22.38% | 35.26% | 20.49% | 1.85% |
| Organizational level | 8.00% | 17.74% | 6.80% | 3.54% |

The largest gaps occur for department, employment type, job level, and region.

### Examples

At the global top-10% reporting cutoff:

- Customer Support had a 44.61% selection rate.
- Engineering had a 1.21% selection rate.
- Hourly employees had a 26.43% selection rate.
- Salaried employees had a 6.87% selection rate.
- Job level 1 had a 25.16% selection rate.
- Job level 3 had a 2.77% selection rate.

These are model-selection rates, not termination decisions.

The global cutoff magnifies relatively small score differences. If one group
receives scores concentrated just above the cutoff and another receives scores
just below it, their selection rates can differ sharply even when their
average observed attrition rates are closer.

## Why the differences may occur

The Version 2 hazard intentionally includes employment and workforce factors:

- Hourly employment
- Job level
- Organizational level
- Department-period variation
- Location-period variation
- Performance, compensation, training, and event histories

The model also includes department and job level, and it can use correlated
features as proxies for workforce structure.

Age and education were explicitly excluded from the attrition hazard. They
were nevertheless available to the predictive model because Checkpoint 39
selected them during a development-stage feature policy. They may capture
random patterns or indirect structure rather than a true data-generating
effect.

Because all records are synthetic, these findings describe this simulator and
this fitted model. They do not describe Tesla, another employer, or real
employees.

## Age-and-education exclusion sensitivity

Checkpoint 44 fits a diagnostic version of Logistic Regression after removing:

```text
approx_age
education_level
```

It uses the same:

- 2023 training period
- Out-of-fold sigmoid calibration process
- 2024 validation period
- Top-10% reporting fraction

It does not replace or reselect the production candidate.

### Overall comparison

| Metric | Selected model | Age and education excluded |
| --- | ---: | ---: |
| Brier score | 0.09321 | 0.09313 |
| PR-AUC | 0.16381 | 0.16683 |
| ROC-AUC | 0.62113 | 0.62001 |
| Top-10% precision | 19.71% | 20.07% |
| Top-10% capture | 18.60% | 18.94% |
| Captured cases | 109 | 111 |

The changes are small and are not treated as evidence that one variant is
meaningfully better.

Removing age and education did not consistently reduce subgroup gaps. For
example, the age-band recall gap increased from 7.75% to 14.28%, and the
education recall gap increased from 4.82% to 15.73%.

This illustrates an important fairness principle:

> Removing a direct demographic feature does not guarantee equal model
> behavior because other features can retain correlated or proxy information.

## Screening flags

The configuration contains portfolio review triggers:

```text
Selection-rate gap:           greater than 10 percentage points
Recall gap:                   greater than 10 percentage points
False-positive-rate gap:      greater than 10 percentage points
Absolute calibration gap:     greater than 5 percentage points
```

These triggers identify findings that deserve investigation. They are not
legal thresholds, policy standards, or proof of unfair treatment.

For the selected model:

- Region triggered selection, recall, and false-positive review.
- Employment type triggered selection, recall, and false-positive review.
- Department triggered selection, recall, and false-positive review.
- Job level triggered selection, recall, and false-positive review.
- Organizational level triggered recall review.
- Age band and education did not trigger the configured screens.

## Prediction Fairness Does Not Guarantee Allocation Equity

Checkpoint 44 audits calibrated probabilities and a top-10% reporting
scenario before the economic policy is applied. Checkpoint 46 later ranks:

\[
P(\text{attrition})
\times
\text{salary-based replacement cost}
\]

Salary therefore enters the decision after the original model fairness
analysis. Job level is only a partial proxy and cannot replace direct
salary-band analysis.

Checkpoint 56 adds a separate post-policy allocation-equity audit. In the
current population:

- Eligible mean salary is $95,722.
- Frozen-policy selected mean salary is $125,670.
- The highest salary quintile has a 21.22% selection rate.
- The lowest salary quintile has a 0.82% selection rate.
- The highest-to-lowest selection-rate ratio is 25.83.
- Only 321 of the 700 frozen selections overlap probability-only top-700
  selection.

This concentration is a structural consequence of the financial objective,
not merely a statistical subgroup fluctuation. The expected-value policy is
financially optimized, not equity-neutral.

The audit compares probability-only, salary-capped, and constant-cost
sensitivity policies. It does not access outcome columns, retune the final
test, or replace the frozen policy. A different policy would require a new
holdout or prospective evaluation.

See the
[retention policy salary-allocation equity audit](retention_policy_equity.md).

## Ethical use restrictions

This project must not be represented as an automated employment-decision
system.

If a real organization considered a retention model, it would need:

- Legal and human-resources review
- Verified, consented, and appropriately governed data
- Independent bias and validity assessment
- Clearly defined permitted uses
- Human review and appeal processes
- Monitoring for drift and changing subgroup behavior
- Strict controls preventing punitive use
- Evaluation of whether intervention benefits are equitably available

The appropriate use of a retention score is supportive resource planning, not
discipline, termination, promotion denial, or surveillance.

## Limitations

1. The data is synthetic.
2. Several legally and ethically relevant attributes are unavailable.
3. The analysis considers one subgroup dimension at a time.
4. Intersectional analysis is deferred because many combinations would be
   sparse.
5. The top-10% scenario is not a chosen business policy.
6. Max-minus-min gaps can be sensitive to category definitions.
7. Confidence intervals for individual rates do not by themselves establish
   the uncertainty of every disparity difference.
8. Removing direct fields cannot remove all proxy information.
9. Observational predictive differences do not establish causation.
10. Model-level subgroup diagnostics do not evaluate salary-driven allocation
    introduced by the later expected-value policy.

## Reproduction

Run:

```powershell
.\scripts\run_checkpoint44.ps1
```

The script rebuilds Checkpoint 42 only when required inputs are missing or when
the optional rebuild switch is used.

Generated outputs are saved under:

```text
data/processed/retention_fairness_group_metrics.csv
data/processed/retention_fairness_disparities.csv
data/processed/retention_fairness_model_comparison.csv
data/processed/retention_fairness_predictions.csv
data/processed/retention_fairness_validation.csv
data/processed/fairness_figures/
```
