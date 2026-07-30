# Temporal Base-Rate Drift and Prior-Shift Risk

## Purpose

Checkpoint 65 makes temporal outcome prevalence an explicit model-monitoring
result. Earlier checkpoints reported the three historical rates in notebook
output but did not explain how their upward movement affects probability
calibration transported from validation to the final test.

The audit answers four questions:

1. How much does observed 12-month attrition prevalence change across periods?
2. Is the same pattern present in the population used by the classifier?
3. How does the validation-to-test change relate to aggregate underprediction?
4. What can be concluded without recalibrating on the reserved final test?

The audit is post-evaluation and diagnostic. It does not change the selected
model, calibrated probabilities, frozen intervention policy, current
dashboard, or employee review population.

## Results Snapshot

The full historical population has these observed rates:

| Snapshot role | Rows | Positive cases | Observed attrition rate |
| --- | ---: | ---: | ---: |
| 2023 train | 4,303 | 414 | 9.62% |
| 2024 validation | 5,625 | 586 | 10.42% |
| 2025 final test | 6,745 | 803 | 11.91% |

The increase from 9.62% to 11.91% is:

- 2.284 percentage points in absolute terms; and
- 23.74% relative to the first period.

The upward movement is material. It should be discussed whenever the project
claims that a probability scale learned in one period can be transported to a
later period.

## Classifier Population

The primary model excludes Senior Managers and Department Heads from
individual scoring. Its prevalence is therefore slightly different from the
full historical population:

| Snapshot role | Model-eligible rows | Positive cases | Observed attrition rate |
| --- | ---: | ---: | ---: |
| 2023 train | 4,206 | 414 | 9.84% |
| 2024 validation | 5,521 | 586 | 10.61% |
| 2025 final test | 6,641 | 803 | 12.09% |

The model-eligible rate rises 22.84% relative from train to final test. More
directly relevant to calibration transport, it rises from 10.61% during
validation to 12.09% during the final test:

- 1.478 percentage points absolute; and
- 13.92% relative.

This distinction prevents the 9.62%–11.91% full-population diagnostic from
being confused with the 10.61%–12.09% population on which model calibration
is evaluated.

## Calibration Transport

Sigmoid calibration was selected using the 2024 validation period. Aggregate
calibration then changes as follows:

| Period | Observed rate | Mean predicted probability | Mean calibration gap |
| --- | ---: | ---: | ---: |
| 2024 validation | 10.61% | 10.02% | −0.59 percentage points |
| 2025 final test | 12.09% | 11.02% | −1.07 percentage points |

The final model underpredicts the final-test aggregate by 1.07 percentage
points. That result is real and is now documented directly rather than left
only in notebook output.

The temporal decomposition is:

| Component | Validation-to-test change |
| --- | ---: |
| Observed attrition rate | +1.478 percentage points |
| Mean predicted probability | +1.002 percentage points |
| Observed increase not tracked by predicted mean | +0.476 percentage points |
| Calibration-gap change | −0.476 percentage points |

The model does react to the changed feature mix: its mean prediction rises by
about one percentage point. It does not capture the entire prevalence
increase, so the negative calibration gap becomes about 0.476 percentage
points worse.

This is more precise than saying the period drift explains the final
underprediction "exactly." Validation already has a −0.59 percentage-point
gap. Temporal prevalence drift is an important contributor to the larger
final gap, but the aggregate evidence does not identify one exclusive cause.

## Prior-Probability Shift

The standard monitoring term is **prior-probability shift** or **label-prior
shift**: the marginal prevalence of the outcome changes between the source
period and the target period.

For validation and final test, the observed prior-odds multiplier is:

```text
[0.1209 / (1 - 0.1209)] / [0.1061 / (1 - 0.1061)] = 1.1584
```

This means the observed final-test attrition odds are about 15.84% higher than
the validation-period attrition odds.

The evidence is described as **prior-probability shift risk**, not proof of
pure label shift. A pure label-shift assumption would also require the feature
distribution conditional on the outcome to remain stable. This audit does not
establish that stronger condition. Feature drift, relationship drift, and
sampling differences can coexist with changing prevalence.

Prevalence alone does not prove pure label shift.

## Why Final-Test Recalibration Is Prohibited

The 12.09% final-test rate becomes known only after the final outcome window
ends. Using that number to modify final-test probabilities and then reporting
the adjusted result as model performance would tune on the test.

Checkpoint 65 therefore does not:

- fit a new calibrator;
- apply a post-hoc prior correction to employee probabilities;
- recalculate the frozen intervention policy;
- change dashboard scores;
- claim improved final-test Brier score; or
- reopen model selection.

The final test remains a once-only evaluation of the frozen development
process. Its underprediction is retained as honest evidence about temporal
transport.

## Monitoring Response in a Real Deployment

A production system would handle this limitation prospectively:

1. Record the score date, model version, and predicted-probability
   distribution.
2. Wait until the corresponding outcome window matures.
3. Compare the matured observed rate with the mean prediction.
4. Track calibration gap, Brier score, reliability bins, and subgroup gaps by
   period.
5. Trigger review when prevalence or calibration changes beyond a committed
   tolerance.
6. Fit any recalibration only on a newly designated labeled development or
   monitoring window.
7. Evaluate the revised calibration on a later untouched period before
   deployment.

Current future outcomes are unknown, so the project cannot legitimately
estimate a current-period calibration correction. Current probabilities
remain scenario inputs for human review and planning, not verified forecasts.

## Evidence and Reproduction

Run:

```bash
python src/audit_temporal_prior_shift.py
```

The audit writes aggregate artifacts to:

```text
data/processed/temporal_prior_shift/
```

The generated tables are:

- `temporal_base_rate_summary.csv`
- `calibration_shift_summary.csv`
- `prior_shift_decomposition.csv`
- `validation_checks.csv`

The generated figures are:

- `temporal_base_rate_drift.png`
- `observed_vs_predicted_rate.png`

No employee identifier, employee-level outcome, employee-level probability,
or review-selection flag is saved.

## Validation Contract

The automated contract verifies that:

- all three full-population denominators and positive counts reconcile;
- all three model-eligible denominators and positive counts reconcile;
- the reported period rates reproduce the source panel;
- the 23.74% first-to-final relative drift is explicit;
- validation metrics reproduce the selected sigmoid calibration output;
- final metrics reproduce the once-only frozen result;
- the validation-to-test prevalence increase and gap deterioration reconcile;
- prior-shift wording remains qualified;
- saved outputs remain aggregate; and
- no model, policy, or dashboard probability is changed.

## Interpretation Boundary

This synthetic result demonstrates how to detect and communicate calibration
transport risk. It does not prove that any real workforce has the same drift,
that attrition is caused by time itself, or that a simple prior correction
would be sufficient in production.

The defensible conclusion is:

> Attrition prevalence rises materially across the three synthetic periods.
> The final model partially tracks that increase but underpredicts the final
> aggregate by 1.07 percentage points. This is consistent with
> prior-probability shift risk, although prevalence alone does not prove pure
> label shift. The final test is not used for recalibration.
