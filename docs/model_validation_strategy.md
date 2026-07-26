# Version 2 Model Validation Strategy

## Purpose

Checkpoint 40 separates model development from final evaluation.

The earlier Version 1 workflow used a random train/test split from one
snapshot. That is useful for a teaching baseline, but it does not answer
the operational question:

> If the model learns from earlier workforce history, how well does it
> predict a later year?

Version 2 therefore uses chronological periods.

## Primary Temporal Split

| Role | Snapshot | Outcome window | Permitted use |
|---|---|---|---|
| Train | 2023-06-30 | 2023-07-01 through 2024-06-30 | Fit model parameters |
| Validation | 2024-06-30 | 2024-07-01 through 2025-06-30 | Compare models, calibrate probabilities, and select policies |
| Test | 2025-06-30 | 2025-07-01 through 2026-06-30 | One final evaluation after decisions are fixed |

The prediction windows touch at their boundaries but do not overlap.
Each termination can belong to only one outcome window.

## Why Validation Is Separate From Test

Choosing a model, probability calibration method, risk cutoff, or
intervention policy after examining test performance would make the test
set part of development.

That produces an overly optimistic final result.

The Version 2 rule is:

1. Fit candidate models with training data.
2. Make every model and policy decision with validation data.
3. Freeze those decisions.
4. Use the test period once for final reporting.

Checkpoint 40 does not include the target in
`model_split_assignments.csv`, does not assign test rows to grouped
folds, masks the test target inside the split-building process, and does
not report the test positive rate.

## Modeling Population

The split policy inherits the Checkpoint 39 population rule.

Included:

- Individual Contributors
- Team Managers

Excluded from retention-model development:

- Department Heads
- Senior Managers

The excluded levels remain in the workforce data. They are not modeled
because the Version 2 generator structurally protects those levels from
termination.

## Employee-Grouped Robustness Check

The primary temporal validation intentionally permits the same employee
to appear in more than one yearly snapshot. This reflects operational
use: an employee who remains active can be scored again the next year.

Checkpoint 40 also creates five employee-grouped folds across the 2023
and 2024 development periods.

For this secondary check:

- All rows for one employee receive the same fold.
- An employee cannot appear in both training and validation within a
  fold.
- The 2025 test period is never assigned a grouped fold.
- Stratification keeps attrition rates reasonably balanced.

This grouped analysis is a robustness check, not a replacement for the
chronological validation result. It answers a different question:

> Does performance remain similar when validation employees are entirely
> unseen during fitting?

## Generated Artifacts

`model_split_assignments.csv`

- One row per eligible employee-snapshot.
- Contains the primary split and optional grouped fold.
- Does not contain the attrition target.

`model_split_summary.csv`

- Reports train and validation target distributions.
- Reports only the row count for the reserved test period.

`group_cv_summary.csv`

- Reports row counts, employee counts, positive rates, and employee
  overlap for each robustness fold.

`model_split_validation.csv`

- Records every leakage, chronology, target-access, and group-integrity
  check.

## Relationship to Version 1

The older scripts based on `retention_modeling_dataset.csv` are retained
as Version 1 portfolio history. Their random split and threshold analysis
must not be used for Version 2 model selection or final reporting.

Beginning with Checkpoint 41, Version 2 modeling scripts must use
`retention_multi_snapshot.csv` joined to
`model_split_assignments.csv`.

## Interpretation Limits

These splits improve evaluation discipline, but they do not turn the
synthetic data into real employee evidence. Results remain properties of
the documented simulation.

The grouped folds are not temporal folds because they contain rows from
both development snapshots. They measure employee-disjoint robustness;
the 2024 validation period remains the primary time-based model-selection
test.
