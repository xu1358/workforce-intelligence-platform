# Multi-Snapshot Temporal Dataset Design

## Purpose

Checkpoint 38 replaces the single historical modeling snapshot with a
temporal panel suitable for backtesting. The design asks the same
question at several points in time:

> Using information available on the snapshot date, which employees
> will terminate during the following twelve months?

This structure is more realistic than randomly dividing one snapshot
into training and test rows. It allows later checkpoints to measure how
a model trained on earlier workforce conditions performs in a later
period.

## Historical snapshots

| Sequence | Feature snapshot | Prediction window ends | Eligible employees | Positive cases | Positive rate |
|---:|---|---|---:|---:|---:|
| 1 | 2023-06-30 | 2024-06-30 | 4,303 | 414 | 9.62% |
| 2 | 2024-06-30 | 2025-06-30 | 5,625 | 586 | 10.42% |
| 3 | 2025-06-30 | 2026-06-30 | 6,745 | 803 | 11.91% |

An employee is eligible when the employee:

- Was hired on or before the snapshot date.
- Had not terminated on or before the snapshot date.
- Meets the configured minimum-tenure rule, currently zero days.

The target is one only when termination occurs:

```text
after the snapshot date
and
on or before the prediction end date
```

Otherwise, the target is zero.

The three outcome periods are adjacent and do not overlap. An exit on
one boundary date cannot be counted in both neighboring windows.

## What one row represents

One historical row represents one employee at one snapshot date.

The same employee may appear in multiple snapshots if still employed.
This is intentional. The dataset contains:

- 16,673 historical employee-snapshot rows
- 7,745 unique historical employees
- 5,414 employees appearing in more than one snapshot

Because rows belonging to one employee are related, later model
validation must recognize employee groups. Checkpoint 40 will define
the final temporal and employee-disjoint evaluation strategies.

## Point-in-time feature rules

Every feature is calculated using records available on or before its
snapshot. The builder creates features from:

- Employee and organization attributes
- Compensation history
- Performance reviews
- Training history
- Promotion history
- Transfer history
- Manager-change history
- Leave history

Examples:

- June 2023 salary uses the latest compensation record available by
  2023-06-30.
- June 2023 performance uses reviews completed by 2023-06-30.
- Trailing training features use the twelve months ending on the
  snapshot.
- Promotion recency uses promotions occurring by the snapshot.
- An event from July 2023 cannot become a June 2023 model feature.

Direct outcome fields are excluded:

- `employment_status`
- `termination_date`
- `termination_type`

The employee identifier is retained for joining, auditing, and grouped
validation. It will not be used as a predictive feature.

## Historical location reconstruction

The employee table stores the final generated location. Using that
value for an earlier snapshot could leak a later transfer backward in
time.

Checkpoint 38 reconstructs location as follows:

1. Read the employee's hire location from the hire event.
2. Find the latest transfer on or before the snapshot.
3. Use that transfer destination when one exists.
4. Never use a later transfer as a model feature.

This changed the location assignment for 86 historical
employee-snapshot rows relative to the final employee record.

Department history is not reconstructed because the Version 2
generator currently produces location transfers only. Checkpoint 34
confirmed there were no hire-versus-current department mismatches.

## Current active scoring population

The separate file
`current_active_scoring_population.csv` represents the workforce active
on 2026-06-30.

It contains 7,409 employees and uses the same feature definitions as
the historical dataset. Its `attrition_next_12m` field is intentionally
blank because future outcomes after 2026-06-30 are unknown.

This prevents two different tasks from being mixed:

| Dataset | Purpose | Outcome known? |
|---|---|---|
| Historical temporal panel | Model development and backtesting | Yes |
| Current active population | Operational risk scoring | No |

The dashboard redesign in Checkpoint 48 will preserve this distinction.

Checkpoint 39 adds a downstream model-eligibility rule. Department
heads and senior managers remain in this current active file, but they
are excluded from retention-model fitting and scoring because the
Version 2 hierarchy protects those levels from generated termination.
The eligible current scoring population is therefore 7,305 employees.

## Main outputs

```text
data/processed/retention_multi_snapshot.csv
data/processed/current_active_scoring_population.csv
data/processed/temporal_dataset_validation/
```

Validation artifacts include:

- `validation_checks.csv`
- `snapshot_summary.csv`
- `panel_summary.csv`
- `feature_source_cutoffs.csv`
- `dataset_fingerprints.csv`

The generated CSV files remain excluded from Git because they can be
reproduced from the committed code and configuration.

## Implementation files

```text
config/temporal_snapshots.yaml
src/build_multi_snapshot_retention_dataset.py
sql/retention_modeling_multi_snapshot.sql
scripts/checkpoints/run_checkpoint38.ps1
notebooks/21_temporal_dataset_validation.ipynb
```

The Python implementation is the reproducible file-output path. The SQL
file provides an auditable PostgreSQL version of the same point-in-time
eligibility, feature, and target logic.

## Validation results

All Checkpoint 38 checks pass:

- Exact configured snapshot dates
- Unique employee-snapshot keys
- No direct outcome leakage columns
- Complete historical targets
- Binary historical targets
- Unknown current targets
- Complete identifier, organization, role, and salary fields
- Source records do not exceed snapshot dates
- Employees are active on their snapshot dates
- Historical location reconstruction is active
- Sufficient rows and positive cases in every period
- Positive rates between 9.62% and 11.91%
- Non-overlapping outcome windows
- Exact target reconciliation with termination dates
- Exact current active-population reconciliation

## Current limitations

- The data remains synthetic.
- Historical department changes are not simulated.
- Current scoring has no observable post-2026 outcomes.
- Multiple rows from one employee are statistically dependent.
- Feature redundancy and encoding are deferred to Checkpoint 39.
- Final train, validation, and test assignments are deferred to
  Checkpoint 40.
