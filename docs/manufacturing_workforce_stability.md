# Manufacturing Workforce Stability and Backfill Planning

## Purpose

Checkpoint 48 reframes the retention project around an operational workforce
question:

> Given current attrition probabilities and a constrained retention policy,
> what probability-weighted backfill demand should a manufacturing
> organization prepare for over the next 12 months?

This checkpoint does not train another model or change the frozen policy. It
translates the current scores from Checkpoint 46 into department and
manufacturing-role planning units.

## Why Manufacturing

Manufacturing is the largest synthetic department in the current active
workforce. It also has substantial probability-weighted departure exposure.
This makes it a useful focus for workforce-capacity, line-coverage, replacement
training, and backfill-planning examples.

The analysis recalculates Version 2 values. It does not reuse the older
Version 1 headcount or turnover percentages quoted in the original revision
proposal.

## Data and Timeline

The analysis uses:

- `current_active_scoring_population.csv`
- `current_retention_policy_scores.csv`
- `current_retention_policy_summary.csv`
- `retention_policy_decision.csv`

All inputs describe the synthetic workforce as of `2026-06-30`. Outcomes after
that date are unknown and are not used.

## Core Calculations

### Expected departures

For a group of employees:

```text
Expected departures = Sum of individual attrition probabilities
```

An expected value of `229.7` means that the probabilities sum to approximately
229.7 departures. It does not predict that exactly 230 named people will leave.

### Expected prevented departures

For selected employees:

```text
Expected prevented departures
    = Attrition probability
    × Intervention success probability
```

The base policy uses a 25% intervention-success assumption. This value is a
scenario assumption, not a causal estimate from an experiment.

### Residual expected backfills

```text
Residual expected backfills
    = Expected departures
    − Expected prevented departures
```

This value represents probability-weighted backfill exposure after applying the
policy assumptions. It is not an approved hiring requisition count.

### Replacement-cost exposure

```text
Replacement-cost exposure
    = Attrition probability
    × Replacement cost
```

The base replacement scenario uses 100% of salary. Low and high scenarios
remain documented in the Checkpoint 45 sensitivity analysis.

### Operational units

Residual backfill demand is translated into:

- Vacancy days
- Time-to-productivity days
- Replacement training hours
- Temporary coverage hours

These are separate planning units. They should not be added together and are
not dollar amounts.

## Manufacturing Scenario

Using the frozen base assumptions, the current synthetic Manufacturing
scenario reports approximately:

- 1,832 active employees
- 1,808 model-eligible active employees
- 24 protected-level active employees
- 229.7 probability-weighted expected departures
- 162 employees selected for human review
- 8.4 expected prevented departures
- 221.3 residual expected backfills
- $405,000 planned intervention spend
- $492,919 expected net value

The expected backfill exposure is concentrated in:

1. Production Technicians
2. Manufacturing Engineers
3. Production Supervisors

These values are scenario outputs for a synthetic portfolio project.

## Dashboard

Checkpoint 48 adds a new **Workforce Stability** tab without removing the
existing tabs. It displays:

- Manufacturing summary metrics
- Department backfill exposure
- Expected net value by department
- Manufacturing role-level backfill estimates
- Vacancy, productivity, training, and coverage units
- Explicit forecast and governance caveats

## Governance

- Current probabilities are not known future outcomes.
- Expected prevented departures are not guaranteed.
- Selection remains human-review only.
- Automatic employment action is prohibited.
- Backfill estimates support planning; they do not authorize hiring or
  employment decisions.

## Outputs

Checkpoint 48 writes:

- `workforce_stability_by_department.csv`
- `manufacturing_backfill_by_role.csv`
- `manufacturing_stability_summary.csv`
- `workforce_stability_validation.csv`

Dashboard copies are written under `data/processed/dashboard/`.

Generated data remains excluded from Git and can be recreated with:

```powershell
.\scripts\checkpoints\run_checkpoint48.ps1
```
