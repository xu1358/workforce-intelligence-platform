# Version 2 Attrition Hazard Implementation

## Purpose

Checkpoint 36 replaces the Version 1 one-time termination draw and
uniform termination-date sampling with a monthly, competing-risks
simulation.

Each employee-month has three possible outcomes:

- Remain active
- Voluntary termination
- Involuntary termination

The simulator uses only information dated before the month being
simulated. This prevents future compensation, reviews, training, or
events from influencing an earlier termination decision.

## Two-Stage Generation Design

The Version 2 pipeline separates workforce history generation from
outcome generation:

1. Generate the workforce with every employee initially active.
2. Generate potential compensation, performance, training, and event
   histories without using termination outcomes.
3. Simulate monthly voluntary and involuntary attrition.
4. Remove records dated after each sampled termination.
5. Add one termination event for every terminated employee.
6. Repair location and manager assignments when a potential future
   event no longer occurs.
7. Reassign reports when a team manager terminates.

This ordering removes the Version 1 circular relationship in which
termination type directly changed generated performance.

## Hazard Inputs

The monthly logits combine:

- Smooth tenure effects
- Prior performance level and trend
- Prior compensation position and growth
- Months since prior promotion
- Prior training completion and failure
- Prior manager changes
- Employment type
- Job level
- Organizational level
- Seasonality
- Employee-level latent frailty
- Department, location, and organization-period effects
- Monthly random noise

Protected demographic attributes and future information are excluded.

All coefficients, probability bounds, latent-effect scales, and
calibration targets are stored in:

```text
config/attrition_hazard_config.yaml
```

### Executable simulation controls

The `simulation` mapping is an enforced runtime contract rather than
descriptive metadata:

| Setting | Committed value | Implemented behavior |
| --- | --- | --- |
| `time_step` | `month` | Builds monthly risk periods |
| `as_of_date` | `2026-06-30` | Stops observation at the current boundary |
| `cause_model` | `multinomial_logit` | Dispatches the three-outcome probability rule |
| `feature_lag_months` | `1` | Uses dated records strictly before the outcome month |
| `allow_first_month_exit` | `true` | Allows hire-month risk with event dates bounded by hire |
| `censor_at_as_of_date` | `true` | Right-censors active employees at the boundary |

Missing, mistyped, and unsupported controls fail before simulation. See
`docs/hazard_configuration_contract.md` for executable examples and
validation evidence.

## Hierarchy Handling

The static hierarchy now creates spare team-manager capacity by
targeting eight initial direct reports while retaining a hard maximum
of twelve.

Department heads and senior managers are structurally protected during
this version of the simulation. Team managers retain nonzero attrition
probabilities and may terminate. When one does, surviving reports are
assigned to an active manager in the same department with a valid level
and available capacity.

Every simulated reassignment creates a dated `Manager Change` event.
Potential manager changes that became impossible because a referenced
manager had already left are removed.

## Generated Audit Artifacts

The hazard step creates:

```text
data/processed/attrition_hazard_outcomes.csv
data/interim/attrition_hazard_monthly_diagnostics.csv
```

The outcomes file records each employee's simulated status, exit cause,
probabilities at the final observed month, and latent effects.

The diagnostics file records monthly risk-set size, average
cause-specific probabilities, sampled exits, and protected records.

These files are generated artifacts and remain excluded from Git.

## Validation

The implementation checks:

- Every employee receives exactly one simulated final status.
- Active employees have no termination date or termination event.
- Terminated employees have an exit date, exit cause, and exactly one
  termination event.
- Compensation, performance, training, and employee events do not
  occur after employment ends.
- Training completion does not occur after employment ends.
- Active employees report to active managers in the same department.
- Manager levels and direct-report capacities remain valid.
- Manager-change histories form a valid chronological chain.
- Promotion events match promotion compensation records.
- Protected hierarchy levels remain active.
- Team-manager attrition is nonzero.
- Monthly probabilities remain between zero and one.

## Reproducibility

All random draws use the configured seed. Repeated full data-generation
runs produce identical raw tables, attrition outcomes, and monthly
diagnostics.

After the Checkpoint 37 calibration review, the current reference run
produces:

- 10,000 simulated employees
- 2,591 cumulative terminations
- 25.91% cumulative termination rate across all hire cohorts
- 71.17% voluntary share
- 276 team-manager exits
- 1,913 manager reassignments
- 803 positive outcomes in the 2025-06-30 twelve-month modeling window

Checkpoint 37 compares these relationships with Version 1 and records
the calibration decision before temporal modeling begins.
