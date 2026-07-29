# Executable Attrition-Hazard Configuration

## Purpose

Checkpoint 60 closes a configuration-to-code gap in the Version 2 synthetic
attrition generator. Four values in `config/attrition_hazard_config.yaml`
previously described the intended simulation but were not read by the
implementation:

- `cause_model`
- `feature_lag_months`
- `allow_first_month_exit`
- `censor_at_as_of_date`

The committed values already matched the hard-coded implementation. This
checkpoint therefore wires and validates the settings without changing the
reference seed, coefficients, generated outcomes, model results, frozen
policy, or dashboard data.

## Runtime Contract

`hazard_simulation_settings()` reads the full `simulation` mapping once and
returns a typed, immutable `HazardSimulationSettings` object. Missing,
mistyped, or unsupported values fail before random simulation begins.

| YAML key | Committed value | Runtime use |
| --- | --- | --- |
| `time_step` | `month` | Selects monthly risk periods |
| `as_of_date` | `2026-06-30` | Sets the last observed date |
| `cause_model` | `multinomial_logit` | Selects the competing-risk probability calculation |
| `feature_lag_months` | `1` | Sets the exclusive cutoff for dated history |
| `allow_first_month_exit` | `true` | Includes the hire month in the risk set |
| `censor_at_as_of_date` | `true` | Administratively censors active employees at the observation boundary |

The implementation contains an explicit runtime reference for every row in
this table. The test suite also compares the YAML keys with the committed
contract, so adding a new simulation key without implementing it will fail.

## Feature-Lag Meaning

The simulated outcome period starts on the first day of a month. Dated
features use records strictly before the configured cutoff.

For a March 2025 outcome month:

| Lag | Exclusive record cutoff | Latest possible included date |
| --- | --- | --- |
| 1 month | `2025-03-01` | `2025-02-28` |
| 2 months | `2025-02-01` | `2025-01-31` |

The committed one-month lag reproduces the original implementation exactly:
records dated in the outcome month cannot affect that month’s exit draw.
Tenure remains measured at the outcome-month start because it is an elapsed
time quantity, not a dated history record.

## First-Month Exit Meaning

When `allow_first_month_exit: true`, an employee hired during a month may
enter that month’s risk set. The sampled event date is still bounded between
the hire date and the end of the observed month.

When the value is `false`, the employee first enters the risk set in the
following month. Unit tests exercise both branches even though the committed
reference setting remains `true`.

## Cause-Model Meaning

`cause_model: multinomial_logit` selects the implemented three-outcome
probability rule:

1. stay active;
2. voluntary exit; or
3. involuntary exit.

The two cause logits are exponentiated relative to the stay-active reference,
bounded by the committed probability limits, and normalized to a valid
distribution. An unrecognized model name now raises an error instead of
silently using multinomial logit.

Only `multinomial_logit` is currently supported. Supporting another model
would require a separate implementation branch and tests before its name
could be accepted in YAML.

## Administrative Censoring

The generated workforce has no observation horizon after `2026-06-30`.
Therefore `censor_at_as_of_date` must be Boolean and must remain `true`.
Active employees are observed through the as-of date without inventing a
future exit.

Setting the value to `false` now fails with an actionable error. Treating it
as a freely switchable option would imply the generator knew what happened
after the available history boundary.

## Validation Evidence

`src/validate_hazard_configuration.py` verifies:

- the exact six-key simulation contract;
- typed materialization of every configured value;
- an implementation reference for every setting;
- different feature-lag cutoffs;
- enabled and disabled hire-month risk;
- cause-model dispatch and probability reconciliation;
- mandatory as-of censoring;
- preservation of the committed default semantics; and
- governance isolation from model, policy, and dashboard results.

The validator saves aggregate evidence under:

```text
data/processed/hazard_configuration/
```

The automated tests additionally reject missing keys, invalid lags,
non-Boolean switches, an unsupported cause model, and an uncensored open
horizon.

Run the checkpoint on Windows with:

```powershell
.\scripts\checkpoints\run_checkpoint60.ps1
```

Run the portable engineering suite on any supported platform with:

```bash
python scripts/run_project.py quality
```

## Checkpoint 59 Compatibility Repair

While tracing this configuration through the cross-platform runner,
Checkpoint 60 also repairs the archived PowerShell runners’ project-root
calculation. Moving those files into `scripts/checkpoints/` added one
directory level, so they now resolve the repository root through the parent
`scripts/` directory.

This is a runner-path correction only. It does not execute, regenerate, or
retune any analytical artifact.
