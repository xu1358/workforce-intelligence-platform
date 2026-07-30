# Version 1 and Version 2 Data Comparison

## Checkpoint 37 Decision

Version 2 passes the required temporal-integrity, calibration, hierarchy,
leakage, and signal checks and is approved for multi-snapshot dataset
construction.

Checkpoint 37 initially identified that the first hazard configuration
produced insufficient observable signal: a leakage-safe five-fold Logistic
Regression generator diagnostic reached approximately 0.54 ROC-AUC, below the
development-only acceptance minimum of 0.62.

The hazard was recalibrated before approval:

- Baseline monthly exit probabilities were reduced.
- Observable performance, compensation, promotion, training, and
  manager-change effects were strengthened.
- Unobserved frailty and period-noise scales were reduced but retained.

This adjustment increased detectable signal without making the outcome
deterministic or pushing the twelve-month attrition rate outside its
acceptance range. It occurred before formal temporal model development and
without access to the reserved final-test target.

## Headline Comparison

| Metric | Version 1 | Version 2 |
|---|---:|---:|
| Total generated employees | 10,000 | 10,000 |
| Cumulative terminations | 1,713 | 2,591 |
| Cumulative attrition rate | 17.13% | 25.91% |
| Voluntary share | 75.83% | 71.17% |
| Leadership terminations | 0 | 276 |
| 2025 snapshot eligible employees | 7,386 | 6,745 |
| Next-12-month positive cases | 627 | 803 |
| Next-12-month positive rate | 8.49% | 11.91% |
| Manager reassignments after exit | 0 | 1,913 |

The cumulative rate is not an annual attrition rate. It represents all
terminations across multiple hire cohorts between 2021 and the
2026-06-30 administrative censor date.

The 11.91% snapshot rate is the relevant annualized modeling quantity:
employees active on 2025-06-30 who terminate during the following
twelve months.

## Structural Improvements

### Termination timing

Version 1 selected a termination outcome once and then sampled a date
uniformly between hire plus 60 days and the as-of date.

Version 2 evaluates each active employee monthly. Event timing emerges
from changing cause-specific hazards and the employee's time at risk.

### Behavioral history

Version 1 decided termination before compensation, performance,
training, and event histories existed. Several apparent associations
were therefore created by censoring and exposure duration rather than
by the termination rule.

Version 2 first creates potential histories without knowledge of the
future outcome. The monthly hazard then uses only records available
before each risk month.

### Leadership

Version 1 made all 842 leaders permanently active.

Version 2 creates additional team-manager capacity and permits
team-manager attrition. Department heads and senior managers remain
protected structural levels in this version. When a team manager exits,
surviving reports receive a valid same-department reassignment and a
dated manager-change event.

## Generator Acceptance Results

The final Version 2 reference run produces:

- 2,591 cumulative terminations
- 71.17% voluntary termination share
- 276 team-manager exits
- 1,913 manager reassignments
- 803 positive cases in the 2025 twelve-month target window
- 11.91% next-twelve-month positive rate

These ranges validate the synthetic data-generating process. They are not
performance targets for the later classifier:

| Check | Accepted range | Version 2 |
|---|---:|---:|
| Twelve-month positive rate | 7%–13% | 11.91% |
| Positive cases | 500–1,100 | 803 |
| Voluntary share | 65%–80% | 71.17% |
| Development-only generator ROC-AUC diagnostic | 0.62–0.75 | 0.6298 |
| Maximum numeric target correlation | At most 0.40 | 0.1292 |
| Maximum large-group rate ratio | At most 2.00 | 1.3611 |

## Detectable but Imperfect Signal

A simple five-fold Logistic Regression diagnostic produces:

| Metric | Result |
|---|---:|
| Mean ROC-AUC | 0.6298 |
| ROC-AUC standard deviation | 0.0239 |
| Mean PR-AUC | 0.1929 |
| No-skill PR-AUC | 0.1191 |
| PR-AUC lift over no-skill | 1.62 |

This is deliberately not the project's final model evaluation. Its only
purpose is to reject a generator with either nearly random or unrealistically
easy observable signal. Formal temporal splitting, validation, model
selection, calibration, and policy evaluation occur later.

The frozen final model subsequently produced `0.6061` ROC-AUC on the once-only
2025 temporal test, below the generator diagnostic band. That result was
retained without post-test retuning. See
`docs/synthetic_signal_governance.md` for the full chronology and
methodological safeguards.

The strongest individual numeric relationship is recent performance,
with an absolute target correlation of approximately 0.129. No
individual feature nearly determines the outcome.

## Temporal Integrity

Automated checks confirm:

- Zero compensation records after termination
- Zero performance reviews after termination
- Zero training starts or completions after termination
- Zero employee events after termination
- Exactly one termination event per terminated employee
- No termination events for active employees
- No active employee reporting to an inactive manager
- No cross-department current manager assignments
- All snapshot feature dates on or before 2025-06-30
- One hazard outcome for every employee

## Reproducibility

Checkpoint 37 writes SHA-256 fingerprints for the regenerated raw
tables, hazard outcomes, and monthly diagnostics. Repeating the pipeline
with the same configuration and random seed should reproduce the same
fingerprints.

The fingerprints are stored in generated output:

```text
data/processed/v2_validation/reproducibility_hashes.csv
```

## Generated Validation Outputs

The complete validation package is written to:

```text
data/processed/v2_validation/
```

Important files include:

- `validation_summary.csv`
- `v1_v2_comparison.csv`
- `snapshot_summary.csv`
- `numeric_signal_checks.csv`
- `categorical_signal_checks.csv`
- `diagnostic_model_summary.csv`
- `diagnostic_model_folds.csv`
- `monthly_hazard_summary.csv`
- `feature_source_cutoffs.csv`
- `reproducibility_hashes.csv`

These generated files are excluded from Git. The validation code,
configuration, notebook, and this methodology document are tracked.

## Limitations

- The data remains synthetic and does not describe a real employer.
- Department assignments remain static in the present generator.
- Department heads and senior managers are structurally protected.
- The diagnostic cross-validation is not a substitute for the later
  temporal holdout design.
- The coefficient values are design assumptions rather than causal
  estimates.
- Passing these checks supports internal consistency, not real-world
  validity.
- The `0.62–0.75` band is a synthetic-design choice, not an external industry
  benchmark or a promise about final-model performance.
