# Employee Survival Analysis

## Purpose

Checkpoint 55 adds a censoring-aware view of employee retention over tenure.
It answers a different question from the primary classifier:

- The primary model asks whether an employee active at a snapshot will leave
  during the next 12 months.
- Survival analysis asks how the probability of remaining employed changes as
  time since hire accumulates.

The extension uses the same deterministic synthetic Version 2 history, but it
does not replace the classifier, reopen the once-only final test, modify the
frozen intervention policy, change the dashboard, or generate a new employee
review list.

## Why Censoring Matters

At the 2026-06-30 analytical boundary:

- 2,591 model-eligible employees have a known termination date; and
- 7,305 model-eligible employees are still active.

An active employee is not a negative event and should not be assigned an
invented future exit date. Instead, that employee is **right-censored**: the
pipeline knows the person remained employed through the as-of date but does
not know the eventual exit time.

Ignoring those 7,305 incomplete observations would overstate turnover and
discard most of the available tenure information. Survival methods use both
observed exits and censored durations correctly.

## Cohort Construction

The script builds one internal row per synthetic employee.

| Field | Definition |
| --- | --- |
| Start date | Employee hire date |
| Event date | Known termination date |
| Censor date | 2026-06-30 for an active employee |
| Observation end | Event date when terminated; censor date otherwise |
| Event indicator | 1 for a known termination; 0 for right-censoring |
| Duration | Inclusive calendar days from hire through observation end, divided by 30.4375 |

The all-employee data-quality cohort contains 10,000 unique employees. The
modeled cohort contains 9,896 Individual Contributors and Team Managers.
Eight Department Heads and 96 Senior Managers remain visible in the workforce
count but are excluded from the Cox model because their synthetic termination
process is structurally protected.

No row-level cohort is saved. The construction occurs in memory and only
aggregate tables and figures are written.

## Baseline-at-Hire Feature Boundary

Time-to-event models are vulnerable to another form of leakage: using a later
state as though it were known at hire. For example, the current department,
location, job role, or salary may reflect a promotion or transfer that
happened years after employment began.

Checkpoint 55 therefore reconstructs baseline fields from historical source
records:

- department, location, and job role come from the employee's Hire event note;
- base salary comes from the compensation row whose reason is `Hire`;
- salary position divides that hire salary by the midpoint of the original
  role's salary band; and
- age at hire uses the hire year and synthetic birth year.

Later compensation changes, performance reviews, training, promotions,
transfers, manager changes, leaves, current model scores, and review-selection
flags are not Cox features.

## Kaplan–Meier Retention Estimates

The Kaplan–Meier estimator updates the probability of remaining employed at
each observed exit while accounting for how many employees are still under
observation immediately beforehand.

The overall synthetic estimates are:

| Tenure horizon | Estimated retention | Cumulative attrition |
| ---: | ---: | ---: |
| 6 months | 94.56% | 5.44% |
| 12 months | 89.53% | 10.47% |
| 24 months | 79.74% | 20.26% |
| 36 months | 71.80% | 28.20% |
| 48 months | 64.54% | 35.46% |
| 60 months | 57.51% | 42.49% |

The estimated median retention time is not reached. That does not mean
employees never leave. It means the Kaplan–Meier curve remains above 50%
through the longest supported follow-up period.

The full step curve and 95% confidence interval are saved in
`survival_overall_curve.csv`. The compact horizon table is saved in
`survival_horizon_summary.csv`.

## Descriptive Group Curves

Separate Kaplan–Meier summaries are produced for:

- employment type at hire;
- department at hire;
- job level at hire; and
- region at hire.

Omnibus multivariate log-rank tests evaluate whether the full curves differ
across groups. Every configured dimension has a small p-value in this
synthetic run.

That result means the generated tenure distributions are detectably different.
It does not show that a group label causes attrition. Large samples can also
make small differences statistically detectable, and group labels can stand
in for correlated job, compensation, or organizational structure.

The group tables are descriptive evidence. They are not fairness verdicts,
policy thresholds, or instructions to intervene.

## Cox Proportional-Hazards Model

### Model form

The Cox model represents the event hazard as:

\[
h(t \mid x) = h_0(t)\exp(x^\top\beta)
\]

where:

- \(h_0(t)\) is an unspecified baseline hazard over tenure;
- \(x\) contains baseline-at-hire covariates; and
- \(\exp(\beta)\) is the adjusted hazard ratio.

The implementation uses `lifelines.CoxPHFitter` with Breslow baseline-hazard
estimation, Efron tie handling, and an L2 penalizer of 0.05.

### Features

Two numeric features are standardized:

- age at hire; and
- salary position at hire.

Four categorical features use reference coding:

| Feature | Reference |
| --- | --- |
| Employment type | Hourly |
| Education level | Associate |
| Department at hire | Customer Support |
| Job level at hire | Level 1 |

The full-rank matrix contains 17 encoded covariates. A categorical hazard
ratio compares one level with its reference while adjusting for the other
included features. A numeric hazard ratio represents a one-standard-deviation
increase.

### Reading a hazard ratio

- A hazard ratio above 1 indicates a higher modeled instantaneous exit rate.
- A hazard ratio below 1 indicates a lower modeled instantaneous exit rate.
- A confidence interval crossing 1 means the adjusted direction is not
  estimated precisely enough to exclude no association at the 95% level.

For salary position at hire, the fitted hazard ratio is approximately 0.838
per standard-deviation increase, with a 95% interval of about 0.804 to 0.873.
This is a synthetic adjusted association. It is not evidence that changing
salary alone would reduce a real employee's exit risk by a corresponding
amount.

Most categorical intervals cross 1 after adjustment, even though unadjusted
group curves differ. That is expected: the Cox model separates overlapping
baseline factors, applies shrinkage, and answers a different question from an
unadjusted log-rank test.

## Out-of-Fold Concordance

Five stratified folds fit the Cox model on four-fifths of the employees and
score the held-out fifth. Each employee appears once in validation, and no row
is used for fitting and validation in the same fold.

| Fold | Concordance |
| ---: | ---: |
| 1 | 0.5809 |
| 2 | 0.5719 |
| 3 | 0.5660 |
| 4 | 0.5707 |
| 5 | 0.5795 |
| Mean | 0.5738 |

Concordance is the probability that, for a comparable employee pair, the
person with the higher modeled hazard exits sooner. A value of 0.5 is random
ordering and 1.0 is perfect ordering. The result therefore shows modest
baseline signal, consistent with the deliberately noisy synthetic generator.

This cross-validation is a robustness diagnostic for the separate survival
extension. It is not a replacement for the primary classifier's chronological
train, validation, and once-only final-test design.

## Proportional-Hazards Diagnostic

The Cox model assumes covariate hazard ratios remain proportional over tenure.
Checkpoint 55 runs the rank-time Schoenfeld-residual diagnostic for every
encoded covariate using a precommitted p-value flag of 0.01.

Sixteen of 17 encoded features do not trigger the review flag. Standardized
salary position at hire does trigger it with p approximately 0.0048.

The flag is reported, not hidden or tuned away. It means the salary-position
association may vary with tenure, so its single hazard ratio should be read as
an average adjusted association rather than an unchanging effect. A real
follow-up study could investigate a time interaction, stratification,
time-varying covariates, or a different model.

No model or policy is retuned in response to this post hoc diagnostic.

## Generated Outputs

Checkpoint 55 writes aggregate artifacts beneath
`data/processed/survival_analysis/`:

| Artifact | Purpose |
| --- | --- |
| `survival_population_summary.csv` | Cohort, censoring, duration, and concordance summary |
| `survival_overall_curve.csv` | Full Kaplan–Meier step curve and confidence interval |
| `survival_horizon_summary.csv` | Retention estimates at six committed horizons |
| `survival_group_summary.csv` | Descriptive group retention estimates |
| `survival_logrank_tests.csv` | Omnibus group-curve comparisons |
| `cox_feature_manifest.csv` | Encoded-to-raw feature mapping and references |
| `cox_hazard_ratios.csv` | Adjusted hazard ratios and confidence intervals |
| `cox_cross_validation.csv` | Five held-out concordance results |
| `cox_proportional_hazards_test.csv` | Per-feature assumption diagnostics |
| `survival_validation.csv` | Complete Checkpoint 55 validation contract |

Four figures are produced:

- `overall_survival_curve.png`;
- `survival_by_employment_type.png`;
- `survival_by_hire_department.png`; and
- `cox_hazard_ratios.png`.

## Validation

The checkpoint validates:

- unique and complete employee coverage;
- exact model-eligible hierarchy levels;
- observed-exit and right-censoring counts;
- event, termination-date, and employment-status reconciliation;
- chronological start and observation-end dates;
- positive finite durations;
- complete hire-event and hire-compensation reconstruction;
- bounded, nonincreasing Kaplan–Meier estimates;
- complete horizons, groups, and log-rank tests;
- no direct leakage in the Cox feature matrix;
- unique finite encoded features and hazard ratios;
- complete disjoint cross-validation folds;
- plausible modest held-out concordance;
- complete proportional-hazards diagnostics;
- aggregate-only saved artifacts;
- isolation from the primary model, test target, policy, and dashboard; and
- human-review and noncausal governance.

The full automated suite adds unit tests for hire-note parsing, censoring,
inclusive durations, baseline salary position, Kaplan–Meier behavior, horizon
ordering, reference encoding, numeric standardization, and isolation rules.

## Reproduction

Install the new survival dependency once:

```powershell
python -m pip install --no-build-isolation --require-hashes -r requirements-lock.txt
```

Run the checkpoint:

```powershell
.\scripts\run_checkpoint55.ps1
```

The runner compiles source and tests, runs Ruff, checks test formatting,
regenerates the aggregate survival artifacts and figures, and executes the
complete test suite.

## Governance and Limitations

- All employees, events, compensation, groups, and results are synthetic.
- The generator intentionally creates observable attrition associations.
- Censoring is treated as noninformative, which may not hold in real systems.
- Hire-time features omit later changes instead of modeling time-varying
  covariates.
- The Cox model assumes proportional hazards; one feature is transparently
  flagged for review.
- Log-rank p-values can detect small differences in a large synthetic sample.
- Hazard ratios are adjusted associations, not causal intervention effects.
- The model is not an individual probability calculator or employment-action
  system.
- No employee-level survival record is exported.
- No real-data validation, legal review, privacy review, or deployment
  monitoring is performed.

For real use, an organization would need consent and lawful-purpose review,
data minimization, missingness and censoring analysis, external validation,
time-varying feature design, fairness assessment, monitoring, and independent
human governance.

## Method References

- [lifelines KaplanMeierFitter documentation](https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html)
- [lifelines CoxPHFitter documentation](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html)
- [lifelines proportional-hazard assumption documentation](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html)
