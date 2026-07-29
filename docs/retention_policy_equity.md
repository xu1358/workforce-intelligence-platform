# Retention Policy Salary-Allocation Equity Audit

## Purpose

This audit examines a limitation that model-level fairness analysis cannot
answer:

> Does the salary-based expected-value policy allocate limited retention
> resources disproportionately toward higher-paid employees?

The answer is **yes**. The effect is structural because the frozen policy
ranks employees using:

\[
\text{ExpectedNetValue}_i
=
p_i
\times 0.25
\times \text{BaseSalary}_i
- \$2{,}500
\]

Holding probability constant, a higher salary always produces a higher
ranking score. The policy is therefore financially optimized, not
equity-neutral.

This audit quantifies that mechanism. It does not access outcome columns,
does not change the frozen policy, and does not make a binary legal or
fairness verdict.

## Results Snapshot

| Population | Eligible mean salary | EV-selected mean salary | Difference | Ratio |
|---|---:|---:|---:|---:|
| 2024 validation | $92,810 | $121,964 | +$29,155 | 1.314 |
| 2025 final test | $94,467 | $106,393 | +$11,926 | 1.126 |
| 2026 current | $95,722 | $125,670 | +$29,948 | 1.313 |

The salary difference appears in all three periods. The 2026 selected
population earns 31.3% more on average than the eligible current population.

### Current selection by salary quintile

| Salary quintile | Eligible employees | Selected | Selection rate |
|---|---:|---:|---:|
| Lowest 20% | 1,461 | 12 | 0.82% |
| Second 20% | 1,461 | 70 | 4.79% |
| Middle 20% | 1,461 | 81 | 5.54% |
| Fourth 20% | 1,461 | 227 | 15.54% |
| Highest 20% | 1,461 | 310 | 21.22% |

The highest salary quintile is selected **25.83 times** as often as the
lowest salary quintile.

### Manufacturing finding

The current Manufacturing population contains:

| Manufacturing measure | Value |
|---|---:|
| Eligible employees | 1,808 |
| Employees selected | 162 |
| Eligible mean salary | $88,464 |
| Selected mean salary | $108,935 |
| Selected salary difference | +$20,470 |

The selected Manufacturing group earns approximately 23.1% more than the
eligible Manufacturing population.

## The Effect Is Not Explained by Risk Alone

The current frozen policy is compared with the already-defined top-700
probability policy:

| Current policy | Selected mean salary | Mean probability | Predicted net value under original economics |
|---|---:|---:|---:|
| Frozen expected value | $125,670 | 20.02% | $2,396,216 |
| Top 700 probability | $82,995 | 23.75% | $1,666,616 |

Only 321 employees overlap. The salary-based ranking substitutes **379
employees** for people who would have been selected by probability alone.

This is direct evidence that salary changes allocation rather than merely
describing the same high-risk population.

Within the current highest risk quintile:

- 100% of employees in the highest salary quintile are selected.
- 2.56% of employees in the lowest salary quintile are selected.

Coarse risk stratification does not remove the salary-selection difference.

## Why Checkpoint 44 Did Not Detect This

Checkpoint 44 evaluates the model score before economic optimization:

```text
Employee features → attrition probability → model fairness diagnostics
```

The salary mechanism enters later:

```text
Attrition probability × salary-based replacement cost → policy selection
```

Prediction fairness and allocation equity are related but different:

- **Prediction fairness** asks whether probabilities, calibration, and errors
  differ across groups.
- **Allocation equity** asks who receives scarce supportive resources after
  the model is combined with costs, capacity, and a ranking objective.

Adding job level to model fairness is not sufficient. Salary varies within
job level, and salary is used directly by the policy.

## Salary Groups Audited

The audit reports:

1. Fixed salary bands from under $60,000 through $150,000 and above.
2. Population salary quintiles.
3. Salary quintiles within each job level.
4. Salary quintiles within probability quintiles.
5. Department-level selected-versus-eligible salary differences.

Within-job-level comparisons help distinguish direct pay prioritization from
the broader effect of seniority.

## Sensitivity Policies

Four policies receive the same 700-person capacity and $1.75 million budget:

### Frozen expected-value policy

Uses the original salary-dependent expected-value score. This is the tested
Version 2 policy and remains unchanged.

### Top 700 probability

Ranks only by calibrated attrition probability. It removes salary from
selection but reduces predicted value under the original financial objective.

### Salary-capped expected value

Caps replacement cost at **$111,000**, the validation-period 75th salary
percentile. The cap is frozen using validation salaries only.

For the current population, this policy produces:

- Selected mean salary: $110,734
- Mean selected probability: 21.26%
- Predicted net value under original economics: $2,251,288
- Overlap with frozen policy: 568 of 700

It reduces the selected mean salary by approximately $14,935 while retaining
approximately $2.25 million of predicted value under the original assumptions.

### Constant replacement cost

Uses **$86,400**, the validation median salary, for every employee's ranking
score. Because cost is constant, this produces the same ordering as
probability-only selection.

## Interpreting the Tradeoff

The original policy answers:

> Which employees maximize salary-based predicted financial return?

It does not answer:

> How should supportive retention resources be distributed equitably?

Those objectives can conflict. The current sensitivity comparison shows:

- Salary weighting increases modeled financial value.
- Salary weighting substantially increases selected salary.
- Probability-only selection targets higher modeled risk.
- Salary caps can reduce pay concentration while preserving much of the
  modeled financial value.

The project should not claim that one objective is universally correct. A
real organization would need an approved multi-objective policy that balances
financial impact, operational need, access to supportive resources, subgroup
effects, and legal or ethical requirements.

## Review Triggers

The committed diagnostic triggers are:

| Diagnostic | Review trigger |
|---|---:|
| Selected-to-population mean salary ratio | Greater than 1.10 |
| Highest-to-lowest salary-quintile selection-rate ratio | Greater than 2.0 |
| Highest-risk-quintile salary selection gap | Greater than 10 percentage points |

All three periods trigger review. These are portfolio diagnostic thresholds,
not legal standards and not proof that the policy is unlawful or unfair.

## Governance Decision

This audit does not retroactively select a new policy. Changing the policy
after inspecting the once-only final-test period would violate the project's
no-post-test-retuning rule.

The correct governance sequence is:

1. Preserve the frozen Version 2 policy and its historical evaluation.
2. Report the salary-allocation limitation explicitly.
3. Treat probability-only, salary-capped, and constant-cost policies as
   sensitivity analyses.
4. Define any replacement policy before accessing a new outcome period.
5. Evaluate it using a new holdout or prospective evaluation.

All selections remain supportive human-review candidates. Automatic
employment action is prohibited.

## Outcome and Privacy Boundary

The audit reads:

- Employee ID temporarily for deterministic ranking
- Snapshot date
- Department
- Organizational level
- Employment type
- Job level
- Base salary
- Calibrated attrition probability
- Existing current selection flag for reconciliation

It does not load:

- `actual_attrition`
- `attrition_next_12m`
- Termination dates
- Termination types

Employee-level audit rows are not saved. Every committed output is aggregate.

## Outputs

Checkpoint 56 creates:

| Output | Purpose |
|---|---|
| `policy_equity_population_summary.csv` | Headline salary concentration and review flags |
| `policy_equity_salary_band_metrics.csv` | Selection by absolute, relative, and within-level pay bands |
| `policy_equity_policy_comparison.csv` | Value and salary tradeoffs across policies |
| `policy_equity_risk_salary_intersections.csv` | Salary selection within risk quintiles |
| `policy_equity_department_summary.csv` | Department salary concentration |
| `policy_equity_frozen_thresholds.csv` | Validation-frozen salary cap and constant cost |
| `policy_equity_validation.csv` | Scope, reconciliation, and governance checks |

Figures are saved under:

```text
data/processed/policy_equity/figures/
```

## Evidence and Reproduction

Run from the project root:

```powershell
.\scripts\checkpoints\run_checkpoint56.ps1
```

The implementation is
[audit_retention_policy_equity.py](../src/audit_retention_policy_equity.py),
and the executed reviewer artifact is
[Notebook 34](../notebooks/34_retention_policy_equity.ipynb).

## Limitations

1. Salary is not itself a protected-class field, but it can reflect or proxy
   organizational and historical inequities.
2. The data, salaries, outcomes, and cost assumptions are synthetic.
3. Equal allocation rates are not automatically the correct policy goal.
4. The replacement-cost multiplier and intervention effect are assumed.
5. Salary quintiles are relative to each audited population.
6. Risk-quintile comparisons are coarse and do not prove causal treatment
   differences.
7. The audit cannot evaluate protected attributes absent from the schema.
8. A policy alternative requires new evidence before operational adoption.
