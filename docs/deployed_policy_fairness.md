# Deployed Retention Policy Fairness

## Purpose

This audit answers a different question from model-probability fairness:

> Which subgroups are selected when the project applies the exact frozen
> budget-constrained expected-value policy?

Checkpoint 44 used a global top-10% probability cutoff as a model-ranking
diagnostic. That was never the deployed rule. Checkpoint 46 instead ranks
employees using expected net value, which depends on both calibrated attrition
probability and salary-based replacement cost:

\[
\text{expected net value}
=
P(\text{attrition})
\times
\text{replacement cost}
\times
\text{assumed intervention success}
-
\text{intervention cost}.
\]

Because the rules select different employees, their subgroup selection rates
must not be interchanged.

## Exact implementation boundary

The audit does not copy or approximate the policy formula. It:

1. reads the frozen Checkpoint 46 decision;
2. imports `policy_flags()` from `src/optimize_retention_policy.py`;
3. applies the saved `budget_expected_value` rule to the original 2024
   validation probabilities and salaries;
4. reconciles the resulting count and selected mean salary to
   `retention_policy_comparison.csv`; and
5. reports only aggregate subgroup evidence.

This shared-function design prevents the fairness implementation from drifting
away from the deployed selection implementation.

## Why the correction matters

The probability-only diagnostic selected 553 employees. The frozen
expected-value rule selected 700. Only 184 employees appeared in both sets, a
Jaccard overlap of 17.21%.

The rules therefore produce substantially different subgroup stories:

| Subgroup | Probability-only top 10% | Frozen expected-value policy |
| --- | ---: | ---: |
| Customer Support | 44.61% | 1.25% |
| Engineering | 1.21% | 24.02% |
| Hourly | 26.43% | 0.67% |
| Salaried | 6.87% | 14.98% |
| Job level 1 | 25.16% | 0.63% |
| Job level 3 | 2.77% | 19.21% |

These corrected rates reverse several of the most visible comparisons. The
old numbers remain useful as a probability-ranking diagnostic, but they do not
describe the policy allocation.

## Overall validation-period comparison

| Metric | Frozen expected-value policy | Probability-only top 10% |
| --- | ---: | ---: |
| Selected employees | 700 | 553 |
| Selected positive cases | 85 | 109 |
| Precision | 12.14% | 19.71% |
| Capture | 14.51% | 18.60% |
| Mean selected probability | 12.91% | 16.78% |
| Mean selected salary | $121,964 | $70,136 |

The expected-value rule is not intended to maximize precision or capture
alone. It optimizes the committed financial objective under capacity, budget,
and positive-value constraints. Its higher selected salary is the allocation
mechanism already investigated by the salary-equity audit.

## Population and outcome scope

The subgroup audit uses:

- the 5,521-row 2024 validation population;
- 586 known next-12-month attrition outcomes;
- the selected sigmoid-calibrated probabilities;
- the frozen Checkpoint 46 decision; and
- age band, education, region, employment type, department, job level, and
  organizational level.

The reserved 2025 final-test target is not read. The final test is not
reopened, and current future outcomes remain unknown.

Validation outcomes support descriptive precision, true-positive-rate, and
false-positive-rate comparisons. They do not establish that the policy causes
different outcomes.

## Saved evidence

The audit writes aggregate files to
`data/processed/retention_policy_fairness/`:

- `deployed_policy_group_metrics.csv`
- `deployed_policy_disparities.csv`
- `selection_rule_comparison.csv`
- `selection_rule_overall.csv`
- `validation_checks.csv`
- three aggregate PNG figures

No employee identifiers, employee-level selection flags, or outcome rows are
exported by this checkpoint.

## Governance

This checkpoint:

- does not change the selected model;
- does not recalibrate probabilities;
- does not change any employee selection;
- does not change the frozen policy;
- does not reopen the final test;
- does not change dashboard data;
- does not assign a binary fair/unfair verdict; and
- does not authorize automated employment action.

The results are descriptive evidence about a fictional workforce. A different
selection policy would require new holdout or prospective evaluation.

## Reproduction

Run the focused audit:

```bash
python src/audit_retention_policy_fairness.py
```

Or run the historical checkpoint wrapper on Windows:

```powershell
.\scripts\checkpoints\run_checkpoint64.ps1
```

Successful execution ends with:

```text
DEPLOYED RETENTION POLICY FAIRNESS AUDIT COMPLETED SUCCESSFULLY
```
