# Retention Intervention Policy Analysis

## Purpose

Checkpoint 46 converts the calibrated attrition model and Checkpoint 45 cost
assumptions into a constrained decision:

> Which employees should receive limited retention resources, and what is
> the expected financial return under explicit assumptions?

This is the first checkpoint that evaluates the previously reserved 2025
test period. The policy is selected and cryptographically hashed using the
2024 validation period **before** the test target is accessed. After the test
is evaluated once, the policy is not changed.

## Results Snapshot

The frozen policy selected by validation was **Budget-constrained expected
value**. It ranked employees by estimated economic value while enforcing the
committed 700-person capacity and $1.75 million budget.

| Result | 2024 validation | 2025 final test |
|---|---:|---:|
| Population | 5,521 | 6,641 |
| Observed attrition cases | 586 | 803 |
| Employees selected | 700 | 700 |
| Selected attrition cases | 85 | 137 |
| Precision among selected employees | 12.14% | 19.57% |
| Attrition cases captured | 14.51% | 17.06% |
| Intervention spend | $1,750,000 | $1,750,000 |
| Predicted expected net value | $840,974 | $1,390,488 |
| Outcome-aligned net value | $512,125 | $2,033,700 |

The final-test outcome-aligned net-value interval from 500 employee-level
bootstrap resamples was **$1,450,673 to $2,587,584**. The corresponding
capture-rate interval was **14.45% to 19.30%**.

The selected policy identified 137 of the 803 employees who subsequently
terminated while staying within the frozen capacity. Its 19.57% precision
was higher than the final-test population attrition rate of 12.09%, producing
a lift of approximately 1.62.

These dollar values are scenario calculations, not realized savings. No
intervention was performed. Outcome-aligned value uses observed departures
with the assumed 25% effectiveness and salary-based replacement cost.

## Beginner Summary

The model supplies an estimated twelve-month attrition probability.
Checkpoint 45 supplies:

- Replacement cost
- Intervention cost
- Intervention effectiveness

Checkpoint 46 combines them:

\[
\text{ExpectedNetValue}_i
=
p_i
\times
\text{InterventionSuccess}
\times
\text{ReplacementCost}_i
-
\text{InterventionCost}
\]

An employee with higher attrition probability or higher salary-dependent
replacement cost can have greater predicted expected value from an
intervention.

## Reference Assumptions

The selected policy uses the Checkpoint 45 reference scenario:

| Assumption | Value |
|---|---:|
| Replacement cost | 100% of base salary |
| Intervention cost | $2,500 per selected employee |
| Intervention effectiveness | 25% |
| Maximum intervention capacity | 700 employees |
| Maximum budget | $1,750,000 |

The budget equals:

\[
700 \times \$2{,}500 = \$1{,}750{,}000
\]

These are synthetic scenario assumptions, not estimates from Tesla or
another real company.

## Compared Policies

### 1. No intervention

Selects nobody. This provides a zero-spend comparison.

### 2. Fixed 0.50 threshold

Selects employees whose calibrated probability is at least 50%. This was the
old project rule and is retained only as a comparator.

### 3. Validation cost-optimal threshold

Orders validation employees by probability and selects the global cutoff
that maximizes cumulative predicted expected net value. The cutoff is frozen
before test evaluation.

This policy is not capacity constrained and can select substantially more
than 700 employees.

### 4. Top 10% probability

Selects the highest-risk 10% of each evaluation population. It corresponds
to the ranking analysis from Checkpoint 43.

### 5. Top 700 probability

Selects the 700 employees with the highest calibrated probabilities,
regardless of salary-dependent replacement cost.

### 6. Budget-constrained expected value

Ranks employees by:

\[
\text{PredictedExpectedNetValue}_i
\]

It then selects employees while:

- Expected net value remains positive
- The number selected does not exceed 700
- Total spend does not exceed $1,750,000

This is the selected policy because it directly answers the committed
limited-resource question.

## Leakage-Safe Sequence

The implementation follows this order:

1. Load 2024 validation probabilities and non-target cost features.
2. Calculate the validation cost threshold.
3. Define every policy rule.
4. Freeze the policy in a JSON-compatible record.
5. Create a SHA-256 hash of that record.
6. Only then load the test outcomes.
7. Refit the selected Logistic Regression on train plus validation history.
8. Learn sigmoid calibration from employee-disjoint out-of-fold scores.
9. Score the 2025 test population.
10. Apply the unchanged policies once.
11. Report results without retuning.

Repeated employee histories stay in the same calibration fold, so one
employee cannot appear in both the fit and calibration sides of one fold.

## Expected Versus Outcome-Aligned Value

Two financial quantities are reported.

### Predicted expected value

This uses the model probability:

\[
\sum_{i \in S}
\left(
p_i
\times
q
\times
R_i
-
C
\right)
\]

where:

- \(S\) is the selected employee set
- \(p_i\) is calibrated attrition probability
- \(q\) is assumed intervention effectiveness
- \(R_i\) is salary-dependent replacement cost
- \(C\) is intervention cost

### Outcome-aligned value

This replaces model probability with the observed test outcome:

\[
\sum_{i \in S}
\left(
y_i
\times
q
\times
R_i
-
C
\right)
\]

This answers whether the frozen ranking identified employees who later
terminated, under the assumed effectiveness and replacement cost.

It is **not** observed intervention profit. No intervention was actually
performed, and no causal treatment effect was measured.

## Final Model Evaluation

The final test model is:

- Base model: Logistic Regression
- Calibration: Sigmoid
- Final fitting data: 2023 train plus 2024 validation
- Test period: 2025-06-30 snapshot predicting through 2026-06-30

After final test evaluation, the model is refitted on all three historical
periods to score the current active population as of 2026-06-30.

## Current Workforce Plan

The generated current plan contains:

- Employee ID
- Department
- Organizational level
- Employment type
- Job level
- Base salary
- Calibrated probability
- Replacement-cost estimate
- Predicted avoidable cost
- Predicted expected net value
- Expected-value rank
- Human-review selection flag

Names are intentionally excluded.

The plan is an anonymized synthetic portfolio output. Every selected row is
marked for human review. Automatic employment action is prohibited.

The current 2026 plan applies the unchanged policy to 7,305 eligible active
employees. It selects 700 for human review, projects 35.03 prevented
departures, $4.146 million of avoided cost, and $2.396 million of expected
net value under the reference scenario. Because outcomes after 2026-06-30
are unknown, these values are projections rather than another performance
test.

## Fairness Boundary

Checkpoint 44 found substantial descriptive differences across department,
employment type, region, job level, and organizational level.

Expected-value ranking can also favor employees with higher salaries because
their assumed replacement costs are larger. Therefore:

- Financial value must not be treated as employee value.
- The plan must not be used for termination, compensation, or promotion
  decisions.
- Department and subgroup selection rates must be reviewed.
- A real deployment would require Legal, HR, and fairness governance.
- Alternative allocation constraints should be examined before operational
  use.

Checkpoint 46 reports department selection rates so this concentration
remains visible.

## Scenario Sensitivity

The frozen employee selections are revalued under all 27 Checkpoint 45 cost
scenarios:

- 3 replacement-cost assumptions
- 3 intervention-cost assumptions
- 3 effectiveness assumptions

The selected employees are not changed during sensitivity analysis. This
shows how conclusions depend on economic assumptions without creating 27
different test-tuned policies.

## Bootstrap Uncertainty

The selected frozen policy is evaluated in 500 stratified test bootstraps.
The analysis reports 95% intervals for:

- Outcome-aligned net value
- Attrition capture rate

The policy is not reselected inside the bootstrap.

## Output Files

| File | Purpose |
|---|---|
| `retention_policy_comparison.csv` | Validation and test comparison of six policies |
| `retention_policy_threshold_analysis.csv` | Validation-only cost-threshold search |
| `retention_policy_decision.csv` | Frozen policy, hash, and final results |
| `retention_final_test_predictions.csv` | Once-only final test scores |
| `retention_final_test_model_metrics.csv` | Final probability and ranking metrics |
| `retention_final_calibration_folds.csv` | Employee-disjoint calibration audit |
| `retention_policy_sensitivity.csv` | All 27 cost scenarios |
| `retention_policy_test_bootstrap.csv` | Final policy uncertainty |
| `current_retention_policy_scores.csv` | Current anonymized human-review plan |
| `current_retention_policy_summary.csv` | Current projected policy value |
| `current_retention_policy_group_summary.csv` | Department selection concentration |
| `retention_policy_validation.csv` | Checkpoint validation checks |

The figures are written to:

```text
data/processed/policy_figures/
```

## Run Command

```powershell
.\scripts\run_checkpoint46.ps1
```

## Limitations

1. The workforce and outcomes are synthetic.
2. Intervention success is assumed, not experimentally estimated.
3. Replacement-cost multipliers are scenarios, not accounting facts.
4. Outcome-aligned value is not realized profit.
5. The model is predictive, not causal.
6. Salary-dependent value can create distributional concerns.
7. Current scores have no observable future outcome.
8. A single synthetic test period cannot establish production stability.

The correct portfolio claim is that the project demonstrates transparent,
leakage-aware decision analysis under explicit assumptions—not that it proves
a retention intervention will succeed in a real workforce.

## Evidence and Reproduction

- Executed analysis:
  [Notebook 28](../notebooks/28_retention_policy_analysis.ipynb)
- Curated reviewer narrative:
  [Fairness, Economics, and Retention Policy](../notebooks/portfolio/03_fairness_economics_and_policy.ipynb)
- Implementation:
  [optimize_retention_policy.py](../src/optimize_retention_policy.py)
- Primary generated evidence: `data/processed/retention_policy_decision.csv`,
  `data/processed/retention_policy_comparison.csv`, and
  `data/processed/retention_final_test_model_metrics.csv`

Reproduce the policy analysis from the project root:

```powershell
.\scripts\run_checkpoint46.ps1
```
