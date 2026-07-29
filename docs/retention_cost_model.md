# Retention Cost Model

## Purpose

Checkpoint 45 translates an avoidable employee departure into transparent
financial and operational planning units. It is the economic-input layer
between the validation-stage model analysis and the policy optimization
planned for Checkpoint 46.

This checkpoint answers:

> Under a stated set of assumptions, what financial and operational
> disruption could one successful retention intervention avoid?

It does **not** answer:

- Which employee should receive an intervention?
- How many employees should be selected?
- Which probability cutoff should be used?
- Whether a particular employee will leave?

Those are policy questions. Checkpoint 46 will compare policies by combining
the cost assumptions with validation-stage probabilities.

## Results Snapshot

The base replacement, standard intervention, and base effectiveness
combination is the reference scenario used by the later policy analysis.

| Result | 2024 validation | Current active workforce |
|---|---:|---:|
| Model-eligible employees | 5,521 | 7,305 |
| Average base salary | $92,810 | $95,722 |
| Median replacement cost | $86,400 | $90,900 |
| Intervention cost per employee | $2,500 | $2,500 |
| Assumed intervention effectiveness | 25% | 25% |
| Median maximum avoidable cost | $21,600 | $22,725 |
| Median break-even attrition probability | 11.57% | 11.00% |

All **27 combinations** of replacement impact, intervention cost, and
effectiveness were calculated for both populations. Under the current
reference scenario, a median-salary employee must have an estimated
attrition probability of about 11.00% for the modeled expected benefit to
equal the $2,500 intervention cost.

The cost model did not load model probabilities, choose a threshold, rank
employees, or access reserved test outcomes. It established transparent
economic inputs for Checkpoint 46.

## Scope and Data Boundary

The calculation uses non-target features from two populations:

| Population | Snapshot | Purpose |
|---|---|---|
| 2024 validation | 2024-06-30 | Economic inputs for later policy validation |
| Current active | 2026-06-30 | Current workforce cost profile |

Only Individual Contributors and Team Managers are included because these
are the outcome-eligible levels defined by the Version 2 model policy.
Department Heads and Senior Managers remain outside the retention-model
population because the synthetic generator protects those levels from
termination.

The program explicitly reads only:

- Employee ID
- Snapshot date
- Organizational level
- Department
- Employment type
- Job level
- Base salary

It does not load:

- `attrition_next_12m`
- Termination dates or status
- Model probabilities
- Risk rankings

The reserved 2025 test outcomes therefore remain untouched.

## Scenario Structure

The model separates uncertainty into three dimensions instead of committing
to one optimistic estimate.

### Replacement impact

| Scenario | Salary multiple | Vacancy days | Ramp-up days | Training hours | Coverage hours |
|---|---:|---:|---:|---:|---:|
| Low | 50% | 30 | 60 | 40 | 60 |
| Base | 100% | 60 | 90 | 80 | 180 |
| High | 150% | 90 | 150 | 120 | 360 |

Coverage hours equal vacancy days multiplied by assumed daily coverage
hours.

### Intervention cost

| Scenario | Cost per targeted employee |
|---|---:|
| Lean | $1,000 |
| Standard | $2,500 |
| Intensive | $5,000 |

### Intervention effectiveness

| Scenario | Probability the intervention prevents departure |
|---|---:|
| Conservative | 10% |
| Base | 25% |
| Optimistic | 40% |

The full cross-product produces:

`3 replacement scenarios × 3 cost scenarios × 3 effectiveness scenarios = 27 scenarios`

No scenario is selected in Checkpoint 45.

## Financial Formulas

For employee \(i\) and replacement scenario \(s\):

\[
\text{ReplacementCost}_{i,s}
=
\text{BaseSalary}_{i}
\times
\text{SalaryMultiplier}_{s}
\]

For effectiveness assumption \(e\):

\[
\text{MaximumAvoidableCost}_{i,s,e}
=
\text{ReplacementCost}_{i,s}
\times
\text{InterventionSuccessProbability}_{e}
\]

This is the expected avoidable value **conditional on the employee otherwise
departing**. It does not yet include the model's estimated probability of
departure.

For intervention-cost assumption \(c\), the predicted attrition probability
required to break even is:

\[
\text{BreakEvenRisk}_{i,s,c,e}
=
\frac{
\text{InterventionCost}_{c}
}{
\text{ReplacementCost}_{i,s}
\times
\text{InterventionSuccessProbability}_{e}
}
\]

Checkpoint 46 will compare this break-even value with calibrated model
probabilities and evaluate complete intervention policies.

## Why Operational Units Stay Separate

The cost model reports:

- Vacancy days
- Time-to-productivity days
- Training hours per replacement
- Coverage or overtime hours

These measures are not automatically converted to dollars. Adding them to a
salary-based replacement multiplier without separately supported prices
could count the same disruption twice.

For example, a replacement-cost multiplier may already be intended to cover
some recruiting, onboarding, vacancy, and productivity losses. Reporting
operational units separately keeps that uncertainty visible.

## Reference Scenario

The reference scenario is:

- Base replacement impact: 100% of salary
- Standard intervention: $2,500 per targeted employee
- Base intervention effectiveness: 25%

The reference scenario is a reporting anchor, not a claim that these are the
correct real-world values.

## Output Files

The program creates:

| File | Purpose |
|---|---|
| `retention_cost_assumption_table.csv` | Readable list of assumptions |
| `retention_cost_scenarios.csv` | All 27 combinations for both populations |
| `retention_cost_population_summary.csv` | Salary and reference-cost summaries by department |
| `retention_cost_validation.csv` | Formula, scope, and boundary checks |

All files are written to `data/processed/` and can be reproduced by running:

```powershell
.\scripts\run_checkpoint45.ps1
```

## Interpretation Rules

### These outputs support

- Sensitivity analysis
- Budget planning
- Break-even analysis
- Comparison of intervention intensity
- Translation of model ranking into business units

### These outputs do not support

- A causal claim that an intervention will work
- A claim that an employee intends to leave
- Automatic employment decisions
- A real company's replacement-cost estimate
- A final threshold or contact list

## Synthetic-Data Limitation

The salaries, employment histories, and attrition outcomes are synthetic.
The cost assumptions are also scenario inputs chosen for portfolio analysis.
They are not Tesla estimates and are not derived from a confidential company
dataset.

In a real deployment, Finance, Operations, HR, and Legal stakeholders would
need to approve:

- Replacement-cost definitions
- Intervention types and costs
- Realistic intervention effectiveness ranges
- Operational capacity constraints
- Permitted uses of individual risk scores
- Measurement and experimentation plans

## Handoff to Checkpoint 46

Checkpoint 46 will use the validation-stage calibrated probabilities and
these economic assumptions to compare policies by:

1. Expected avoided replacement cost
2. Total intervention spend
3. Expected net value
4. Break-even behavior
5. Budget and top-k capacity
6. Sensitivity across all scenario combinations

Only after those comparisons will an operating policy be proposed. The
reserved 2025 test outcomes will remain protected until the planned final
evaluation stage.

## Evidence and Reproduction

- Executed analysis:
  [Notebook 27](../notebooks/27_retention_cost_model.ipynb)
- Curated reviewer narrative:
  [Fairness, Economics, and Retention Policy](../notebooks/portfolio/03_fairness_economics_and_policy.ipynb)
- Implementation:
  [calculate_retention_economics.py](../src/calculate_retention_economics.py)
- Primary generated evidence: `data/processed/retention_cost_scenarios.csv`,
  `data/processed/retention_cost_population_summary.csv`, and
  `data/processed/retention_cost_validation.csv`
