# Historical Assignment Attribution Analysis

## Purpose

This analysis tests whether historical workforce activity is incorrectly
assigned to an employee's final department or location. The original
hypothesis assumed that Version 1 transfer events recorded department
changes. Inspection showed that they record location changes instead.

The audit therefore answers two separate questions:

1. Does department-history attribution bias exist in Version 1?
2. How much location-history attribution changes when event-time location is
   reconstructed correctly?

## Results Snapshot

### Department history

| Validation result | Value |
|---|---:|
| Employees checked | 10,000 |
| Employees missing hire department | 0 |
| Hire-versus-current department mismatches | 0 |
| Department-transfer events | 0 |
| Department mismatch rate | 0.00% |

All 10,000 Version 1 employees retained their hire department. As a result,
using current department to group historical events did not create
department-history bias in this generated dataset.

### Location history

Transfer events changed location, so using final location to summarize all
past records reassigned historical activity:

| Historical domain | Total records | Records reassigned | Employees affected | Reassignment rate |
|---|---:|---:|---:|---:|
| Training | 26,433 | 1,299 | 553 | 4.91% |
| Compensation | 30,997 | 1,249 | 553 | 4.03% |
| Promotion | 2,146 | 74 | 70 | 3.45% |
| Leave | 964 | 32 | 18 | 3.32% |
| Performance reviews | 17,972 | 558 | 357 | 3.10% |
| Manager changes | 632 | 18 | 18 | 2.85% |
| Terminations | 1,713 | 0 | 0 | 0.00% |

The largest displayed aggregate relative shift was approximately 5.15%:
Buffalo training hours changed from 41,385 under historical attribution to
39,359 under current-location attribution. Austin training hours moved in
the opposite direction, from 88,648 to 90,898.

## Correct Interpretation

The result is not “no historical bias.” The precise conclusion is:

- There is no department-history mismatch because Version 1 generated no
  department transfers.
- There is measurable location-history attribution error when past events
  are assigned using each employee's final location.
- Reconstructing location as of the event date corrects that error.

This distinction matters because an apparently valid department audit could
miss a similar problem in another changing organizational dimension.

## Reconstruction Method

For an event dated \(t\):

1. Start from the employee's original location.
2. Sort that employee's transfer events chronologically.
3. Apply only transfers occurring on or before \(t\).
4. Attribute the event to the reconstructed location at \(t\).

Records before a transfer remain assigned to the previous location, while
records after the transfer use the new location. This prevents future
organizational information from rewriting historical aggregates.

## Why the Result Matters

Using current assignments for historical records can distort:

- Location-level training totals
- Compensation record counts and averages
- Review and promotion activity
- Operational comparisons across sites

The bias was modest in this synthetic dataset, but the audit demonstrates a
general temporal-data rule: group labels must be reconstructed at the event
date whenever those labels can change.

## Limitations

- Version 1 contains location transfers but no department transfers.
- The measured differences are synthetic and depend on the generated
  transfer frequency.
- Reassignment rates quantify attribution changes, not causal bias.
- The analysis does not imply that any location caused the measured
  employee outcomes.

## Evidence and Reproduction

- Executed analysis:
  [Notebook 19](../notebooks/19_department_history_bias.ipynb)
- Implementation:
  [analyze_department_history_bias.py](../src/analyze_department_history_bias.py)
- Generated evidence directory: `data/processed/department_history_bias/`

Reproduce the analysis from the project root:

```powershell
python src\analyze_department_history_bias.py
```
