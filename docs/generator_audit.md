# Version 1 Termination Generator Audit

## Purpose

This audit explains how Version 1 generated employee termination outcomes
and checks whether the realized synthetic data matched those rules. It
separates intentional assumptions from artifacts that a predictive model
could learn.

No production generator logic was changed during this audit.

## Results Snapshot

| Audit result | Value |
|---|---:|
| Employees generated | 10,000 |
| Observed terminations | 1,713 |
| Expected terminations from generator probabilities | 1,710.25 |
| Observed attrition rate | 17.130% |
| Expected attrition rate | 17.103% |
| Difference between observed and expected rate | 0.027 percentage points |
| Voluntary terminations | 1,299 |
| Involuntary terminations | 414 |
| Individual contributors | 9,158 |
| Permanently active leaders | 842 |
| Individual contributors under 120 days | 528 |

The realized termination total was 2.75 employees above the probability
sum expected by the generator. This small difference is consistent with
random sampling and confirms that the overall rate behaved as designed.

## Generator Scope

The project contains two employee generators:

| Generator | Purpose | Employee count |
|---|---|---:|
| `src/generate_employees.py` | Small teaching and validation sample | 50 |
| `src/generate_full_workforce.py` | Main Version 1 portfolio workforce | 10,000 |

The production audit focuses on `generate_full_workforce.py`.

## Inputs to the Version 1 Termination Rule

The production termination function receives:

- Hire date
- Department ID
- Location ID

It does not directly receive:

- Performance history
- Compensation history or growth
- Promotion history
- Training participation
- Manager changes
- Transfer history
- Job level
- Employment type
- Education
- Age

Therefore, later associations between these excluded variables and
termination are indirect synthetic relationships created by workforce
structure or by the order in which records were generated.

## Eligibility and Tenure Rules

Only individual contributors were eligible for termination in Version 1.
Department heads, senior managers, and team managers were forced active.

| Organizational level | Employees | Terminations | Observed rate |
|---|---:|---:|---:|
| Individual Contributor | 9,158 | 1,713 | 18.71% |
| Team Manager | 766 | 0 | 0.00% |
| Senior Manager | 68 | 0 | 0.00% |
| Department Head | 8 | 0 | 0.00% |

The generator also used tenure-dependent eligibility:

| Tenure rule | Employees | Terminations | Observed rate | Expected rate |
|---|---:|---:|---:|---:|
| 0–119 days: forced active | 528 | 0 | 0.00% | 0.00% |
| 120–364 days: reduced probability | 1,165 | 165 | 14.16% | 15.46% |
| 365+ days: full probability | 8,307 | 1,548 | 18.63% | 18.42% |

The zero terminations among protected levels and employees below 120 days
confirm that both hard-coded constraints operated correctly.

## Department and Location Behavior

Observed department attrition varied because department and location were
direct generator inputs:

| Department | Employees | Observed rate | Expected rate |
|---|---:|---:|---:|
| Manufacturing | 2,499 | 21.13% | 20.62% |
| Customer Support | 800 | 19.38% | 20.49% |
| Information Technology | 1,000 | 16.40% | 15.49% |
| Engineering | 2,000 | 15.25% | 15.44% |
| Human Resources | 701 | 14.41% | 15.40% |

The largest observed location rate was Reno at 18.37%; the smallest was
Buffalo at 15.16%. These differences are generator behavior, not evidence
about a real workforce.

## Downstream Synthetic Associations

Employees who terminated had shorter histories and therefore fewer
opportunities to accumulate later records:

| Descriptive measure | Active | Terminated |
|---|---:|---:|
| Average days since hire | 1,012 | 1,104 |
| Average latest performance rating | 3.52 | 3.35 |
| Average compensation growth | 12.26% | 6.13% |
| Average completed training count | 2.42 | 2.31 |
| Average promotion events | 0.234 | 0.121 |

The audit does not claim that training, promotion, or performance caused
termination. Some differences arise because termination stops the creation
of future records, while other differences arise from shared department,
location, or tenure structure.

## What This Audit Achieved

The audit showed that:

1. Overall observed termination closely matched the generator expectation.
2. Protected-level and minimum-tenure constraints were enforced exactly.
3. Department and location patterns were deliberately encoded.
4. Several attractive modeling signals were consequences of synthetic
   record generation rather than independent behavioral mechanisms.

These findings motivated the Version 2 monthly hazard generator and its
temporal, leakage-safe validation design.

## Limitations

- This audit describes Version 1, not the final Version 2 generator.
- All rates are synthetic by construction.
- Agreement between observed and expected counts validates implementation,
  not real-world realism.
- Descriptive group differences are neither causal nor suitable for
  employment decisions.

## Evidence and Reproduction

- Executed analysis: [Notebook 17](../notebooks/17_generator_audit.ipynb)
- Implementation:
  [audit_termination_generator.py](../src/audit_termination_generator.py)
- Generated evidence directory: `data/processed/audit/`

Reproduce the audit from the project root:

```powershell
python src\audit_termination_generator.py
```
