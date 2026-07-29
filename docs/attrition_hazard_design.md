# Version 2 Attrition Hazard Design

## Purpose

Version 1 assigns a single termination probability to each eligible employee and then uniformly selects a termination date across the employee's available history.

Checkpoint 32 demonstrated that this creates:

- Abrupt tenure cliffs
- Permanently active leaders
- Hard-coded department and location outcomes
- Non-uniform calendar patterns
- Outcome-conditioned behavioral histories
- Predictive signal generated after the outcome is known

Version 2 will replace that process with a monthly, discrete-time competing-risks simulation.

This checkpoint defines the design only. Production generator implementation begins in Checkpoint 36.

## Definition of Hazard

A hazard is the probability that an employee who is active at the beginning of a period experiences an event during that period.

Version 2 uses monthly periods and three possible outcomes:

1. Remain active
2. Voluntary termination
3. Involuntary termination

Once an employee terminates, later months are not simulated for that employee.

## Reference Baseline

For a reference employee in a reference month:

| Outcome | Monthly probability |
|---|---:|
| Remain active | 99.13% |
| Voluntary termination | 0.65% |
| Involuntary termination | 0.22% |

The combined monthly termination probability is 0.87%.

If risk remained constant for 12 months, the approximate annual probability would be:

```text
1 - (0.9913 ** 12) ≈ 9.96%
```
