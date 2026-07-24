# Workforce Exploratory Analysis

## Purpose

This report summarizes exploratory analysis of the Version 1 synthetic workforce and retention-modeling dataset.

The objectives are to:

- Describe workforce composition
- Measure target imbalance
- Examine numeric and categorical distributions
- Identify missing-data patterns
- Compare univariate attrition rates
- Detect redundant features
- Separate descriptive findings from generator artifacts

The results are descriptive and should not be interpreted causally.

## Analysis Populations

| Population | Records | Interpretation |
|---|---:|---|
| Full synthetic workforce | 10,000 | Employee records generated through June 30, 2026 |
| Retention-modeling population | 7,386 | Employees eligible at the June 30, 2025 snapshot |
| Twelve-month attrition cases | 627 | Terminations from July 1, 2025 through June 30, 2026 |
| Twelve-month non-attrition cases | 6,759 | Employees without a termination in that window |

The modeling population contains one record per employee.

## Target Imbalance

The twelve-month attrition rate is:

```text
627 / 7,386 = 8.49%