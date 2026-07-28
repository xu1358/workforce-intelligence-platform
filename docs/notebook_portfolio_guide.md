# Curated Notebook Portfolio

## Purpose

Checkpoint 51 changes how the analytical work is presented without changing
the data generator, model, calibration, policy, dashboard, or workforce
planning calculations.

Thirty checkpoint notebooks document the construction history in detail. That
history is valuable for auditing, but it is too long to serve as the primary
portfolio narrative. The curated layer reduces the reviewer-facing path to
four executed notebooks while retaining all supporting notebooks.

## Portfolio sequence

### 1. Data Foundation and Temporal Design

This notebook establishes:

- Version 2 data-integrity results
- The difference between cumulative attrition and a forward twelve-month label
- Three nonoverlapping historical prediction windows
- Chronological train, validation, and reserved-test assignments
- Feature redundancy controls and the target-free feature policy
- Leakage boundaries and current-scoring separation

### 2. Model Development and Final Evaluation

This notebook presents:

- Logistic Regression, Gradient Boosting, and Random Forest comparisons
- Paired uncertainty and the simplicity rule used for provisional selection
- Sigmoid probability calibration
- Precision, capture, and lift at multiple review capacities
- The once-only 2025 out-of-time test
- A clear distinction between ranking quality and probability quality

### 3. Fairness, Economics, and Retention Policy

This notebook connects:

- Descriptive subgroup diagnostics
- Age-and-education exclusion sensitivity
- Replacement-cost, intervention-cost, and effectiveness scenarios
- Six policy comparators
- The frozen 700-person, $1.75 million expected-value policy
- Human-review and no-automatic-action requirements

Subgroup flags trigger investigation. They do not constitute a legal or binary
fairness verdict.

### 4. Current Manufacturing Workforce Stability Plan

This notebook translates the tested policy into:

- The current active and model-eligible populations
- Department-level expected departures and residual backfill scenarios
- Manufacturing role-level planning
- Intervention spend, avoided-cost, and net-value projections
- Vacancy, training, coverage, and time-to-productivity units
- Explicit planning caveats

Current probabilities are projections. Outcomes after the as-of date are
unknown, so this notebook is not another performance evaluation.

## Curation principles

The portfolio sequence follows six rules:

1. **Decision first:** every notebook begins with an executive summary.
2. **Aggregate only:** employee-level human-review lists are not embedded.
3. **Executed evidence:** every code cell has an execution count and no error
   output.
4. **Historical/current separation:** known evaluation outcomes are not mixed
   with current unknown outcomes.
5. **Synthetic disclosure:** every notebook identifies the data as synthetic.
6. **No causal overclaiming:** associations and scenarios are not described as
   causal effects or guaranteed forecasts.

## Why the original notebooks remain

The supporting notebooks preserve:

- Step-by-step generator validation
- SQL and Version 1 baseline work
- The termination-generator audit
- EDA and historical-assignment analysis
- Checkpoint-specific Version 2 diagnostics

Keeping them avoids destroying development history. The new
`notebooks/README.md` makes the four-notebook path the explicit entry point.

## Automated validation

`src/validate_portfolio_notebooks.py` checks:

- Four unique manifest entries in order
- Valid notebook JSON and notebook format version
- Exact titles and required headings
- Executed code cells
- Absence of error outputs
- Minimum embedded output coverage
- Previous/next navigation
- Synthetic-data and noncausal language
- Absence of employee identifiers and machine-specific absolute paths
- File-size limits
- Index links and the supporting-notebook inventory

The validation produces:

```text
data/processed/portfolio_notebook_validation.csv
```

The output is generated locally and remains outside Git.

## What this checkpoint does not do

Checkpoint 51 does not:

- Retrain or retune a model
- Reopen the final test period
- Change the frozen policy
- Change the dashboard
- Regenerate a current employee ranking
- Delete the original notebooks
- Add the IBM external benchmark

The IBM benchmark remains planned for Checkpoint 53.
