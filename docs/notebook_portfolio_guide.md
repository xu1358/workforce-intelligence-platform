# Curated Notebook Portfolio

## Purpose

Checkpoint 51 changes how the analytical work is presented without changing
the data generator, model, calibration, policy, dashboard, or workforce
planning calculations.

Thirty-eight checkpoint notebooks document the construction history in detail.
That history is valuable for auditing, but it is too long to serve as the
primary portfolio narrative. The curated layer reduces the reviewer-facing
path to four executed notebooks while retaining all supporting notebooks.

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

The reviewer-visible Version 2 notebooks `20` through `30` also retain their
executed tables and figures. A dedicated execution script and validation
contract prevent these technical notebooks from being committed as blank
templates while leaving the four-notebook reviewer sequence concise.

## External benchmark notebook

Checkpoint 53 adds
`notebooks/31_ibm_external_benchmark.ipynb` as a separate supporting
benchmark. It is not a fifth notebook in the primary portfolio sequence.

Notebook 31 records aggregate evidence from a pinned fictional IBM HR
Analytics dataset. Because that table has no dates or stated attrition
horizon, it uses repeated stratified cross-validation rather than temporal
testing. Its metrics are not directly comparable with the primary model's
reserved future-time test.

Checkpoint 54 adds
`notebooks/32_retention_explanation_stability.ipynb` as a second optional
supporting extension. It presents aggregate current-model drivers, exact
linear-SHAP additivity, employee-grouped importance stability, and noncausal
interpretation limits. It does not expose employee-level explanations or
change the curated four-notebook sequence.

Checkpoint 55 adds
`notebooks/33_survival_analysis.ipynb` as a third optional supporting
extension. It presents right-censored Kaplan–Meier retention curves,
baseline-at-hire Cox hazard ratios, held-out concordance, and
proportional-hazards diagnostics. It answers a time-to-exit question and does
not replace the primary 12-month classifier, reopen the final test, or change
the frozen review policy.

Checkpoint 56 adds
`notebooks/34_retention_policy_equity.ipynb` as a fourth optional supporting
extension. It audits salary allocation after economic optimization, compares
salary-neutral and salary-capped sensitivities, and preserves the frozen
policy and outcome-access boundaries.

Checkpoint 64 adds `notebooks/35_deployed_policy_fairness.ipynb` as the
authoritative decision-layer subgroup audit. It reproduces the exact frozen
expected-value policy instead of substituting probability-only selection.

Checkpoint 65 adds `notebooks/36_temporal_prior_shift.ipynb` as a temporal
calibration-transport extension. It quantifies material base-rate drift,
documents prior-probability shift risk, and prohibits recalibration on the
once-only final test.

Checkpoint 66 adds two authoritative data-foundation successors:

- `notebooks/37_v2_exploratory_analysis.ipynb` profiles the temporal panel,
  current population, training-only group rates, and review-history
  availability without inspecting reserved outcomes for EDA; and
- `notebooks/38_v2_department_history.ipynb` verifies immutable department
  assignment, location-semantic transfers, and dated location reconstruction
  without accessing attrition targets.

Notebooks 18–19 remain explicitly labeled historical Version 1 evidence.

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
- Saved execution counts and outputs for configured supporting notebooks
- Absence of error outputs and machine-specific paths in those notebooks
- Portable kernels and GitHub-safe file sizes for those notebooks

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

The IBM benchmark is implemented separately in Checkpoint 53 and does not
change any of the four curated notebooks.

The explanation extension is implemented separately in Checkpoint 54 and also
leaves the four curated notebooks unchanged.
