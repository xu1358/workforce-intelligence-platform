# Retention Model Explanations and Stability

## Purpose

Checkpoint 54 explains the selected Version 2 Logistic Regression and tests
whether its aggregate explanation is stable when the historical employee
sample changes.

The checkpoint answers four questions:

1. Which raw features contribute most to the current synthetic workforce
   scores?
2. Do the feature contributions exactly reconstruct the fitted model score?
3. Do the important features remain similar across employee-grouped refits?
4. Can this evidence be presented without exporting employee-level
   explanations or changing the frozen intervention policy?

This is a model-transparency checkpoint. It is not another model-selection,
test-evaluation, calibration, threshold, or policy-selection checkpoint.

## Model Being Explained

The explained model is the already selected Version 2 model:

| Contract item | Committed value |
|---|---|
| Base model | Logistic Regression |
| Calibration | Sigmoid |
| Historical refit rows | 16,368 |
| Current eligible scoring rows | 7,305 |
| Raw model features | 21 |
| Encoded model columns | 35 |
| Current snapshot | 2026-06-30 |

The model is refitted on all three completed historical snapshots, exactly as
it was for the current Checkpoint 46 scoring population. The script also
recomputes the grouped out-of-fold sigmoid calibration and verifies that the
current calibrated probabilities reproduce the committed Checkpoint 46
probabilities.

The final 2025 test result is not rerun, and the frozen top-700 expected-value
policy is not changed.

## Exact Linear SHAP Method

The selected model is linear on its transformed feature matrix. For the fitted
log-odds model

```text
f(x) = intercept + beta · x
```

and the mean transformed historical background `E[X]`, the contribution for
encoded feature `j` is:

```text
SHAP_j = beta_j × (x_j - E[X_j])
```

The base value is:

```text
base = intercept + beta · E[X]
```

Therefore:

```text
base + sum(SHAP_j) = model decision_function(x)
```

This is the exact interventional linear SHAP decomposition under the selected
background assumption. Checkpoint 54 validates the equality for every current
row to numerical tolerance.

No new `shap` dependency is required. For this specific linear model, the
closed-form calculation is transparent, exact, and easier to audit.

## Why the Values Use Log-Odds

The contribution values explain the Logistic Regression's native log-odds,
not calibrated probability percentage points.

Checkpoint 42's sigmoid calibration converts the raw Logistic Regression
score into a better probability estimate. The mapping is monotonic, so it
preserves ranking, but the mapping is nonlinear. A contribution of `0.20`
therefore means `+0.20` on the base-model log-odds scale; it does not mean
`+20 percentage points` of attrition probability.

The dashboard should continue to display calibrated 12-month probabilities.
The explanation artifacts should be described as model-score drivers.

## Raw-Feature Grouping

The preprocessor creates 35 encoded columns from 21 raw features. Numerical
features have one encoded column. Categorical features have multiple one-hot
columns after dropping the configured reference category.

Checkpoint 54 sums all encoded contributions belonging to one categorical
feature. For example, the department one-hot columns are reported together as
`department_name`.

The script validates that:

```text
sum(encoded contributions) = sum(grouped raw-feature contributions)
```

Grouping makes the result easier to communicate and prevents a categorical
feature with many levels from occupying several positions in the global
importance table.

## Current Aggregate Explanation

The main output ranks features by mean absolute SHAP value across all 7,305
eligible current employees. Mean absolute magnitude measures how strongly a
feature changes model scores in either direction.

The leading aggregate drivers are:

| Rank | Raw feature | Mean absolute SHAP, log-odds |
|---:|---|---:|
| 1 | `salary_position_percent` | 0.4586 |
| 2 | `tenure_years` | 0.2604 |
| 3 | `performance_rating` | 0.2467 |
| 4 | `no_prior_review` | 0.1268 |
| 5 | `department_name` | 0.1064 |
| 6 | `job_level` | 0.1053 |
| 7 | `no_prior_promotion` | 0.0930 |
| 8 | `completed_training_hours_12m` | 0.0834 |
| 9 | `salary_growth_12m_percent` | 0.0638 |
| 10 | `performance_trend` | 0.0500 |

These values describe the fitted synthetic model. They do not establish that
changing salary position, performance, review timing, department, or another
feature would cause an employee to stay or leave.

Mean signed contribution is reported separately. It describes whether a
feature's current-population contribution is higher or lower than the
historical transformed background on average. It should not be confused with
the sign of every employee's contribution.

## Probability-Quartile Analysis

The current population is divided into four equal-count groups using the
committed calibrated probabilities:

- lowest probability;
- lower-middle probability;
- upper-middle probability; and
- highest probability.

For every raw feature and quartile, the output reports:

- number of employees;
- mean calibrated probability;
- mean signed SHAP value;
- mean absolute SHAP value; and
- descriptive direction on the model log-odds scale.

This comparison helps show which aggregate model contributions distinguish
the highest score group from the lower score groups. It does not define a new
threshold and does not change the 700-person human-review plan.

## Driver-Frequency Analysis

For each current employee, the script identifies the largest positive and
largest negative raw-feature contribution in memory. Only aggregate counts
and shares are saved.

This output shows how often a feature is the strongest model-score driver. It
does not expose:

- employee IDs;
- employee probabilities;
- employee outcomes;
- human-review selections; or
- employee-level SHAP values.

## Employee-Grouped Stability Design

One fitted model can make an importance ranking appear more certain than it
is. Checkpoint 54 therefore creates five new diagnostic refits.

`StratifiedGroupKFold` assigns complete employee histories to folds. For each
fold:

1. four employee groups fit the selected Logistic Regression;
2. the fifth employee group remains outside the fit;
3. the fitted model explains the same current 7,305-row population;
4. encoded contributions are grouped into the same 21 raw features; and
5. aggregate feature importance and highest-quartile direction are recorded.

No employee appears in both the fit and holdout side of one stability fold.
These refits test explanation robustness. They do not produce alternative
deployed models.

## Stability Measures

The checkpoint uses three complementary measures:

| Measure | Meaning | Committed minimum |
|---|---|---:|
| Pairwise Spearman correlation | Similarity of all 21 importance ranks | 0.85 |
| Pairwise top-10 Jaccard | Overlap of the ten most important feature sets | 0.65 |
| Top-feature direction stability | Share of folds with the same highest-quartile direction | 0.90 |

Observed results:

| Measure | Result |
|---|---:|
| Minimum pairwise Spearman correlation | 0.8857 |
| Median pairwise Spearman correlation | 0.9299 |
| Minimum pairwise top-10 Jaccard | 0.6667 |
| Minimum top-10 direction stability | 1.0000 |

The evidence supports a careful claim: the broad importance ordering is
stable across these grouped refits, although the exact rank of a feature can
move.

The top-10 Jaccard minimum of `0.6667` means the least-similar fold pair still
shared eight of ten features:

```text
intersection / union = 8 / 12 = 0.6667
```

It does not mean that every top-ten position stayed identical.

## Generated Outputs

Checkpoint 54 saves aggregate outputs under:

```text
data/processed/retention_explanations/
```

| Output | Purpose |
|---|---|
| `retention_explanation_global_importance.csv` | Current raw-feature importance |
| `retention_explanation_fold_stability.csv` | Per-fold aggregate feature results |
| `retention_explanation_pairwise_stability.csv` | Fold-pair rank and top-k comparisons |
| `retention_explanation_stability_summary.csv` | Raw-feature stability summary |
| `retention_explanation_probability_quartiles.csv` | Aggregate quartile contributions |
| `retention_explanation_driver_frequency.csv` | Aggregate strongest-driver frequencies |
| `retention_explanation_validation.csv` | Mathematical, stability, privacy, and governance checks |

The three generated figures are:

- `retention_global_shap_importance.png`;
- `retention_shap_fold_stability.png`; and
- `retention_shap_probability_quartiles.png`.

The executed supporting notebook is:

```text
notebooks/32_retention_explanation_stability.ipynb
```

## Governance and Interpretation

The explanation configuration requires:

- explanations are associations, not causes;
- employee-level explanation exports are prohibited;
- current scores contain no known future outcome;
- human review remains required;
- automatic employment action remains prohibited;
- the once-only final test is not reopened; and
- policy retuning is not permitted.

The explanations can support questions such as:

- Which information is the model relying on most?
- Does that reliance remain similar across employee-grouped refits?
- Are high-score groups driven by plausible and reviewable patterns?
- Which model behaviors deserve further fairness or data-quality review?

They cannot answer:

- What caused a particular employee to leave?
- What intervention will definitely retain an employee?
- Is an employee disloyal or planning to resign?
- Should an employment action be taken automatically?

## Limitations

1. Both the main workforce and its outcomes are synthetic.
2. Linear SHAP uses a transformed-feature mean background and an
   interventional independence assumption.
3. One-hot categories are mutually related by construction. Grouping them
   improves presentation but does not remove all dependence concerns.
4. Aggregate stability across five refits does not guarantee stability under
   future data drift.
5. The explanations apply to this fitted model and current synthetic scoring
   population, not to all organizations or time periods.
6. Feature importance is not causal intervention value.
7. Calibrated probabilities and log-odds contributions use different scales.

## Reproduction

From the project root in PowerShell:

```powershell
.\scripts\run_checkpoint54.ps1
```

The runner:

1. compiles source and tests;
2. runs Ruff lint checks;
3. checks test formatting;
4. reproduces the current model explanations;
5. performs five grouped stability refits;
6. validates privacy and policy safeguards; and
7. runs the complete automated test suite.

