# Feature Redundancy and Interpretation

## Purpose

Checkpoint 39 defines which columns may enter Version 2 retention
models and how their coefficients may be discussed.

The temporal dataset contains 52 columns, but using every available
column would create several problems:

- Some columns contain the same information in different forms.
- Some categories are completely determined by other categories.
- Some counts primarily repeat tenure.
- One-hot encoding every category without a reference creates exact
  linear dependence.
- A stable predictive coefficient can still be noncausal.

The final policy keeps a compact, auditable set before model comparison.

## Development and holdout separation

Target-informed diagnostics use only:

```text
2023-06-30 snapshot
2024-06-30 snapshot
```

This provides 9,727 model-eligible development rows.

The 2025-06-30 snapshot contains 6,641 model-eligible rows and remains
reserved. Its target is not used in the Checkpoint 39 coefficient
bootstrap. Checkpoint 40 will formalize its role in the final temporal
validation strategy.

## Modeling population

The attrition generator protects department heads and senior managers
from termination so the hierarchy remains valid. Including those rows
would teach a model an artificial rule:

> Protected leadership level means guaranteed retention.

Checkpoint 39 therefore defines the model-eligible levels as:

- Individual Contributor
- Team Manager

Department heads and senior managers remain in the workforce data and
dashboard totals but are excluded from retention-model development and
future intervention scoring.

The current active file contains 7,409 employees. Applying the feature
policy leaves 7,305 eligible scoring candidates.

## Original redundancy findings

### Numeric duplication

The strongest development-period correlations include:

| First feature | Second feature | Absolute correlation |
|---|---|---:|
| Days since compensation change | Days since review | 1.000 |
| Promotion compensation count | Prior promotion events | 1.000 |
| Initial base salary | Current base salary | 0.985 |
| Latest performance rating | Average performance rating | 0.980 |
| Tenure | Compensation record count | 0.953 |

The first two pairs are exact duplicates in the generated histories.
Keeping both would add columns without adding information.

### Categorical duplication

The strongest corrected Cramer's V values include:

| First feature | Second feature | Corrected Cramer's V |
|---|---|---:|
| City | Region | 1.000 |
| City | Location type | 1.000 |
| Department | Department group | 1.000 |
| Job title | Job family | 0.999 |
| Job title | Job level | 0.999 |
| Employment type | Job level | 0.957 |
| Department | Job family | 0.849 |

These results reflect the synthetic reference-table design. For
example, a city determines its region and location type. Keeping all
three would repeatedly encode one location.

Department and job family also created exact dependencies after
one-hot encoding for several synthetic groups. Department is retained
because it is directly relevant to operational workforce planning.
Job family is removed from the Logistic Regression feature policy.

## Selected raw model features

The policy keeps 21 raw features.

### Numerical features

```text
approx_age
tenure_years
salary_growth_12m_percent
salary_position_percent
days_since_compensation_change
performance_rating
performance_trend
no_prior_review
completed_training_hours_12m
failed_training_programs_12m
prior_transfer_events
manager_changes_12m
leaves_12m
months_since_promotion
no_prior_promotion
```

### Categorical features

```text
education_level
organizational_level
department_name
city
job_level
promotion_recommended
```

The complete decision and reason for every dataset column is stored in:

```text
data/processed/feature_diagnostics.csv
```

## Promotion-recency correction

When an employee has never received a promotion, the raw temporal
dataset fills promotion recency using tenure. That is useful for
general reporting but creates strong overlap between:

```text
tenure_years
months_since_promotion
```

Before modeling, Checkpoint 39 changes promotion recency to missing for
employees with no prior promotion and retains:

```text
no_prior_promotion = 1
```

The preprocessing pipeline then median-imputes promotion recency. This
separates:

- Whether a promotion has ever occurred
- How long ago the promotion occurred, when one exists

The raw temporal dataset is not modified.

## Reference-category encoding

Categorical variables use one omitted reference category:

| Feature | Reference |
|---|---|
| Education | Bachelor's |
| Organizational level | Individual Contributor |
| Department | Engineering |
| City | Austin |
| Job level | 2 |
| Promotion recommendation | False |

For example:

```text
department_name_Manufacturing
```

means Manufacturing compared with the Engineering reference, holding
the other model columns fixed.

Reference encoding produces 35 encoded columns from the 21 raw
features. The matrix has full column rank.

## Collinearity diagnostics

### Variance inflation factor

VIF measures how well one encoded column can be explained by the other
encoded columns.

General interpretation:

```text
VIF near 1: little redundancy
VIF below 5: acceptable
VIF 5–10: elevated
VIF above 10: serious concern
```

The maximum selected-feature VIF is:

```text
3.95
```

### Condition number

The standardized design-matrix condition number is:

```text
5.29
```

The configured acceptable limit is 30. The selected matrix is therefore
well conditioned for the planned regularized Logistic Regression.

## Coefficient stability

Checkpoint 39 fits regularized Logistic Regression repeatedly using 200
employee-cluster bootstrap samples.

Employee-cluster sampling keeps all available development snapshots for
a sampled employee together. This recognizes that multiple rows from
one employee are related.

A direction is reportable only when:

1. The empirical 95% bootstrap interval excludes zero.
2. At least 90% of bootstrap coefficients have the same sign.

Nine of 35 encoded coefficient directions meet both rules.

Selected examples:

| Encoded feature | Direction | Important qualification |
|---|---|---|
| Salary position | Lower predicted risk at higher relative pay | Designed synthetic relationship; not causal evidence |
| Performance rating | Lower predicted risk at higher performance | Designed synthetic relationship; not causal evidence |
| Customer Support department | Higher predicted risk than Engineering | Conditional synthetic department association |
| Job level 3 or 4 | Lower predicted risk than level 2 | Conditional association with role and hierarchy |
| No prior review | Lower predicted risk | Likely reflects onboarding and tenure structure |
| Prior transfer count | Slightly higher predicted risk | Transfer is not a proven causal driver |

Education coefficients also appear stable in this sample even though
education is not a direct attrition-hazard input. This demonstrates an
important principle:

> Bootstrap stability does not transform an association into a causal
> relationship.

Education can remain useful for predictive diagnostics, but its
coefficient should not be interpreted as the effect of changing an
employee's education.

## What the bootstrap intervals mean

The reported ranges are empirical coefficient-stability intervals for
a regularized predictive model. They are not classical confidence
intervals and are not causal-effect estimates.

They answer:

> Does the coefficient keep a similar direction when employees are
> repeatedly resampled?

They do not answer:

> Would changing this feature cause an employee to stay or leave?

## Files produced

```text
data/processed/feature_diagnostics.csv
data/processed/numeric_correlation_diagnostics.csv
data/processed/categorical_association_diagnostics.csv
data/processed/vif_diagnostics.csv
data/processed/condition_number_diagnostics.csv
data/processed/encoded_feature_manifest.csv
data/processed/coefficient_stability.csv
data/processed/feature_policy_validation.csv
data/processed/bootstrap_run_summary.csv
```

Generated diagnostic CSV files remain excluded from Git and can be
recreated from committed code.

## Remaining limitations

- The data-generating process is synthetic.
- Feature selection incorporates semantic judgment, not one automatic
  statistical rule.
- L2 regularization changes coefficient magnitude.
- Correlated predictors can still share or redistribute coefficients
  even when VIF is acceptable.
- Bootstrap intervals measure resampling stability, not causal
  uncertainty.
- The reserved 2025 snapshot has not yet been evaluated.
- Final train, validation, and test rules are deferred to Checkpoint 40.
