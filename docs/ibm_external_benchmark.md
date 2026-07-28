# IBM HR Analytics External Benchmark

## Purpose

Checkpoint 53 tests whether the project's core modeling discipline can be
reproduced on a second, independently generated workforce dataset.

This is a **methodological benchmark**, not a transport test of the primary
model. The IBM data has different features, different synthetic relationships,
no snapshot dates, and no stated attrition horizon. The primary Version 2 model
cannot be applied directly because the schemas and target definitions do not
match.

The checkpoint does not:

- merge IBM rows with the primary temporal panel;
- reuse or replace the primary fitted model;
- access the primary final-test employee rows;
- change the frozen intervention policy;
- create an IBM employee-review list; or
- modify the dashboard.

## Source and License

The source is IBM's archived
[`employee-attrition-aif360`](https://github.com/IBM/employee-attrition-aif360)
repository. IBM describes the file as Kaggle-supplied HR analytics data and
documents the dataset under the Open Database License and Database Content
License.

| Source item | Committed value |
| --- | --- |
| IBM repository commit | `13287d5f717dc978eda249aef4665e04c7cec8b0` |
| Source file | `data/emp_attrition.csv` |
| Local generated path | `data/external/ibm_emp_attrition.csv` |
| SHA-256 | `a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659` |
| Dataset license | ODbL + DbCL |
| Rows | 1,470 |
| Columns | 35 |

The repository was archived by IBM on July 18, 2024 and is read-only. The
download URL is pinned to a 40-character commit rather than the movable
`master` branch. The downloader verifies the complete file checksum, schema,
target counts, constants, missingness, and identifier uniqueness before use.

OpenML dataset 43893 independently records the same IBM repository provenance,
file dimensions, and ODbL/DbCL license metadata.

## Data Is Fictional

This dataset is commonly called the IBM HR Analytics Employee Attrition &
Performance dataset. It is fictional and does not contain real IBM employee
records.

The source contains:

- 1,233 `Attrition = No` records;
- 237 `Attrition = Yes` records; and
- a positive rate of 16.12%.

The target does not define when attrition occurred or the length of the
prediction window.

## Why the Evaluation Is Not Temporal

The primary project has dated event histories and explicit 12-month prediction
windows. The IBM table is one static row per employee and has no event dates,
snapshot date, or prediction-end date.

Checkpoint 53 therefore uses:

- five outer stratified folds;
- ten repeats;
- 50 total outer evaluations;
- three inner folds for sigmoid calibration; and
- one out-of-fold score per employee in every repeat.

The ten out-of-fold probabilities for each employee are averaged before
aggregate metrics are calculated.

This reduces dependence on one random split, but it does not recreate an
out-of-time test. The result must never be described as temporal validation.

## Feature Policy

The benchmark uses 26 raw predictors:

- 20 numerical features; and
- 6 categorical features.

The following source fields are excluded:

| Field | Reason |
| --- | --- |
| `Attrition` | Target |
| `EmployeeNumber` | Identifier |
| `EmployeeCount` | Constant |
| `Over18` | Constant |
| `StandardHours` | Constant |
| `DailyRate` | Opaque synthetic rate |
| `HourlyRate` | Opaque synthetic rate |
| `MonthlyRate` | Opaque synthetic rate |
| `Gender` | Descriptive subgroup diagnostic only |

Gender is never supplied to the model. Age remains a predictor and is also
reported in broad bands for descriptive review.

Numerical values use median imputation and standard scaling. Categorical values
use most-frequent imputation and reference-category one-hot encoding. The
classifier reuses the primary model family's balanced Logistic Regression
parameters.

## Calibration

The balanced uncalibrated model overpredicts the minority outcome:

| Method | Mean probability | Brier score |
| --- | ---: | ---: |
| Uncalibrated Logistic Regression | 36.65% | 0.1597 |
| Sigmoid-calibrated Logistic Regression | 16.32% | 0.0975 |
| Observed attrition rate | 16.12% | — |

Sigmoid calibration is fitted only inside each outer training fold. The held-out
outer fold does not train either the base classifier or its calibration mapping.

## Aggregate Benchmark Results

The repeated out-of-fold calibrated result is:

| Metric | Estimate | 95% bootstrap interval |
| --- | ---: | ---: |
| PR-AUC | 0.6094 | 0.5520–0.6614 |
| ROC-AUC | 0.8277 | 0.7977–0.8562 |
| Brier score | 0.0975 | 0.0916–0.1038 |
| Top-decile precision | 69.39% | 61.55%–75.19% |
| Top-decile capture | 43.04% | 38.18%–46.64% |
| Top-decile lift | 4.30 | 3.82–4.66 |

The top decile contains 147 employees and 102 positive cases.

These results show that the fixed preprocessing, balanced Logistic Regression,
sigmoid calibration, PR-AUC, lift, and bootstrap framework works on a second
synthetic dataset. They do not prove performance on real employees.

## Why the IBM Metrics Are Higher

The IBM benchmark produces much stronger ranking than the primary temporal
model. That does not mean the IBM model is universally better.

Important differences include:

- The IBM target has no stated future horizon.
- The benchmark uses repeated random stratification rather than future-time
  testing.
- IBM includes strong static fields such as overtime, travel, satisfaction,
  marital status, and job role.
- The two simulators encode different relationships between predictors and
  attrition.
- The IBM dataset is smaller and contains only one static record per employee.

For these reasons, the metrics are presented side by side only as context.
Every comparison row is explicitly marked `directly_comparable = False`.

## Coefficient Stability

Across 50 outer training folds, the largest stable positive modeled
associations include:

- overtime;
- frequent business travel;
- laboratory technician role;
- single marital status; and
- sales representative role.

Several role and education-field indicators have stable negative associations.
These are multivariable model coefficients, not causal effects. Correlation,
reference-category choice, and synthetic data construction affect their signs
and magnitudes.

## Descriptive Subgroup Diagnostics

Checkpoint 53 reports aggregate measures by:

- gender;
- age band; and
- department.

The measures include observed rate, mean probability, calibration gap,
top-decile selection rate, true-positive rate, false-positive rate, precision,
and Brier score.

Gender is excluded from the predictors, but subgroup differences can still
remain through correlated features and different outcome base rates. These
tables are screening diagnostics, not a binary fairness verdict.

No IBM row-level probabilities, rankings, or contact lists are saved.

## Feature Mapping

Several fields have approximate conceptual matches:

| IBM field | Primary Version 2 concept | Limitation |
| --- | --- | --- |
| `Age` | `approx_age` | Close conceptual match |
| `Education` | `education_level` | Ordinal code versus named category |
| `Department` | `department_name` | Different synthetic organizations |
| `JobLevel` | `job_level` | Different scales |
| `MonthlyIncome` | `salary_position_percent` | Absolute versus peer-relative pay |
| `YearsAtCompany` | `tenure_years` | Close conceptual match |
| `YearsSinceLastPromotion` | `months_since_promotion` | Different unit and missing-history treatment |
| `Attrition` | `attrition_next_12m` | Unknown horizon versus explicit 12 months |

The mapping supports conceptual discussion. It is not sufficient for combining
the datasets or applying one fitted model to the other.

## Reproduction

Run the complete external benchmark with:

```powershell
.\scripts\run_checkpoint53.ps1
```

The runner:

1. compiles source and tests;
2. runs Ruff and formatting checks;
3. downloads or verifies the pinned dataset;
4. runs all 50 outer benchmark folds;
5. calculates 500 bootstrap samples;
6. writes aggregate tables and figures;
7. validates the isolation and governance contracts; and
8. runs the complete automated test suite.

The full project runner includes the external benchmark by default. Use
`-SkipExternalBenchmark` only when an offline end-to-end run is necessary.

## Generated Outputs

Generated files are written under `data/processed/ibm_benchmark/` and excluded
from Git:

- data profile;
- aggregate calibrated and uncalibrated metrics;
- outer-fold summary;
- bootstrap samples and intervals;
- coefficient-stability table;
- subgroup metrics;
- primary-versus-IBM context table;
- feature mapping;
- validation checks; and
- three aggregate figures.

The executed supporting notebook
`notebooks/31_ibm_external_benchmark.ipynb` preserves aggregate evidence for
GitHub review.

## Limitations

- Both the IBM data and the primary workforce are synthetic.
- No outcome date or prediction horizon exists in the IBM table.
- Random stratification is weaker than a true future-time test.
- Repeated folds are not independent new organizations.
- Strong performance may partly reflect how the fictional target was
  generated.
- No hyperparameter search is performed.
- Subgroup analysis is descriptive and sample sizes differ.
- The benchmark does not establish real-world calibration, fairness, causal
  impact, or policy value.
