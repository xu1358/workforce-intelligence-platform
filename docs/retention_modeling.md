# Retention Modeling

## Objective

The retention modeling component predicts whether an employee who is employed at a historical snapshot date will leave during the following twelve months.

## Snapshot design

Snapshot date:

`2025-06-30`

Prediction window:

`2025-07-01` through `2026-06-30`

Target:

`attrition_next_12m`

The target equals 1 when an employee terminates during the prediction window and 0 otherwise.

## Modeling population

Employees are included when:

- Their hire date is on or before the snapshot date.
- They are still employed on the snapshot date.

Employees who terminated before the snapshot are excluded.

Employees hired after the snapshot are excluded.

## Historical features

Features are constructed using information available on or before the snapshot date.

Feature groups include:

- Employee demographics
- Employment type
- Original recruiting source
- Original hiring organization
- Previous experience
- Compensation history
- Performance history
- Training history
- Promotion history
- Transfer history
- Manager-change history
- Leave history

## Leakage prevention

The modeling dataset does not contain direct outcome columns such as:

- `employment_status`
- `termination_date`
- `termination_type`

Compensation records after the snapshot are excluded.

Performance reviews after the snapshot are excluded.

Training outcomes occurring after the snapshot are excluded from completed or failed training summaries.

Employee events after the snapshot are excluded.

## Generated dataset

The modeling dataset is generated with:

`python src/build_retention_dataset.py`

The SQL query is stored in:

`sql/retention_modeling_dataset.sql`

The generated file is:

`data/processed/retention_modeling_dataset.csv`

The processed dataset is generated locally and is not committed to Git.

## Validation

The dataset is validated in:

`notebooks/12_retention_dataset_validation.ipynb`

The validation checks:

- One row per employee
- Unique employee identifiers
- Correct snapshot dates
- Binary target values
- Presence of both target classes
- Absence of direct target-leakage columns
- Missing-value patterns
- Numerical feature distributions
- Categorical feature distributions

## Baseline model

The first attrition-classification model uses Logistic Regression.

The baseline workflow includes:

- Stratified 80/20 train/test split
- Median imputation for numerical features
- Standard scaling for numerical features
- Most-frequent imputation for categorical features
- One-hot encoding for categorical features
- Balanced class weights
- Logistic Regression classification

A Dummy Classifier is also evaluated as a simple benchmark.

## Evaluation metrics

The baseline models are evaluated with:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Confusion matrix

Accuracy is not used as the only evaluation measure because employee attrition may be an imbalanced classification problem.

## Baseline training

The model can be trained with:

`python src/train_baseline_retention_model.py`

The trained pipeline is saved locally as:

`models/baseline_retention_model.joblib`

Test-set predictions are saved locally as:

`data/processed/baseline_retention_predictions.csv`

The generated model and prediction files are not committed to Git.

## Baseline notebook

Detailed baseline-model analysis is performed in:

`notebooks/13_baseline_retention_model.ipynb`