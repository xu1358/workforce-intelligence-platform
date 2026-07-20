# Dashboard Data Layer

## Purpose

The dashboard data layer prepares workforce, recruiting, and retention-model information for the Workforce Intelligence and Retention Decision Platform.

The layer combines:

- PostgreSQL workforce data
- Recruiting data
- Job requisition data
- Retention-model outputs
- Retention risk segmentation
- Model evaluation metrics

The data layer separates data preparation from dashboard visualization so that the dashboard can read clean, predefined datasets.

## Build process

The dashboard datasets are generated with:

`python src/build_dashboard_data.py`

Generated files are stored locally under:

`data/processed/dashboard/`

These generated files are excluded from Git.

## Dashboard datasets

### overview_kpis.csv

Contains high-level business metrics including:

- Active headcount
- Active department count
- Active location count
- 2025 terminations
- 2025 turnover rate
- Open requisitions
- Total applications
- Hired applications
- Retention snapshot population
- High-, medium-, and low-risk employee counts

### headcount_by_department.csv

Contains active workforce headcount and workforce percentage by department.

### headcount_by_location.csv

Contains active workforce headcount and workforce percentage by location and region.

### recruiting_funnel.csv

Contains recruiting outcomes by application source, including application counts, hires, rejections, withdrawals, declined offers, active applications, cancelled positions, and hire rates.

### requisition_metrics.csv

Contains requisition counts, target headcount, and average days open by requisition status.

### retention_risk_summary.csv

Contains summary statistics for Low, Medium, and High retention-risk groups.

### retention_risk_employees.csv

Contains employee-level model risk scores and selected workforce characteristics for the retention snapshot population.

The full snapshot population is scored for dashboard prioritization. These scores should not be used as unbiased estimates of predictive performance because part of the population was used during model training.

### model_performance.csv

Contains the held-out test-set performance of the compared retention models.

### model_summary.csv

Contains the selected model, recommended classification threshold, risk-segmentation thresholds, and selected-model test metrics.

## Risk interpretation

The dashboard uses relative retention-risk segments:

- High risk: approximately the top 10% of model scores
- Medium risk: approximately the next 20%
- Low risk: approximately the remaining 70%

Risk groups are intended for prioritization rather than causal interpretation.

Predicted probabilities have not been calibrated and should not be interpreted as exact real-world probabilities of employee departure.

## Validation

Dashboard data outputs are validated in:

`notebooks/16_dashboard_data_validation.ipynb`