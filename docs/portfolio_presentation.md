# Workforce Intelligence Platform — Portfolio Presentation

## Project Title

Workforce Intelligence and Retention Decision Platform

## Project Summary

This project is an end-to-end workforce analytics platform built with Python, PostgreSQL, SQL, scikit-learn, Plotly, and Streamlit.

It simulates fragmented enterprise HR systems, integrates synthetic workforce data into a relational database, supports workforce and recruiting analytics, builds an employee attrition-classification workflow, and presents decision-support insights through an interactive dashboard.

The project uses entirely synthetic employee and candidate data.

---

## Portfolio Highlights

- Generated approximately 10,000 synthetic employees and 40,000 candidates.
- Simulated six enterprise HR source systems.
- Designed and populated a 12-table PostgreSQL relational schema.
- Built automated CSV-to-PostgreSQL ingestion, analytical views, and
  content-parity validation.
- Developed SQL analytics for workforce, turnover, recruiting, compensation, performance, and training.
- Created a leakage-aware employee retention modeling dataset.
- Compared Logistic Regression, Random Forest, and Gradient Boosting.
- Selected the final model primarily using PR-AUC.
- Evaluated classification thresholds and retention-risk segments.
- Built a five-section interactive Streamlit dashboard.
- Added one cross-platform Python pipeline with a thin PowerShell wrapper.

---

## Generator Signal Governance

The generator used a development-only ROC-AUC diagnostic before formal model
development. Its purpose was to reject both a nearly random synthetic problem
and an unrealistically easy one.

The accepted generator diagnostic was `0.6298`. The later once-only final test
produced `0.6061` ROC-AUC with no retuning. Therefore the generator acceptance
range is not a final-model performance requirement.

Detailed evidence and safeguards are documented in
[`synthetic_signal_governance.md`](synthetic_signal_governance.md).

---

## Temporal Calibration Transport

Historical 12-month attrition prevalence rises from `9.62%` to `10.42%` to
`11.91%`, a `23.74%` relative increase. In the model-eligible population,
sigmoid calibration is selected on a `10.61%` validation period and evaluated
on a `12.09%` final-test period.

The final mean prediction is `11.02%`, so the aggregate calibration gap is
`−1.07` percentage points. The model tracks part of the period change but not
all of it. This is described as prior-probability shift risk rather than proof
of pure label shift.

The final test is not used to recalibrate probabilities. In a real deployment,
recalibration would be fit only after a new labeled monitoring window matures
and would be evaluated on a later untouched period.

Detailed evidence is documented in
[`temporal_prior_shift.md`](temporal_prior_shift.md).

---

## Technology Stack

- Python
- pandas
- NumPy
- Faker
- PostgreSQL
- SQL
- psycopg2
- scikit-learn
- joblib
- Plotly
- Streamlit
- Jupyter Notebook
- Git and GitHub

---

## Data Architecture

The project simulates six fictional source systems:

1. Human Resources Information System
2. Recruiting System
3. Compensation System
4. Performance Review System
5. Learning and Training System
6. Employee Event System

The data pipeline is:

```text
Synthetic source systems
        ↓
Python data generators
        ↓
Raw CSV files
        ↓
PostgreSQL database
        ↓
Version 2 SQL analytical views
        ↓
Point-in-time retention dataset
        ↓
Machine-learning models
        ↓
Risk segmentation
        ↓
Dashboard data layer
        ↓
Streamlit dashboard
```

The default Version 2 pipeline queries PostgreSQL before model development.
The CSV path remains an explicit fallback, and canonical hashes require both
source paths and both temporal output files to reconcile exactly.

Checkpoint 63 also executes the independent temporal SQL in a read-only
transaction and compares all 24,082 rows and 52 columns with the Python
builder. The first complete comparison exposed 13,512 historical target nulls
caused by PostgreSQL three-valued logic for missing termination dates. An
explicit `COALESCE(..., FALSE)` corrected the SQL, after which the
implementations had zero key, null-pattern, or value mismatches and one shared
canonical SHA-256 fingerprint.

This is evidence that the SQL is executed and tested, not an unverified
reference artifact.
