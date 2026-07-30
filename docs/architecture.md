# Workforce Intelligence Platform Architecture

## 1. Architecture Overview

The Workforce Intelligence and Retention Decision Platform is an end-to-end synthetic workforce analytics system.

The architecture contains five major layers:

```text
1. Synthetic Data Sources
        ↓
2. Data Engineering and PostgreSQL
        ↓
3. SQL Analytics
        ↓
4. Machine Learning
        ↓
5. Decision-Support Dashboard
```

Each layer is separated so that data generation, storage, analytics, modeling, and visualization can evolve independently.

---

## 2. End-to-End Data Flow

```text
Synthetic HR Source Systems
        │
        ▼
Python Data Generators
        │
        ▼
Raw CSV Files
        │
        ▼
PostgreSQL Database
        │
        ├───────────────┐
        │               │
        ▼               ▼
SQL Analytics     Retention Dataset
                        │
                        ▼
                 Machine Learning
                        │
                        ├── Logistic Regression
                        ├── Random Forest
                        └── Gradient Boosting
                        │
                        ▼
                 Model Evaluation
                        │
                        ▼
                 Risk Segmentation
                        │
                        ▼
PostgreSQL Analytics + ML Outputs
                │
                ▼
        Dashboard Data Layer
                │
                ▼
        Streamlit Dashboard
```

---

## 3. Synthetic Source Systems

The project represents six fictional enterprise systems.

### Human Resources Information System

Core tables:

```text
employees
departments
locations
job_roles
```

This system contains organizational and employment information.

### Recruiting System

Core tables:

```text
candidates
job_requisitions
applications
```

This system represents candidate sourcing, recruiting activity, requisitions, and hiring outcomes.

### Compensation System

Core table:

```text
compensation_history
```

This system tracks historical compensation changes.

### Performance Review System

Core table:

```text
performance_reviews
```

This system stores historical employee performance evaluations.

### Learning and Training System

Core tables:

```text
training_programs
training_records
```

This system stores employee training activity.

### Employee Event System

Core table:

```text
employee_events
```

This system records events such as:

```text
Hire
Promotion
Transfer
Manager Change
Leave
Termination
```

---

## 4. Data Generation Layer

Python scripts generate synthetic records for each source system.

The generation process creates realistic relationships between entities.

Examples include:

```text
Employee
    ↓
Department

Employee
    ↓
Location

Employee
    ↓
Job Role

Employee
    ↓
Manager

Candidate
    ↓
Application
    ↓
Job Requisition
```

The synthetic workforce also includes a multi-level reporting hierarchy:

```text
Department Head
        ↓
Senior Manager
        ↓
Team Manager
        ↓
Individual Contributor
```

Approximately 10,000 employees are generated.

---

## 5. Raw Data Layer

Generated records are stored as CSV files under:

```text
data/raw/
```

These files provide a reproducible intermediate representation between synthetic data generation and database ingestion.

Raw generated data is excluded from Git.

---

## 6. PostgreSQL Database Layer

The PostgreSQL database centralizes the synthetic workforce data.

Database name:

```text
workforce_intelligence
```

The database contains 12 main relational tables.

Relationships are enforced using:

- Primary keys
- Foreign keys
- Check constraints
- Referential integrity
- Supporting indexes

Python connects to PostgreSQL using:

```text
psycopg2
```

Database credentials are stored locally in:

```text
.env
```

The `.env` file is excluded from Git.

The primary Version 2 pipeline now queries PostgreSQL before model
development. Eight least-privilege analytical views expose only the fields
required by the temporal builder:

```text
sql/create_v2_analytics_views.sql
src/v2_data_access.py
```

The default runner loads the normalized tables, creates these views, builds
the historical and current temporal datasets from PostgreSQL, and validates
canonical content hashes against the explicit CSV fallback.

---

## 7. Database Loading Pipeline

The ingestion workflow is implemented in:

```text
src/load_postgresql_data.py
```

The loader:

1. Validates expected CSV headers.
2. Clears existing table data when required.
3. Loads records in dependency order.
4. Uses PostgreSQL COPY operations.
5. Validates record counts after loading.

Reference tables are loaded before dependent transactional tables.

---

## 8. SQL Analytics Layer

Workforce analytics are implemented using PostgreSQL SQL queries.

The analytics layer includes:

```text
Active workforce headcount
Department workforce distribution
Location workforce distribution
Job-family distribution
Annual turnover
Recruiting funnel
Requisition duration
Compensation analysis
Performance analysis
Training analysis
```

The main SQL analytics file is:

```text
sql/workforce_analytics.sql
```

---

## 9. Retention Modeling Dataset

The retention modeling dataset is created from a historical workforce snapshot.

Snapshot date:

```text
2025-06-30
```

Prediction window:

```text
2025-07-01
through
2026-06-30
```

The target variable is:

```text
attrition_next_12m
```

Definition:

```text
1
Employee terminates during the prediction window

0
Employee does not terminate during the prediction window
```

Only information available on or before the snapshot date is used for feature construction.

This helps reduce future-information leakage.

---

## 10. Machine-Learning Layer

The machine-learning layer evaluates three classification models:

```text
Logistic Regression
Random Forest
Gradient Boosting
```

The dataset is divided using a stratified train/test split.

Models are evaluated using:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
PR-AUC
```

Because employee attrition is an imbalanced classification problem, PR-AUC is used as the primary model-selection metric.

The selected model is:

```text
Logistic Regression
```

---

## 11. Model Interpretation Layer

The selected model is analyzed using feature importance.

For Logistic Regression:

```text
Positive coefficient
→ Associated with higher model-predicted attrition risk

Negative coefficient
→ Associated with lower model-predicted attrition risk
```

These associations should not be interpreted as causal relationships.

Classification thresholds from:

```text
0.10
through
0.90
```

are evaluated.

The project threshold-selection rule requires at least 60% recall and then chooses the eligible threshold with the highest precision.

The selected threshold is:

```text
0.50
```

---

## 12. Risk Segmentation Layer

Employees are ranked using model-generated risk scores.

Risk groups are defined by score distribution:

```text
High Risk
Top 10%

Medium Risk
Next 20%

Low Risk
Remaining 70%
```

The segments provide relative prioritization.

They should not be interpreted as exact probabilities of future employee departure.

---

## 13. Dashboard Data Layer

The dashboard data layer is implemented in:

```text
src/build_dashboard_data.py
```

It combines:

```text
PostgreSQL workforce analytics
+
Recruiting analytics
+
Retention model outputs
```

and creates dashboard-ready datasets under:

```text
data/processed/dashboard/
```

This separates expensive data preparation from interactive dashboard rendering.

---

## 14. Dashboard Layer

The interactive dashboard is implemented using Streamlit:

```text
dashboard/app.py
```

The dashboard contains five sections:

```text
Overview
Workforce
Recruiting
Retention Risk
Model Performance
```

Plotly is used for interactive charts.

The dashboard reads prepared CSV datasets rather than directly executing the complete analytics and modeling workflow during every user interaction.

---

## 15. Design Decisions

### Separate raw data and processed data

The project maintains separate directories for:

```text
data/raw/
data/interim/
data/processed/
```

This improves data-pipeline organization.

### Connect database analytics to machine learning

SQL defines relational constraints and least-privilege analytical views.
Python queries those views, constructs leakage-safe point-in-time features,
and passes the resulting temporal datasets to scikit-learn. Source-table and
final-output hashes ensure that the PostgreSQL and CSV paths do not drift.

### Separate dashboard preparation and presentation

Dashboard data preparation is handled by:

```text
src/build_dashboard_data.py
```

Dashboard visualization is handled by:

```text
dashboard/app.py
```

This keeps visualization code simpler.

### Preserve test-set evaluation

Full-population model scores are used for dashboard prioritization.

Held-out test data remains the basis for reporting model performance.

---

## 16. Security and Privacy Considerations

The current project uses synthetic data.

A production implementation using real employee data would require:

- Authentication
- Authorization
- Encryption
- Role-based access controls
- Data minimization
- Audit logging
- Privacy governance
- Fairness evaluation
- Model monitoring
- Human review

Individual employee risk scores should not be used as the sole basis for employment decisions.

---

## 17. Current Architecture Summary

```text
                    ┌─────────────────────────┐
                    │ Synthetic Source Systems│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Python Data Generation  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Raw CSV Data      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       PostgreSQL        │
                    └───────┬─────────┬───────┘
                            │         │
                ┌───────────┘         └────────────┐
                ▼                                  ▼
       ┌─────────────────┐              ┌─────────────────┐
       │  SQL Analytics  │              │Retention Dataset│
       └────────┬────────┘              └────────┬────────┘
                │                                │
                │                                ▼
                │                       ┌─────────────────┐
                │                       │ Machine Learning│
                │                       └────────┬────────┘
                │                                │
                │                                ▼
                │                       ┌─────────────────┐
                │                       │ Risk Segmentation│
                │                       └────────┬────────┘
                │                                │
                └──────────────┬─────────────────┘
                               ▼
                    ┌─────────────────────────┐
                    │  Dashboard Data Layer   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Streamlit Dashboard    │
                    └─────────────────────────┘
```

This architecture demonstrates an end-to-end workflow spanning data engineering, relational databases, SQL analytics, machine learning, and interactive business intelligence.

## 18. Reproducibility

Direct Python dependencies and the complete resolved environment are
documented in:

`requirements.txt`

`requirements-lock.txt`

Database environment variables are documented using:

`.env.example`

The real `.env` file is excluded from version control.

Generated datasets under:

- `data/raw/`
- `data/interim/`
- `data/processed/`

are excluded from Git.

Trained model artifacts under:

`models/`

are also excluded from Git.

The repository therefore stores the source code and configuration templates required to reproduce the pipeline while avoiding private credentials and generated artifacts.
