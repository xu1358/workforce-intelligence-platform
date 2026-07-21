# Workforce Intelligence and Retention Decision Platform

An end-to-end workforce analytics and employee retention portfolio project built with Python, PostgreSQL, SQL, scikit-learn, and Streamlit.

The platform integrates synthetic data from multiple fictional HR systems into a centralized PostgreSQL database, provides workforce and recruiting analytics, builds an employee attrition classification model, and presents decision-support insights through an interactive dashboard.

> This project uses entirely synthetic employee and recruiting data. It is designed for educational and portfolio purposes.

---

## Project Overview

Organizations often store workforce information across separate systems for:

- Employee records
- Recruiting
- Compensation
- Performance management
- Learning and development
- Employee events

This project simulates that environment and builds a complete analytics workflow that transforms fragmented synthetic HR data into a workforce intelligence platform.

The project includes:

- Synthetic workforce data generation
- PostgreSQL relational data warehouse
- SQL workforce analytics
- Recruiting funnel analysis
- Compensation and performance analytics
- Employee retention modeling
- Classification model comparison
- Model interpretation
- Classification threshold analysis
- Employee risk segmentation
- Dashboard-ready data pipelines
- Interactive Streamlit dashboard

---

## Business Questions

The platform is designed to support questions such as:

- How many employees are currently active?
- How is headcount distributed across departments and locations?
- What is the annual employee turnover rate?
- Which recruiting sources generate the most applications and hires?
- How long do job requisitions remain open?
- Which employees receive higher attrition-risk scores?
- Which workforce characteristics are associated with model-predicted attrition risk?
- How do Logistic Regression, Random Forest, and Gradient Boosting compare?
- How does changing the classification threshold affect precision and recall?

---

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | pandas, NumPy |
| Synthetic Data | Faker |
| Database | PostgreSQL |
| Database Connectivity | psycopg2 |
| Analytics | SQL |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git |

---

## Project Architecture

```text
Synthetic Source Systems
        │
        ├── HRIS
        ├── Recruiting
        ├── Compensation
        ├── Performance
        ├── Learning & Training
        └── Employee Events
        │
        ▼
Python Data Generation
        │
        ▼
Raw CSV Data
        │
        ▼
PostgreSQL
        │
        ├── Workforce Analytics
        ├── Recruiting Analytics
        └── Historical Employee Data
        │
        ▼
Retention Modeling Dataset
        │
        ▼
Machine Learning Pipeline
        │
        ├── Logistic Regression
        ├── Random Forest
        └── Gradient Boosting
        │
        ▼
Model Evaluation & Interpretation
        │
        ├── PR-AUC / ROC-AUC
        ├── Precision / Recall
        ├── Feature Importance
        └── Threshold Analysis
        │
        ▼
Retention Risk Segmentation
        │
        ▼
Dashboard Data Layer
        │
        ▼
Interactive Streamlit Dashboard
```

Detailed architecture documentation is available in:

`docs/architecture.md`

---

## Synthetic Data Environment

The project simulates a fictional company with approximately:

- 10,000 employees
- 40,000 job candidates
- 8 departments
- 5 locations
- 20 job roles
- Multiple years of historical workforce activity

The PostgreSQL database contains 12 main tables:

```text
departments
locations
job_roles
employees
candidates
job_requisitions
applications
compensation_history
performance_reviews
training_programs
training_records
employee_events
```

The data represents six fictional source systems:

1. Human Resources Information System
2. Recruiting System
3. Compensation System
4. Performance Review System
5. Learning and Training System
6. Employee Event System

---

## Data Engineering Pipeline

Python scripts generate synthetic data for:

- Organizational structure
- Employee demographics
- Reporting hierarchies
- Compensation history
- Performance reviews
- Training activity
- Employee events
- Job candidates
- Job requisitions
- Job applications

The generated CSV files are loaded into PostgreSQL using a Python ingestion pipeline.

The database schema includes:

- Primary keys
- Foreign keys
- Data validation constraints
- Relational integrity checks
- Supporting indexes

The project validates that CSV record counts and PostgreSQL table counts match after loading.

---

## SQL Analytics

The PostgreSQL analytics layer supports:

- Active headcount by department
- Active headcount by location
- Workforce distribution by job family
- Annual turnover analysis
- Recruiting funnel analysis
- Requisition duration analysis
- Compensation analysis
- Performance analysis
- Training participation analysis
- Combined employee analytics

SQL analysis is documented in:

`docs/analytics.md`

---

## Employee Retention Modeling

A historical employee snapshot is used to create a supervised classification dataset.

The model predicts:

```text
attrition_next_12m
```

where:

```text
1 = Employee terminates during the following 12 months
0 = Employee does not terminate during the following 12 months
```

The modeling workflow uses historical information available at the snapshot date to reduce target leakage.

Feature groups include:

- Employee characteristics
- Hiring context
- Compensation
- Compensation changes
- Performance
- Training
- Promotions
- Transfers
- Manager changes
- Leave activity

Direct leakage variables such as future termination information are excluded from model features.

---

## Model Comparison

Three classification approaches were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

The selected model was:

```text
Logistic Regression
```

Selected-model test performance:

| Metric | Result |
|---|---:|
| Accuracy | 0.5981 |
| Precision | 0.1376 |
| Recall | 0.7120 |
| F1 Score | 0.2306 |
| ROC-AUC | 0.6968 |
| PR-AUC | 0.1709 |

PR-AUC was used as the primary model-selection metric because attrition represents the minority class.

The model is intended primarily as a portfolio demonstration of an end-to-end classification workflow rather than a production employee decision system.

---

## Threshold Analysis

The default classification threshold of `0.50` was evaluated against thresholds ranging from `0.10` to `0.90`.

The project threshold-selection rule is:

```text
Maintain recall of at least 60%
        ↓
Among eligible thresholds,
select the threshold with the highest precision
```

The resulting recommended threshold was:

```text
0.50
```

This is a project-level decision rule rather than a universal business rule.

---

## Retention Risk Segmentation

Employees are ranked using model-generated attrition-risk scores.

The dashboard groups employees into:

```text
High Risk    → Top 10% of model scores
Medium Risk  → Next 20%
Low Risk     → Remaining 70%
```

In the held-out test population, observed attrition rates were approximately:

| Risk Segment | Observed Attrition Rate |
|---|---:|
| Low | 5.71% |
| Medium | 13.18% |
| High | 18.24% |

The increasing attrition rate across risk groups indicates that the model provides useful relative risk ranking within the synthetic test population.

Model probabilities are not calibrated and should not be interpreted as exact real-world probabilities of employee departure.

---

## Dashboard Preview

### Executive Workforce Overview

The Overview dashboard presents high-level workforce, recruiting, and retention-risk KPIs.

### Employee Retention Risk

The Retention Risk dashboard provides model-based employee risk rankings and interactive department, region, and risk-segment filters.

### Model Performance

The Model Performance dashboard compares Logistic Regression, Random Forest, and Gradient Boosting and presents the selected model's evaluation metrics.

## Interactive Dashboard

The Streamlit dashboard contains five main sections.

### Overview

Displays:

- Active headcount
- Turnover rate
- Open requisitions
- Total applications
- Hired applications
- High-risk employee counts
- Department headcount
- Retention-risk distribution

### Workforce

Analyzes:

- Headcount by department
- Headcount by location
- Workforce distribution

### Recruiting

Analyzes:

- Recruiting outcomes by source
- Hire rates
- Requisition counts
- Target headcount
- Average days open

### Retention Risk

Provides:

- Employee risk rankings
- Low-, Medium-, and High-risk segments
- Risk-score distributions
- Department filters
- Region filters
- Risk-segment filters

### Model Performance

Compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

and displays:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Classification threshold
- Risk-segmentation thresholds

---

## Project Structure

```text
workforce-intelligence-platform/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
│
├── docs/
│   ├── analytics.md
│   ├── architecture.md
│   ├── dashboard.md
│   ├── dashboard_data.md
│   ├── database_setup.md
│   └── retention_modeling.md
│
├── models/
│
├── notebooks/
│
├── sql/
│   ├── create_tables.sql
│   ├── retention_modeling_dataset.sql
│   ├── validate_loaded_data.sql
│   └── workforce_analytics.sql
│
├── src/
│   ├── build_dashboard_data.py
│   ├── build_retention_dataset.py
│   ├── compare_retention_models.py
│   ├── create_database_schema.py
│   ├── load_postgresql_data.py
│   ├── train_baseline_retention_model.py
│   └── analyze_retention_model.py
│
├── .env.example
├── .gitignore
└── README.md
```

Additional data-generation and validation scripts are located in `src/` and `notebooks/`.

---

## Installation

Clone the repository and create a Python virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

## Running the Project

### 1. Activate the virtual environment

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Configure PostgreSQL

Create a `.env` file using `.env.example` as a template.

Example:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=workforce_intelligence
DB_USER=postgres
DB_PASSWORD=your_password
```

The real `.env` file is excluded from Git.

### 3. Build the database schema

```powershell
python src\create_database_schema.py
```

### 4. Load generated data

```powershell
python src\load_postgresql_data.py
```

### 5. Build the retention dataset

```powershell
python src\build_retention_dataset.py
```

### 6. Train the baseline model

```powershell
python src\train_baseline_retention_model.py
```

### 7. Compare retention models

```powershell
python src\compare_retention_models.py
```

### 8. Analyze the selected model

```powershell
python src\analyze_retention_model.py
```

### 9. Build dashboard datasets

```powershell
python src\build_dashboard_data.py
```

### 10. Start the dashboard

```powershell
python -m streamlit run dashboard\app.py
```

The dashboard will normally be available locally at:

```text
http://localhost:8501
```

## End-to-End Validation

The complete project pipeline can be validated with:

```powershell
.\scripts\run_end_to_end.ps1

---

## Project Documentation

Detailed documentation is available under `docs/`:

| Document | Purpose |
|---|---|
| `database_setup.md` | PostgreSQL setup and data loading |
| `analytics.md` | SQL workforce analytics |
| `retention_modeling.md` | Retention dataset and machine-learning workflow |
| `dashboard_data.md` | Dashboard data preparation |
| `dashboard.md` | Interactive dashboard |
| `architecture.md` | End-to-end system architecture |

---

## Limitations

This project has several important limitations:

- All employee and recruiting data is synthetic.
- Model relationships should not be interpreted as causal.
- Attrition probabilities are not calibrated.
- Hyperparameter tuning is limited.
- Model comparison uses a fixed held-out test set.
- The system is not designed for automated employment decisions.
- A production HR analytics system would require additional security, privacy, fairness, governance, monitoring, and human oversight.

---

## Future Improvements

Potential future improvements include:

- Model hyperparameter optimization
- Cross-validation
- Probability calibration
- SHAP-based model interpretation
- Fairness and bias analysis
- Role-based dashboard access
- Automated pipeline orchestration
- Containerization with Docker
- Cloud database deployment
- Cloud dashboard deployment
- Automated model monitoring

---

## Disclaimer

This project is a synthetic portfolio demonstration.

No real employee, candidate, compensation, performance, or recruiting information is used.

The retention model and dashboard are intended to demonstrate data engineering, analytics, machine learning, and visualization skills and should not be used to make real employment decisions.