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
- Built automated CSV-to-PostgreSQL ingestion and validation pipelines.
- Developed SQL analytics for workforce, turnover, recruiting, compensation, performance, and training.
- Created a leakage-aware employee retention modeling dataset.
- Compared Logistic Regression, Random Forest, and Gradient Boosting.
- Selected the final model primarily using PR-AUC.
- Evaluated classification thresholds and retention-risk segments.
- Built a five-section interactive Streamlit dashboard.
- Added one cross-platform Python pipeline with a thin PowerShell wrapper.

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
SQL analytics
        ↓
Retention modeling dataset
        ↓
Machine-learning models
        ↓
Risk segmentation
        ↓
Dashboard data layer
        ↓
Streamlit dashboard
```
