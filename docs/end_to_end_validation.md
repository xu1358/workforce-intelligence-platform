# End-to-End Project Validation

## Purpose

The end-to-end validation workflow confirms that the Workforce Intelligence and Retention Decision Platform can be rebuilt from synthetic source data through the final Streamlit dashboard.

The validated workflow includes:

1. Synthetic data generation
2. PostgreSQL connectivity
3. Database schema creation
4. CSV ingestion
5. Database validation
6. Retention dataset construction
7. Baseline model training
8. Model comparison
9. Model interpretation
10. Dashboard data preparation
11. Final output validation

## Full validation

Run the complete pipeline from the project root:

```powershell
.\scripts\run_end_to_end.ps1