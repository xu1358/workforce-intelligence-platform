from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import psycopg2

from dotenv import load_dotenv


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DASHBOARD_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
)

RETENTION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_modeling_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "selected_retention_model.joblib"
)

MODEL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_model_comparison.csv"
)

THRESHOLD_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_threshold_analysis.csv"
)

AS_OF_DATE = "2026-06-30"


# ---------------------------------------------------------
# Model feature definitions
# ---------------------------------------------------------

NUMERICAL_FEATURES = [
    "approx_age",
    "tenure_years",
    "years_experience_at_hire",
    "initial_base_salary",
    "base_salary",
    "bonus_target",
    "equity_value",
    "salary_growth_percent",
    "days_since_compensation_change",
    "compensation_record_count",
    "promotion_compensation_count",
    "performance_rating",
    "goal_completion",
    "days_since_review",
    "review_count",
    "average_performance_rating",
    "completed_training_programs",
    "failed_training_programs",
    "in_progress_training_programs",
    "completed_training_hours",
    "average_training_score",
    "prior_promotion_events",
    "prior_transfer_events",
    "prior_manager_change_events",
    "prior_leave_events",
    "prior_change_events",
    "days_since_last_change_event",
]


CATEGORICAL_FEATURES = [
    "employment_type",
    "education_level",
    "application_source",
    "hire_department_name",
    "hire_region",
    "hire_job_family",
    "hire_job_level",
    "promotion_recommended",
]


FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# ---------------------------------------------------------
# Database connection
# ---------------------------------------------------------

def connect_database():
    """Connect to PostgreSQL using .env settings."""

    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    settings = {
        "host": os.getenv(
            "DB_HOST"
        ),
        "port": os.getenv(
            "DB_PORT"
        ),
        "dbname": os.getenv(
            "DB_NAME"
        ),
        "user": os.getenv(
            "DB_USER"
        ),
        "password": os.getenv(
            "DB_PASSWORD"
        ),
    }

    missing = [
        key
        for key, value
        in settings.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing database settings: "
            f"{missing}"
        )

    return psycopg2.connect(
        **settings
    )


def query_dataframe(
    connection,
    query,
    parameters=None,
) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""

    with connection.cursor() as cursor:

        cursor.execute(
            query,
            parameters,
        )

        rows = (
            cursor.fetchall()
        )

        columns = [
            description[0]
            for description
            in cursor.description
        ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# ---------------------------------------------------------
# Workforce overview
# ---------------------------------------------------------

def load_overview_kpis(
    connection,
) -> pd.DataFrame:
    """Build high-level workforce KPIs."""

    query = """
    WITH params AS (
        SELECT %s::date AS as_of_date
    ),

    active_current AS (
        SELECT e.*
        FROM employees e
        CROSS JOIN params p
        WHERE
            e.hire_date <= p.as_of_date
            AND (
                e.termination_date IS NULL
                OR e.termination_date > p.as_of_date
            )
    ),

    start_2025 AS (
        SELECT COUNT(*) AS headcount
        FROM employees
        WHERE
            hire_date <= DATE '2025-01-01'
            AND (
                termination_date IS NULL
                OR termination_date > DATE '2025-01-01'
            )
    ),

    end_2025 AS (
        SELECT COUNT(*) AS headcount
        FROM employees
        WHERE
            hire_date <= DATE '2025-12-31'
            AND (
                termination_date IS NULL
                OR termination_date > DATE '2025-12-31'
            )
    ),

    terminations_2025 AS (
        SELECT COUNT(*) AS termination_count
        FROM employees
        WHERE
            termination_date
            BETWEEN DATE '2025-01-01'
            AND DATE '2025-12-31'
    )

    SELECT

        (SELECT COUNT(*)
         FROM active_current)
        AS active_headcount,

        (SELECT COUNT(
            DISTINCT department_id
         )
         FROM active_current)
        AS active_department_count,

        (SELECT COUNT(
            DISTINCT location_id
         )
         FROM active_current)
        AS active_location_count,

        (SELECT termination_count
         FROM terminations_2025)
        AS terminations_2025,

        ROUND(
            100.0
            * (
                SELECT termination_count
                FROM terminations_2025
            )
            /
            NULLIF(
                (
                    (
                        SELECT headcount
                        FROM start_2025
                    )
                    +
                    (
                        SELECT headcount
                        FROM end_2025
                    )
                )
                / 2.0,
                0
            ),
            2
        )
        AS turnover_rate_2025_percent,

        (
            SELECT COUNT(*)
            FROM job_requisitions
            WHERE requisition_status = 'Open'
        )
        AS open_requisitions,

        (
            SELECT COUNT(*)
            FROM applications
        )
        AS total_applications,

        (
            SELECT COUNT(*)
            FROM applications
            WHERE application_status = 'Hired'
        )
        AS hired_applications
    ;
    """

    overview = query_dataframe(
        connection,
        query,
        (
            AS_OF_DATE,
        ),
    )

    overview.insert(
        0,
        "as_of_date",
        AS_OF_DATE,
    )

    return overview


# ---------------------------------------------------------
# Headcount by department
# ---------------------------------------------------------

def load_headcount_by_department(
    connection,
) -> pd.DataFrame:
    """Build active headcount by department."""

    query = """
    WITH active_employees AS (

        SELECT e.*
        FROM employees e

        WHERE
            e.hire_date <= %s::date
            AND (
                e.termination_date IS NULL
                OR e.termination_date > %s::date
            )
    )

    SELECT

        d.department_name,

        COUNT(*) AS active_headcount,

        ROUND(
            100.0
            * COUNT(*)
            /
            SUM(
                COUNT(*)
            ) OVER (),
            2
        )
        AS headcount_percent

    FROM active_employees e

    JOIN departments d
        ON e.department_id
        = d.department_id

    GROUP BY
        d.department_name

    ORDER BY
        active_headcount DESC
    ;
    """

    return query_dataframe(
        connection,
        query,
        (
            AS_OF_DATE,
            AS_OF_DATE,
        ),
    )


# ---------------------------------------------------------
# Headcount by location
# ---------------------------------------------------------

def load_headcount_by_location(
    connection,
) -> pd.DataFrame:
    """Build active headcount by location."""

    query = """
    WITH active_employees AS (

        SELECT e.*
        FROM employees e

        WHERE
            e.hire_date <= %s::date
            AND (
                e.termination_date IS NULL
                OR e.termination_date > %s::date
            )
    )

    SELECT

        CONCAT(
            l.city,
            ', ',
            l.state
        ) AS location_name,

        l.region,

        COUNT(*) AS active_headcount,

        ROUND(
            100.0
            * COUNT(*)
            /
            SUM(
                COUNT(*)
            ) OVER (),
            2
        )
        AS headcount_percent

    FROM active_employees e

    JOIN locations l
        ON e.location_id
        = l.location_id

    GROUP BY
        l.city,
        l.state,
        l.region

    ORDER BY
        active_headcount DESC
    ;
    """

    return query_dataframe(
        connection,
        query,
        (
            AS_OF_DATE,
            AS_OF_DATE,
        ),
    )


# ---------------------------------------------------------
# Recruiting funnel
# ---------------------------------------------------------

def load_recruiting_funnel(
    connection,
) -> pd.DataFrame:
    """Build recruiting funnel by application source."""

    query = """
    SELECT

        c.application_source,

        COUNT(*)
        AS total_applications,

        COUNT(*) FILTER (
            WHERE
                a.application_status
                = 'Hired'
        )
        AS hired,

        COUNT(*) FILTER (
            WHERE
                a.application_status
                = 'Rejected'
        )
        AS rejected,

        COUNT(*) FILTER (
            WHERE
                a.application_status
                = 'Withdrawn'
        )
        AS withdrawn,

        COUNT(*) FILTER (
            WHERE
                a.application_status
                = 'Offer Declined'
        )
        AS offer_declined,

        COUNT(*) FILTER (
            WHERE
                a.application_status
                = 'In Process'
        )
        AS in_process,

        COUNT(*) FILTER (
            WHERE
                a.application_status
                = 'Position Cancelled'
        )
        AS position_cancelled,

        ROUND(
            100.0
            * COUNT(*) FILTER (
                WHERE
                    a.application_status
                    = 'Hired'
            )
            /
            NULLIF(
                COUNT(*),
                0
            ),
            2
        )
        AS hire_rate_percent

    FROM applications a

    JOIN candidates c
        ON a.candidate_id
        = c.candidate_id

    GROUP BY
        c.application_source

    ORDER BY
        total_applications DESC
    ;
    """

    return query_dataframe(
        connection,
        query,
    )


# ---------------------------------------------------------
# Requisition metrics
# ---------------------------------------------------------

def load_requisition_metrics(
    connection,
) -> pd.DataFrame:
    """Build requisition summary metrics."""

    query = """
    SELECT

        requisition_status,

        COUNT(*)
        AS requisition_count,

        SUM(
            target_headcount
        )
        AS target_headcount,

        ROUND(
            AVG(
                CASE

                    WHEN
                        close_date
                        IS NOT NULL

                    THEN
                        close_date
                        - open_date

                    ELSE
                        %s::date
                        - open_date

                END
            ),
            1
        )
        AS average_days_open

    FROM job_requisitions

    GROUP BY
        requisition_status

    ORDER BY
        requisition_count DESC
    ;
    """

    return query_dataframe(
        connection,
        query,
        (
            AS_OF_DATE,
        ),
    )


# ---------------------------------------------------------
# Load modeling outputs
# ---------------------------------------------------------

def load_modeling_files():
    """Load model and modeling datasets."""

    required_paths = [
        RETENTION_DATA_PATH,
        MODEL_PATH,
        MODEL_COMPARISON_PATH,
        THRESHOLD_ANALYSIS_PATH,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing required modeling files: "
            f"{missing_paths}"
        )

    retention = pd.read_csv(
        RETENTION_DATA_PATH
    )

    comparison = pd.read_csv(
        MODEL_COMPARISON_PATH
    )

    threshold_analysis = pd.read_csv(
        THRESHOLD_ANALYSIS_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    return (
        retention,
        comparison,
        threshold_analysis,
        model,
    )


# ---------------------------------------------------------
# Recommended threshold
# ---------------------------------------------------------

def get_recommended_threshold(
    threshold_analysis,
) -> float:
    """Use the Checkpoint 23 threshold rule."""

    eligible = (
        threshold_analysis[
            threshold_analysis[
                "recall"
            ]
            >= 0.60
        ]
        .copy()
    )

    if not eligible.empty:

        selected = (
            eligible
            .sort_values(
                [
                    "precision",
                    "f1",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

    else:

        selected = (
            threshold_analysis
            .sort_values(
                "f1",
                ascending=False,
            )
            .iloc[0]
        )

    return float(
        selected[
            "threshold"
        ]
    )


# ---------------------------------------------------------
# Full-population retention scoring
# ---------------------------------------------------------

def build_retention_risk_data(
    retention,
    model,
    selected_model_name,
):
    """
    Score the full snapshot population.

    These scores are for dashboard prioritization.
    They are not used as unbiased test-set
    performance estimates.
    """

    X = (
        retention[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    high_threshold = float(
        np.quantile(
            probabilities,
            0.90,
        )
    )

    medium_threshold = float(
        np.quantile(
            probabilities,
            0.70,
        )
    )

    risk_segments = np.select(
        [
            probabilities
            >= high_threshold,

            probabilities
            >= medium_threshold,
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    risk_employees = (
        retention[
            [
                "employee_id",
                "snapshot_date",
                "tenure_years",
                "employment_type",
                "hire_department_name",
                "hire_region",
                "hire_job_family",
                "hire_job_level",
                "base_salary",
                "performance_rating",
                "average_performance_rating",
                "completed_training_programs",
                "prior_change_events",
            ]
        ]
        .copy()
    )

    risk_employees[
        "attrition_probability"
    ] = probabilities

    risk_employees[
        "risk_segment"
    ] = risk_segments

    risk_employees[
        "model"
    ] = selected_model_name

    risk_employees = (
        risk_employees
        .sort_values(
            "attrition_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    risk_summary = (
        risk_employees
        .groupby(
            "risk_segment",
            as_index=False,
        )
        .agg(
            employee_count=(
                "employee_id",
                "count",
            ),

            average_attrition_probability=(
                "attrition_probability",
                "mean",
            ),

            average_tenure_years=(
                "tenure_years",
                "mean",
            ),

            average_base_salary=(
                "base_salary",
                "mean",
            ),
        )
    )

    risk_summary[
        "population_percent"
    ] = (
        100
        * risk_summary[
            "employee_count"
        ]
        / len(
            risk_employees
        )
    )

    order = {
        "High": 1,
        "Medium": 2,
        "Low": 3,
    }

    risk_summary[
        "sort_order"
    ] = (
        risk_summary[
            "risk_segment"
        ]
        .map(
            order
        )
    )

    risk_summary = (
        risk_summary
        .sort_values(
            "sort_order"
        )
        .drop(
            columns=[
                "sort_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        risk_employees,
        risk_summary,
        medium_threshold,
        high_threshold,
    )


# ---------------------------------------------------------
# Model dashboard tables
# ---------------------------------------------------------

def build_model_tables(
    comparison,
    threshold_analysis,
    medium_risk_threshold,
    high_risk_threshold,
):
    """Prepare model-performance dashboard tables."""

    model_performance = (
        comparison
        .copy()
    )

    selected_model_name = (
        model_performance
        .sort_values(
            [
                "pr_auc",
                "roc_auc",
            ],
            ascending=False,
        )
        .iloc[0][
            "model"
        ]
    )

    model_performance[
        "selected_model"
    ] = (
        model_performance[
            "model"
        ]
        == selected_model_name
    )

    selected_row = (
        model_performance[
            model_performance[
                "model"
            ]
            == selected_model_name
        ]
        .iloc[0]
    )

    recommended_threshold = (
        get_recommended_threshold(
            threshold_analysis
        )
    )

    model_summary = pd.DataFrame(
        [
            {
                "selected_model": (
                    selected_model_name
                ),

                "selection_metric": (
                    "PR-AUC"
                ),

                "recommended_classification_threshold": (
                    recommended_threshold
                ),

                "medium_risk_threshold": (
                    medium_risk_threshold
                ),

                "high_risk_threshold": (
                    high_risk_threshold
                ),

                "test_accuracy": (
                    selected_row[
                        "accuracy"
                    ]
                ),

                "test_precision": (
                    selected_row[
                        "precision"
                    ]
                ),

                "test_recall": (
                    selected_row[
                        "recall"
                    ]
                ),

                "test_f1": (
                    selected_row[
                        "f1"
                    ]
                ),

                "test_roc_auc": (
                    selected_row[
                        "roc_auc"
                    ]
                ),

                "test_pr_auc": (
                    selected_row[
                        "pr_auc"
                    ]
                ),
            }
        ]
    )

    return (
        model_performance,
        model_summary,
        selected_model_name,
    )


# ---------------------------------------------------------
# Save files
# ---------------------------------------------------------

def save_dashboard_file(
    dataframe,
    filename,
):
    """Save one dashboard dataset."""

    path = (
        DASHBOARD_DIR
        / filename
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    """Build all dashboard-ready datasets."""

    DASHBOARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Building dashboard data layer..."
    )

    # -----------------------------------------------------
    # Load database data
    # -----------------------------------------------------

    connection = (
        connect_database()
    )

    try:

        overview = (
            load_overview_kpis(
                connection
            )
        )

        headcount_department = (
            load_headcount_by_department(
                connection
            )
        )

        headcount_location = (
            load_headcount_by_location(
                connection
            )
        )

        recruiting_funnel = (
            load_recruiting_funnel(
                connection
            )
        )

        requisition_metrics = (
            load_requisition_metrics(
                connection
            )
        )

    finally:

        connection.close()

    # -----------------------------------------------------
    # Load model data
    # -----------------------------------------------------

    (
        retention,
        comparison,
        threshold_analysis,
        model,
    ) = load_modeling_files()

    selected_model_name = (
        comparison
        .sort_values(
            [
                "pr_auc",
                "roc_auc",
            ],
            ascending=False,
        )
        .iloc[0][
            "model"
        ]
    )

    (
        risk_employees,
        risk_summary,
        medium_risk_threshold,
        high_risk_threshold,
    ) = build_retention_risk_data(
        retention,
        model,
        selected_model_name,
    )

    (
        model_performance,
        model_summary,
        selected_model_name,
    ) = build_model_tables(
        comparison,
        threshold_analysis,
        medium_risk_threshold,
        high_risk_threshold,
    )

    # -----------------------------------------------------
    # Add retention KPIs
    # -----------------------------------------------------

    risk_counts = (
        risk_employees[
            "risk_segment"
        ]
        .value_counts()
    )

    overview[
        "retention_snapshot_population"
    ] = len(
        risk_employees
    )

    overview[
        "high_risk_employees"
    ] = int(
        risk_counts.get(
            "High",
            0,
        )
    )

    overview[
        "medium_risk_employees"
    ] = int(
        risk_counts.get(
            "Medium",
            0,
        )
    )

    overview[
        "low_risk_employees"
    ] = int(
        risk_counts.get(
            "Low",
            0,
        )
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    active_headcount = int(
        overview.loc[
            0,
            "active_headcount",
        ]
    )

    if (
        int(
            headcount_department[
                "active_headcount"
            ]
            .sum()
        )
        != active_headcount
    ):
        raise ValueError(
            "Department headcount does not "
            "match overview headcount."
        )

    if (
        int(
            headcount_location[
                "active_headcount"
            ]
            .sum()
        )
        != active_headcount
    ):
        raise ValueError(
            "Location headcount does not "
            "match overview headcount."
        )

    if (
        len(
            risk_employees
        )
        != len(
            retention
        )
    ):
        raise ValueError(
            "Retention risk row count mismatch."
        )

    if (
        set(
            risk_employees[
                "risk_segment"
            ]
        )
        != {
            "Low",
            "Medium",
            "High",
        }
    ):
        raise ValueError(
            "Expected Low, Medium, and High "
            "risk segments."
        )

    if (
        model_performance[
            "selected_model"
        ]
        .sum()
        != 1
    ):
        raise ValueError(
            "Expected exactly one "
            "selected model."
        )

    # -----------------------------------------------------
    # Save dashboard files
    # -----------------------------------------------------

    save_dashboard_file(
        overview,
        "overview_kpis.csv",
    )

    save_dashboard_file(
        headcount_department,
        "headcount_by_department.csv",
    )

    save_dashboard_file(
        headcount_location,
        "headcount_by_location.csv",
    )

    save_dashboard_file(
        recruiting_funnel,
        "recruiting_funnel.csv",
    )

    save_dashboard_file(
        requisition_metrics,
        "requisition_metrics.csv",
    )

    save_dashboard_file(
        risk_summary,
        "retention_risk_summary.csv",
    )

    save_dashboard_file(
        risk_employees,
        "retention_risk_employees.csv",
    )

    save_dashboard_file(
        model_performance,
        "model_performance.csv",
    )

    save_dashboard_file(
        model_summary,
        "model_summary.csv",
    )

    print(
        "\nSelected model:"
    )

    print(
        selected_model_name
    )

    print(
        "\nDashboard data layer "
        "built successfully."
    )


if __name__ == "__main__":
    main()