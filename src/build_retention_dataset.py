from pathlib import Path
import os

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

ENV_PATH = (
    PROJECT_ROOT
    / ".env"
)

SQL_PATH = (
    PROJECT_ROOT
    / "sql"
    / "retention_modeling_dataset.sql"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "retention_modeling_dataset.csv"
)


# ---------------------------------------------------------
# Expected columns
# ---------------------------------------------------------

EXPECTED_COLUMNS = [
    "employee_id",
    "snapshot_date",
    "prediction_end_date",
    "approx_age",
    "tenure_years",
    "employment_type",
    "education_level",
    "application_source",
    "years_experience_at_hire",
    "hire_department_name",
    "hire_region",
    "hire_job_family",
    "hire_job_level",
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
    "promotion_recommended",
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
    "attrition_next_12m",
]


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

def load_database_config() -> dict:
    """Load PostgreSQL settings."""

    if not ENV_PATH.exists():
        raise FileNotFoundError(
            "Missing .env file."
        )

    load_dotenv(
        ENV_PATH
    )

    config = {
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

    missing_settings = [
        key
        for key, value
        in config.items()
        if not value
    ]

    if missing_settings:
        raise ValueError(
            "Missing database settings: "
            f"{missing_settings}"
        )

    return config


# ---------------------------------------------------------
# Load SQL query
# ---------------------------------------------------------

def load_sql_query() -> str:
    """Read retention modeling SQL."""

    if not SQL_PATH.exists():
        raise FileNotFoundError(
            "Missing retention modeling "
            "SQL file."
        )

    return SQL_PATH.read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------
# Query PostgreSQL
# ---------------------------------------------------------

def query_retention_dataset() -> pd.DataFrame:
    """Build the modeling dataset from PostgreSQL."""

    config = (
        load_database_config()
    )

    query = (
        load_sql_query()
    )

    connection = None

    try:
        connection = (
            psycopg2.connect(
                **config
            )
        )

        with connection.cursor() as cursor:

            cursor.execute(
                query
            )

            columns = [
                description[0]
                for description
                in cursor.description
            ]

            rows = (
                cursor.fetchall()
            )

        dataset = pd.DataFrame(
            rows,
            columns=columns,
        )

        return dataset

    finally:
        if connection is not None:
            connection.close()


# ---------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------

def validate_dataset(
    dataset: pd.DataFrame,
) -> None:
    """Validate the retention dataset."""

    if dataset.empty:
        raise ValueError(
            "Retention dataset is empty."
        )

    if (
        dataset.columns.tolist()
        != EXPECTED_COLUMNS
    ):
        raise ValueError(
            "Unexpected retention dataset "
            "columns."
        )

    if not dataset[
        "employee_id"
    ].is_unique:
        raise ValueError(
            "employee_id is not unique."
        )

    if dataset[
        "employee_id"
    ].isna().any():
        raise ValueError(
            "Missing employee IDs."
        )

    valid_targets = {
        0,
        1,
    }

    actual_targets = set(
        dataset[
            "attrition_next_12m"
        ].astype(int)
    )

    if not actual_targets.issubset(
        valid_targets
    ):
        raise ValueError(
            "Invalid attrition target."
        )

    leakage_columns = {
        "employment_status",
        "termination_date",
        "termination_type",
    }

    if leakage_columns.intersection(
        dataset.columns
    ):
        raise ValueError(
            "Potential target leakage "
            "columns detected."
        )

    snapshot_dates = set(
        pd.to_datetime(
            dataset[
                "snapshot_date"
            ]
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )

    if snapshot_dates != {
        "2025-06-30"
    }:
        raise ValueError(
            "Unexpected snapshot date."
        )

    prediction_dates = set(
        pd.to_datetime(
            dataset[
                "prediction_end_date"
            ]
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )

    if prediction_dates != {
        "2026-06-30"
    }:
        raise ValueError(
            "Unexpected prediction end date."
        )

    attrition_count = int(
        dataset[
            "attrition_next_12m"
        ].sum()
    )

    if attrition_count <= 0:
        raise ValueError(
            "No positive attrition "
            "examples were generated."
        )

    if (
        attrition_count
        >= len(dataset)
    ):
        raise ValueError(
            "Every employee has the "
            "positive attrition target."
        )


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

def save_dataset(
    dataset: pd.DataFrame,
) -> None:
    """Save the processed modeling dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

def print_summary(
    dataset: pd.DataFrame,
) -> None:
    """Print modeling dataset statistics."""

    print(
        "\nRetention modeling dataset:"
    )

    print(
        f"Rows: "
        f"{len(dataset):,}"
    )

    print(
        f"Columns: "
        f"{len(dataset.columns)}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        dataset[
            "attrition_next_12m"
        ]
        .value_counts()
        .sort_index()
    )

    attrition_rate = (
        100
        * dataset[
            "attrition_next_12m"
        ].mean()
    )

    print(
        "\nAttrition rate:"
    )

    print(
        f"{attrition_rate:.2f}%"
    )

    print(
        "\nEmployees with "
        "performance reviews:"
    )

    print(
        dataset[
            "performance_rating"
        ].notna().sum()
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Build and save retention dataset."""

    dataset = (
        query_retention_dataset()
    )

    validate_dataset(
        dataset
    )

    save_dataset(
        dataset
    )

    print_summary(
        dataset
    )

    print(
        "\nRetention modeling dataset "
        "built successfully."
    )


if __name__ == "__main__":
    main()