from pathlib import Path
import csv
import os

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

ENV_PATH = (
    PROJECT_ROOT
    / ".env"
)


# ---------------------------------------------------------
# Table definitions
# ---------------------------------------------------------

TABLE_COLUMNS = {
    "departments": [
        "department_id",
        "department_name",
        "department_group",
        "cost_center",
    ],

    "locations": [
        "location_id",
        "city",
        "state",
        "region",
        "location_type",
    ],

    "job_roles": [
        "job_role_id",
        "job_title",
        "job_family",
        "job_level",
        "salary_band_min",
        "salary_band_max",
    ],

    "training_programs": [
        "program_id",
        "program_name",
        "program_category",
        "required_hours",
        "mandatory",
    ],

    "employees": [
        "employee_id",
        "first_name",
        "last_name",
        "hire_date",
        "termination_date",
        "employment_status",
        "termination_type",
        "department_id",
        "location_id",
        "job_role_id",
        "manager_id",
        "employment_type",
        "birth_year",
        "education_level",
        "organizational_level",
    ],

    "candidates": [
        "candidate_id",
        "application_source",
        "education_level",
        "years_experience",
        "candidate_location",
    ],

    "job_requisitions": [
        "requisition_id",
        "job_role_id",
        "department_id",
        "location_id",
        "open_date",
        "close_date",
        "target_headcount",
        "recruiter_id",
        "requisition_status",
    ],

    "compensation_history": [
        "compensation_id",
        "employee_id",
        "effective_date",
        "base_salary",
        "bonus_target",
        "equity_value",
        "change_reason",
    ],

    "performance_reviews": [
        "review_id",
        "employee_id",
        "review_date",
        "review_period",
        "performance_rating",
        "goal_completion",
        "promotion_recommended",
        "reviewer_id",
    ],

    "training_records": [
        "training_record_id",
        "employee_id",
        "program_id",
        "start_date",
        "completion_date",
        "completion_status",
        "training_hours",
        "score",
    ],

    "employee_events": [
        "event_id",
        "employee_id",
        "event_date",
        "event_type",
        "old_value",
        "new_value",
        "notes",
    ],

    "applications": [
        "application_id",
        "candidate_id",
        "requisition_id",
        "application_date",
        "application_status",
        "interview_score",
        "offer_date",
        "decision_date",
        "employee_id",
    ],
}


TABLE_LOAD_ORDER = [
    "departments",
    "locations",
    "job_roles",
    "training_programs",
    "employees",
    "candidates",
    "job_requisitions",
    "compensation_history",
    "performance_reviews",
    "training_records",
    "employee_events",
    "applications",
]


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

def load_database_config() -> dict:
    """Load PostgreSQL settings from .env."""

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
# CSV validation
# ---------------------------------------------------------

def validate_csv_files() -> None:
    """Check that every required CSV exists and has the expected header."""

    print(
        "Validating CSV files..."
    )

    for table_name in (
        TABLE_LOAD_ORDER
    ):
        csv_path = (
            RAW_DATA_DIR
            / f"{table_name}.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing CSV file: "
                f"{csv_path.name}"
            )

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as csv_file:

            reader = csv.reader(
                csv_file
            )

            actual_columns = next(
                reader
            )

        expected_columns = (
            TABLE_COLUMNS[
                table_name
            ]
        )

        if (
            actual_columns
            != expected_columns
        ):
            raise ValueError(
                f"{table_name}.csv has "
                "unexpected columns.\n"
                f"Expected: "
                f"{expected_columns}\n"
                f"Actual: "
                f"{actual_columns}"
            )

        print(
            f"Validated: "
            f"{table_name}.csv"
        )

    print(
        "\nAll CSV files validated."
    )


# ---------------------------------------------------------
# Remove existing database data
# ---------------------------------------------------------

def truncate_tables(
    cursor,
) -> None:
    """
    Remove existing rows before reloading.

    This makes the script safe to run again.
    """

    table_identifiers = [
        sql.Identifier(
            table_name
        )
        for table_name
        in TABLE_LOAD_ORDER
    ]

    truncate_query = (
        sql.SQL(
            "TRUNCATE TABLE {} CASCADE;"
        )
        .format(
            sql.SQL(", ").join(
                table_identifiers
            )
        )
    )

    cursor.execute(
        truncate_query
    )

    print(
        "\nExisting database rows removed."
    )


# ---------------------------------------------------------
# Load one table
# ---------------------------------------------------------

def load_table(
    cursor,
    table_name: str,
) -> None:
    """Load one CSV into one PostgreSQL table."""

    csv_path = (
        RAW_DATA_DIR
        / f"{table_name}.csv"
    )

    columns = (
        TABLE_COLUMNS[
            table_name
        ]
    )

    column_identifiers = [
        sql.Identifier(
            column
        )
        for column
        in columns
    ]

    copy_query = (
        sql.SQL(
            """
            COPY {} ({})
            FROM STDIN
            WITH (
                FORMAT CSV,
                HEADER TRUE
            );
            """
        )
        .format(
            sql.Identifier(
                table_name
            ),
            sql.SQL(", ").join(
                column_identifiers
            ),
        )
    )

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        cursor.copy_expert(
            copy_query.as_string(
                cursor.connection
            ),
            csv_file,
        )

    cursor.execute(
        sql.SQL(
            "SELECT COUNT(*) FROM {};"
        )
        .format(
            sql.Identifier(
                table_name
            )
        )
    )

    row_count = (
        cursor.fetchone()[0]
    )

    print(
        f"Loaded "
        f"{table_name}: "
        f"{row_count:,} rows"
    )


# ---------------------------------------------------------
# Load all tables
# ---------------------------------------------------------

def load_all_tables() -> None:
    """Load all CSV tables into PostgreSQL."""

    validate_csv_files()

    config = (
        load_database_config()
    )

    connection = None

    try:
        connection = (
            psycopg2.connect(
                **config
            )
        )

        with connection.cursor() as cursor:

            # The employee manager foreign key
            # is deferrable.
            cursor.execute(
                "SET CONSTRAINTS ALL DEFERRED;"
            )

            truncate_tables(
                cursor
            )

            print(
                "\nLoading CSV data "
                "into PostgreSQL...\n"
            )

            for table_name in (
                TABLE_LOAD_ORDER
            ):
                load_table(
                    cursor,
                    table_name,
                )

        connection.commit()

        print(
            "\nAll PostgreSQL tables "
            "loaded successfully."
        )

    except Exception:
        if connection is not None:
            connection.rollback()

            print(
                "\nLoad failed. "
                "Database changes rolled back."
            )

        raise

    finally:
        if connection is not None:
            connection.close()

            print(
                "Database connection closed."
            )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    load_all_tables()