from pathlib import Path
import os

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


# ---------------------------------------------------------
# Expected database structure
# ---------------------------------------------------------

EXPECTED_COLUMNS = {
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
}


EXPECTED_PRIMARY_KEYS = {
    "departments": "department_id",
    "locations": "location_id",
    "job_roles": "job_role_id",
    "training_programs": "program_id",
    "employees": "employee_id",
    "candidates": "candidate_id",
    "job_requisitions": "requisition_id",
    "applications": "application_id",
    "compensation_history": (
        "compensation_id"
    ),
    "performance_reviews": "review_id",
    "training_records": (
        "training_record_id"
    ),
    "employee_events": "event_id",
}


EXPECTED_FOREIGN_KEYS = {
    (
        "employees",
        "department_id",
        "departments",
        "department_id",
    ),
    (
        "employees",
        "location_id",
        "locations",
        "location_id",
    ),
    (
        "employees",
        "job_role_id",
        "job_roles",
        "job_role_id",
    ),
    (
        "employees",
        "manager_id",
        "employees",
        "employee_id",
    ),
    (
        "job_requisitions",
        "job_role_id",
        "job_roles",
        "job_role_id",
    ),
    (
        "job_requisitions",
        "department_id",
        "departments",
        "department_id",
    ),
    (
        "job_requisitions",
        "location_id",
        "locations",
        "location_id",
    ),
    (
        "job_requisitions",
        "recruiter_id",
        "employees",
        "employee_id",
    ),
    (
        "applications",
        "candidate_id",
        "candidates",
        "candidate_id",
    ),
    (
        "applications",
        "requisition_id",
        "job_requisitions",
        "requisition_id",
    ),
    (
        "applications",
        "employee_id",
        "employees",
        "employee_id",
    ),
    (
        "compensation_history",
        "employee_id",
        "employees",
        "employee_id",
    ),
    (
        "performance_reviews",
        "employee_id",
        "employees",
        "employee_id",
    ),
    (
        "performance_reviews",
        "reviewer_id",
        "employees",
        "employee_id",
    ),
    (
        "training_records",
        "employee_id",
        "employees",
        "employee_id",
    ),
    (
        "training_records",
        "program_id",
        "training_programs",
        "program_id",
    ),
    (
        "employee_events",
        "employee_id",
        "employees",
        "employee_id",
    ),
}


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

def load_database_config() -> dict:
    """Load PostgreSQL configuration."""

    load_dotenv(
        ENV_PATH
    )

    return {
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


# ---------------------------------------------------------
# Validate tables and columns
# ---------------------------------------------------------

def validate_tables_and_columns(
    cursor,
) -> None:
    """Validate table names and column order."""

    cursor.execute(
        """
        SELECT
            table_name,
            column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY
            table_name,
            ordinal_position;
        """
    )

    actual_columns = {}

    for table_name, column_name in (
        cursor.fetchall()
    ):
        actual_columns.setdefault(
            table_name,
            []
        ).append(
            column_name
        )

    expected_tables = set(
        EXPECTED_COLUMNS
    )

    actual_tables = (
        set(actual_columns)
        & expected_tables
    )

    if actual_tables != expected_tables:
        missing_tables = (
            expected_tables
            - actual_tables
        )

        raise ValueError(
            "Missing database tables: "
            f"{sorted(missing_tables)}"
        )

    for (
        table_name,
        expected_columns,
    ) in EXPECTED_COLUMNS.items():

        actual_table_columns = (
            actual_columns[
                table_name
            ]
        )

        if (
            actual_table_columns
            != expected_columns
        ):
            raise ValueError(
                f"{table_name}: unexpected "
                "columns or column order.\n"
                f"Expected: {expected_columns}\n"
                f"Actual: "
                f"{actual_table_columns}"
            )

    print(
        "Tables and columns validated: "
        f"{len(EXPECTED_COLUMNS)} tables"
    )


# ---------------------------------------------------------
# Validate primary keys
# ---------------------------------------------------------

def validate_primary_keys(
    cursor,
) -> None:
    """Validate primary-key definitions."""

    cursor.execute(
        """
        SELECT
            tc.table_name,
            kcu.column_name
        FROM information_schema.table_constraints
            AS tc
        JOIN information_schema.key_column_usage
            AS kcu
            ON tc.constraint_name
                = kcu.constraint_name
            AND tc.table_schema
                = kcu.table_schema
        WHERE
            tc.table_schema = 'public'
            AND tc.constraint_type
                = 'PRIMARY KEY';
        """
    )

    actual_primary_keys = {
        table_name: column_name
        for table_name, column_name
        in cursor.fetchall()
    }

    for (
        table_name,
        primary_key,
    ) in EXPECTED_PRIMARY_KEYS.items():

        if (
            actual_primary_keys.get(
                table_name
            )
            != primary_key
        ):
            raise ValueError(
                f"{table_name}: incorrect "
                "primary key."
            )

    print(
        "Primary keys validated: "
        f"{len(EXPECTED_PRIMARY_KEYS)}"
    )


# ---------------------------------------------------------
# Validate foreign keys
# ---------------------------------------------------------

def validate_foreign_keys(
    cursor,
) -> None:
    """Validate foreign-key relationships."""

    cursor.execute(
        """
        SELECT
            tc.table_name,
            kcu.column_name,
            ccu.table_name,
            ccu.column_name
        FROM information_schema.table_constraints
            AS tc
        JOIN information_schema.key_column_usage
            AS kcu
            ON tc.constraint_name
                = kcu.constraint_name
            AND tc.table_schema
                = kcu.table_schema
        JOIN information_schema.constraint_column_usage
            AS ccu
            ON ccu.constraint_name
                = tc.constraint_name
            AND ccu.table_schema
                = tc.table_schema
        WHERE
            tc.table_schema = 'public'
            AND tc.constraint_type
                = 'FOREIGN KEY';
        """
    )

    actual_foreign_keys = set(
        cursor.fetchall()
    )

    missing_foreign_keys = (
        EXPECTED_FOREIGN_KEYS
        - actual_foreign_keys
    )

    if missing_foreign_keys:
        raise ValueError(
            "Missing foreign keys: "
            f"{sorted(missing_foreign_keys)}"
        )

    print(
        "Foreign keys validated: "
        f"{len(EXPECTED_FOREIGN_KEYS)}"
    )


# ---------------------------------------------------------
# Display row counts
# ---------------------------------------------------------

def print_row_counts(
    cursor,
) -> None:
    """Print the current number of rows."""

    print(
        "\nCurrent table row counts:"
    )

    for table_name in (
        EXPECTED_COLUMNS
    ):
        cursor.execute(
            f'SELECT COUNT(*) '
            f'FROM "{table_name}";'
        )

        row_count = (
            cursor.fetchone()[0]
        )

        print(
            f"{table_name}: "
            f"{row_count}"
        )


# ---------------------------------------------------------
# Main validation
# ---------------------------------------------------------

def main() -> None:
    """Validate the PostgreSQL schema."""

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

            validate_tables_and_columns(
                cursor
            )

            validate_primary_keys(
                cursor
            )

            validate_foreign_keys(
                cursor
            )

            print_row_counts(
                cursor
            )

        print(
            "\nDatabase schema validation "
            "successful."
        )

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()