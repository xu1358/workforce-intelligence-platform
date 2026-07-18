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


TABLE_NAMES = [
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
# Count CSV rows
# ---------------------------------------------------------

def count_csv_rows(
    table_name: str,
) -> int:
    """Count data rows in one CSV file."""

    csv_path = (
        RAW_DATA_DIR
        / f"{table_name}.csv"
    )

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.reader(
            csv_file
        )

        next(
            reader
        )

        return sum(
            1
            for _ in reader
        )


# ---------------------------------------------------------
# Validate database row counts
# ---------------------------------------------------------

def validate_row_counts(
    cursor,
) -> None:
    """Compare CSV row counts with PostgreSQL."""

    print(
        "CSV vs PostgreSQL row counts:\n"
    )

    for table_name in (
        TABLE_NAMES
    ):

        csv_count = (
            count_csv_rows(
                table_name
            )
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

        database_count = (
            cursor.fetchone()[0]
        )

        matches = (
            csv_count
            == database_count
        )

        print(
            f"{table_name}: "
            f"CSV={csv_count:,}, "
            f"PostgreSQL="
            f"{database_count:,}, "
            f"match={matches}"
        )

        if not matches:
            raise ValueError(
                f"{table_name}: "
                "row count mismatch."
            )


# ---------------------------------------------------------
# Workforce checks
# ---------------------------------------------------------

def validate_workforce(
    cursor,
) -> None:
    """Validate key employee counts."""

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM employees;
        """
    )

    employee_count = (
        cursor.fetchone()[0]
    )

    if employee_count != 10_000:
        raise ValueError(
            "Expected exactly "
            "10,000 employees."
        )

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM employees
        WHERE manager_id IS NULL;
        """
    )

    employees_without_manager = (
        cursor.fetchone()[0]
    )

    print(
        "\nWorkforce checks:"
    )

    print(
        "Employees:",
        employee_count,
    )

    print(
        "Employees without manager:",
        employees_without_manager,
    )


# ---------------------------------------------------------
# Recruiting checks
# ---------------------------------------------------------

def validate_recruiting(
    cursor,
) -> None:
    """Validate key recruiting relationships."""

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM candidates;
        """
    )

    candidate_count = (
        cursor.fetchone()[0]
    )

    if candidate_count != 40_000:
        raise ValueError(
            "Expected exactly "
            "40,000 candidates."
        )

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM applications
        WHERE application_status = 'Hired';
        """
    )

    hired_count = (
        cursor.fetchone()[0]
    )

    if hired_count != 10_000:
        raise ValueError(
            "Expected exactly "
            "10,000 hired applications."
        )

    cursor.execute(
        """
        SELECT COUNT(
            DISTINCT employee_id
        )
        FROM applications
        WHERE application_status = 'Hired';
        """
    )

    hired_employee_count = (
        cursor.fetchone()[0]
    )

    if hired_employee_count != 10_000:
        raise ValueError(
            "Expected exactly "
            "10,000 employees connected "
            "to hired applications."
        )

    cursor.execute(
        """
        SELECT SUM(target_headcount)
        FROM job_requisitions
        WHERE requisition_status = 'Filled';
        """
    )

    filled_headcount = (
        cursor.fetchone()[0]
    )

    if filled_headcount != 10_000:
        raise ValueError(
            "Filled requisition "
            "headcount must equal 10,000."
        )

    print(
        "\nRecruiting checks:"
    )

    print(
        "Candidates:",
        candidate_count,
    )

    print(
        "Hired applications:",
        hired_count,
    )

    print(
        "Employees linked to hires:",
        hired_employee_count,
    )

    print(
        "Filled target headcount:",
        filled_headcount,
    )


# ---------------------------------------------------------
# Data overview
# ---------------------------------------------------------

def print_database_summary(
    cursor,
) -> None:
    """Display useful database statistics."""

    print(
        "\nApplication status counts:"
    )

    cursor.execute(
        """
        SELECT
            application_status,
            COUNT(*)
        FROM applications
        GROUP BY application_status
        ORDER BY COUNT(*) DESC;
        """
    )

    for (
        status,
        count,
    ) in cursor.fetchall():

        print(
            f"{status}: "
            f"{count:,}"
        )

    print(
        "\nEmployee status counts:"
    )

    cursor.execute(
        """
        SELECT
            employment_status,
            COUNT(*)
        FROM employees
        GROUP BY employment_status
        ORDER BY employment_status;
        """
    )

    for (
        status,
        count,
    ) in cursor.fetchall():

        print(
            f"{status}: "
            f"{count:,}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Validate loaded PostgreSQL data."""

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

            validate_row_counts(
                cursor
            )

            validate_workforce(
                cursor
            )

            validate_recruiting(
                cursor
            )

            print_database_summary(
                cursor
            )

        print(
            "\nAll database load "
            "validation checks passed."
        )

    finally:
        if connection is not None:
            connection.close()

            print(
                "Database connection closed."
            )


if __name__ == "__main__":
    main()