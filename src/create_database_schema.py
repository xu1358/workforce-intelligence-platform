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

SQL_PATH = (
    PROJECT_ROOT
    / "sql"
    / "create_tables.sql"
)


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

def load_database_config() -> dict:
    """Load database configuration from .env."""

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
# Read SQL
# ---------------------------------------------------------

def load_schema_sql() -> str:
    """Read the SQL schema file."""

    if not SQL_PATH.exists():
        raise FileNotFoundError(
            "Missing sql/create_tables.sql."
        )

    return SQL_PATH.read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------
# Create schema
# ---------------------------------------------------------

def create_schema() -> None:
    """Create PostgreSQL tables and indexes."""

    config = (
        load_database_config()
    )

    schema_sql = (
        load_schema_sql()
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
                schema_sql
            )

        connection.commit()

        print(
            "Database schema created "
            "successfully."
        )

    except Exception:
        if connection is not None:
            connection.rollback()

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
    create_schema()