from pathlib import Path
import os

import psycopg2
from dotenv import load_dotenv


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = PROJECT_ROOT / ".env"


# ---------------------------------------------------------
# Load database configuration
# ---------------------------------------------------------

def load_database_config() -> dict:
    """Load PostgreSQL settings from the .env file."""

    if not ENV_PATH.exists():
        raise FileNotFoundError(
            "Missing .env file in the project root."
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
        for key, value in config.items()
        if not value
    ]

    if missing_settings:
        raise ValueError(
            "Missing database settings: "
            f"{missing_settings}"
        )

    return config


# ---------------------------------------------------------
# Test PostgreSQL connection
# ---------------------------------------------------------

def test_connection() -> None:
    """Connect to PostgreSQL and print database information."""

    config = (
        load_database_config()
    )

    connection = None

    try:
        connection = psycopg2.connect(
            **config
        )

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), current_user;"
            )

            database_name, user_name = (
                cursor.fetchone()
            )

        print(
            "PostgreSQL connection successful."
        )

        print(
            f"Database: {database_name}"
        )

        print(
            f"User: {user_name}"
        )

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
    test_connection()