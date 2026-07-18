# PostgreSQL Database Setup

## Database

The Workforce Intelligence and Retention Decision Platform uses PostgreSQL as its relational database.

Database name:

`workforce_intelligence`

## Local connection

The local development environment uses the following configuration:

- Host: localhost
- Port: 5432
- Database: workforce_intelligence
- User: postgres

The database password is stored in a local `.env` file and is not committed to Git.

## Environment variables

The required environment variables are:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

An example configuration is provided in `.env.example`.

## Python connection

Python connects to PostgreSQL using `psycopg2`.

The connection can be tested with:

`python src/test_database_connection.py`

A successful test confirms that the application can connect to the `workforce_intelligence` database.

## Current database state

At the end of Checkpoint 16:

- PostgreSQL is running locally.
- The `workforce_intelligence` database exists.
- Python can connect successfully.
- Database credentials are stored securely in `.env`.
- The database tables have not yet been created or loaded.