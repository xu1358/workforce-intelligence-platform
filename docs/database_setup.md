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

## Database schema

The PostgreSQL database contains twelve tables:

1. `departments`
2. `locations`
3. `job_roles`
4. `training_programs`
5. `employees`
6. `candidates`
7. `job_requisitions`
8. `applications`
9. `compensation_history`
10. `performance_reviews`
11. `training_records`
12. `employee_events`

The schema is defined in:

`sql/create_tables.sql`

The schema can be created with:

`python src/create_database_schema.py`

The schema can be validated with:

`python src/validate_database_schema.py`

The database schema includes:

- Primary keys
- Foreign keys
- Self-referencing employee-manager relationships
- Unique constraints
- Check constraints
- Indexes for frequently joined foreign-key columns

At the end of Checkpoint 17, all database tables exist but contain no data. CSV data will be loaded in the next checkpoint.

## Data loading

The synthetic CSV files are loaded into PostgreSQL using:

`python src/load_postgresql_data.py`

The loading script:

- Validates required CSV files and column order.
- Removes existing database rows before reloading.
- Loads tables in foreign-key dependency order.
- Uses PostgreSQL `COPY` for efficient bulk loading.
- Runs all loads inside a database transaction.
- Rolls back the transaction if an error occurs.

The loaded data can be validated with:

`python src/validate_database_load.py`

Validation compares the number of rows in every CSV file with the number of rows in its PostgreSQL table.

Additional SQL validation queries are stored in:

`sql/validate_loaded_data.sql`

At the end of Checkpoint 18, all twelve PostgreSQL tables contain the generated synthetic data.