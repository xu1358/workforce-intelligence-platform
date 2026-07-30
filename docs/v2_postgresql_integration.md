# Version 2 PostgreSQL Integration

## Purpose

Checkpoint 62 makes PostgreSQL an executable input to the primary Version 2
retention pipeline. Previously, the database was loaded and validated only
after the temporal modeling dataset had already been built from CSV files.
That sequence demonstrated ingestion but did not make PostgreSQL part of the
Version 2 analytical boundary.

The default pipeline now performs this sequence:

1. generate and validate synthetic source files;
2. create and validate the normalized PostgreSQL schema;
3. load all twelve source tables with PostgreSQL `COPY`;
4. create eight least-privilege Version 2 analytical views;
5. build the temporal retention datasets by querying those views;
6. compare PostgreSQL extracts with the CSV fallback;
7. confirm the PostgreSQL-built temporal files retain their frozen hashes; and
8. continue to model development, policy analysis, and current planning.

## Analytical Views

The SQL contract is:

```text
sql/create_v2_analytics_views.sql
```

| View | Grain | Version 2 purpose |
| --- | --- | --- |
| `analytics_v2_employees` | One row per employee | Snapshot eligibility, outcome windows, and baseline dimensions |
| `analytics_v2_departments` | One row per department | Department name and group |
| `analytics_v2_locations` | One row per location | Point-in-time city, region, and location type |
| `analytics_v2_job_roles` | One row per role | Job title, family, and level |
| `analytics_v2_compensation_history` | One row per compensation event | Point-in-time salary and growth features |
| `analytics_v2_performance_reviews` | One row per review | Latest and historical performance features |
| `analytics_v2_training_records` | One row per training record | Trailing-year training measures |
| `analytics_v2_employee_events` | One row per employee event | Transfers, promotions, manager changes, and leave history |

The views exclude direct names, manager identifiers, reviewer identifiers, and
other fields not required by the Version 2 builder. The database extraction is
therefore an explicit, least-privilege analytical projection rather than
`SELECT *`.

## Python Data-Access Boundary

`src/v2_data_access.py` owns the two supported backends:

- `postgresql` queries the eight committed analytical views with
  `psycopg2`;
- `csv` reads the same declared columns from `data/raw/`.

Both paths normalize dates and PostgreSQL `NUMERIC` values into the same
in-memory contract. The temporal builder accepts an explicit source:

```bash
python src/build_multi_snapshot_retention_dataset.py --source postgresql
python src/build_multi_snapshot_retention_dataset.py --source csv
```

Running the builder without an argument retains the CSV default for isolated
notebook and helper use. The complete project runner explicitly selects
PostgreSQL whenever database stages are enabled.

## Pipeline Behavior

The default command uses PostgreSQL:

```bash
python scripts/run_project.py pipeline
```

The focused database command loads the database, builds Version 2 temporal
data from PostgreSQL, and runs parity validation:

```bash
python scripts/run_project.py postgres
```

A reviewer without a local PostgreSQL instance can deliberately select the
file-backed fallback:

```bash
python scripts/run_project.py pipeline --skip-postgres
```

The fallback is documented as an infrastructure convenience. It is no longer
the default primary pipeline.

## Parity Validation

`src/validate_v2_postgresql_integration.py` checks:

- all eight source relations are queried;
- CSV and PostgreSQL row counts match;
- selected columns and column order match;
- declared primary keys remain unique;
- canonical content hashes match across backends;
- query provenance identifies PostgreSQL;
- the resulting historical and current temporal datasets retain their
  committed SHA-256 fingerprints; and
- PostgreSQL preparation and parity validation occur before downstream model
  analysis.

The frozen expected temporal outputs are:

| Output | Rows | Columns | Canonical content SHA-256 |
| --- | ---: | ---: | --- |
| `retention_multi_snapshot.csv` | 16,673 | 52 | `6db69525669a82b72ef64439b7744abe5dfc0a2ad3e54b4d311db8c01a141358` |
| `current_active_scoring_population.csv` | 7,409 | 52 | `fce85e6e870f99156374d12d58b2a64006ce30ccc7bb2f472d6990ce289beef4` |

These hashes make the integration test stronger than a row-count-only check:
the database path must reproduce every modeling value. Before hashing, the
validator canonicalizes equivalent numeric representations such as `100000`
and `100000.0`, missing values, and Windows `CRLF` versus Unix `LF` line
endings. This keeps the evidence portable without ignoring any real value,
column, row, or ordering difference.

Aggregate evidence is saved under:

```text
data/processed/v2_postgresql_integration/
```

The audit files contain table names, counts, hashes, and validation statuses.
They do not export employee-level database rows.

## Data-Analyst Evidence

This integration demonstrates:

- normalized relational modeling with keys and constraints;
- bulk ingestion through PostgreSQL `COPY`;
- explicit SQL analytical views;
- environment-based credential management;
- Python-to-PostgreSQL querying with `psycopg2`;
- source-system reconciliation through row and content hashes;
- point-in-time feature construction after database extraction; and
- a tested fallback strategy for reviewers without local infrastructure.

The database is therefore part of the primary analytical workflow, not a
legacy-only exhibit.

Checkpoint 63 extends this source-level evidence by executing
`sql/retention_modeling_multi_snapshot.sql` and comparing its complete
24,082-row, 52-column result with the Python builder. See
[SQL and Python temporal equivalence](sql_python_temporal_equivalence.md).

## Scope and Governance

Checkpoint 62 changes the source boundary and pipeline order. It does not:

- regenerate or alter synthetic source records;
- change temporal feature definitions;
- change historical or current temporal dataset values;
- retrain or select a different model;
- reopen the once-only final test;
- modify the frozen retention policy; or
- modify dashboard data or employee selections.

The PostgreSQL-built output hashes must match the previously approved
file-built hashes before downstream modeling is allowed to continue.

## Reproduction

Create `.env` from `.env.example`, ensure PostgreSQL is running, and run:

```powershell
.\scripts\checkpoints\run_checkpoint62.ps1
```

The checkpoint executes the focused PostgreSQL path followed by the complete
cross-platform quality suite.
