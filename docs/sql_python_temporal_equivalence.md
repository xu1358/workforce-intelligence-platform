# SQL and Python Temporal Equivalence

## Purpose

Checkpoint 63 turns `sql/retention_modeling_multi_snapshot.sql` from a
reference artifact into an executed and tested implementation.

The project now answers a specific reproducibility question:

> Does the PostgreSQL SQL implementation produce the same 24,082
> employee-snapshot rows and 52 analytical columns as the Python temporal
> builder?

This is an implementation-equivalence test. It does not compare two different
models, and it does not create new analytical results.

## Why this check is necessary

Before Checkpoint 63, the repository stated that the large SQL query mirrored
the Python logic. The SQL was not called by the pipeline, and no automated
test compared its result with the Python output.

That created three risks:

1. The SQL could become stale while Python changed.
2. A reviewer could not reproduce the claimed agreement.
3. Differences in dates, nulls, aggregation, or row eligibility could remain
   hidden behind similar aggregate counts.

Checkpoint 63 replaces that unverified statement with executable evidence.

## Discrepancy found and corrected

The first complete execution showed that the implementations were not
equivalent. For historical employees with no termination date, PostgreSQL's
three-valued boolean logic produced a `NULL` target:

```sql
e.termination_date > s.snapshot_date
AND e.termination_date <= s.prediction_end_date
```

Python correctly treated those employees as non-events with target `0`. The
SQL now makes that rule explicit:

```sql
COALESCE(
    e.termination_date > s.snapshot_date
    AND e.termination_date <= s.prediction_end_date,
    FALSE
)::INTEGER
```

The original query therefore had 13,512 incorrect historical target nulls.
After correction, target null patterns and values agree exactly. Current
scoring targets remain intentionally unknown.

## Implementations under comparison

### Python implementation

`src/build_multi_snapshot_retention_dataset.py` constructs:

- 16,673 historical employee-snapshot rows;
- 7,409 current active-scoring rows; and
- 52 columns in each output.

It uses pandas after retrieving the eight governed Version 2 source relations.

### PostgreSQL implementation

`sql/retention_modeling_multi_snapshot.sql` independently expresses the same:

- employee eligibility at each snapshot;
- twelve-month historical target;
- historical location reconstruction;
- point-in-time compensation;
- point-in-time performance;
- trailing-twelve-month training;
- prior and trailing event counts;
- promotion recency;
- salary growth; and
- peer-group salary position.

The SQL now queries all eight `analytics_v2_*` least-privilege views. It no
longer reads unrestricted base tables.

## Execution path

The canonical PostgreSQL command now runs these stages in order:

1. Test the PostgreSQL connection.
2. Create and validate the 12-table schema.
3. Load and reconcile the synthetic CSV source.
4. Create the eight Version 2 analytical views.
5. Build the Python temporal outputs from PostgreSQL.
6. Validate PostgreSQL/CSV source parity.
7. Execute the independent multi-snapshot SQL in a read-only transaction.
8. Compare every SQL result with the Python output.

The SQL equivalence step runs before feature diagnostics, model splitting, or
modeling.

## Comparison contract

The validator compares both complete implementations after sorting by:

```text
snapshot_sequence, employee_id
```

It requires:

- exactly 24,082 rows;
- exactly 52 columns in the committed order;
- identical employee-snapshot keys;
- 16,673 historical rows;
- 7,409 current-scoring rows;
- identical null locations;
- exact identifiers, categories, dates, booleans, labels, and counts;
- computed numeric values equal within absolute and relative tolerance
  `1e-9`; and
- identical full-result SHA-256 fingerprints after deterministic
  eight-decimal numeric normalization.

The complete canonical result fingerprint is:

```text
0ed2538a67d6a7050d33bbdc936f5ef39264565f7c819b40d291032d7238431c
```

The verified comparison contains zero row-key, null-pattern, and value
mismatches. The largest observed SQL/Python arithmetic difference before
tolerance comparison is approximately `4.27e-14`, far below the committed
`1e-9` limit.

The numeric tolerance handles insignificant floating-point differences between
PostgreSQL arithmetic and NumPy/pandas arithmetic. It does not permit material
feature differences.

## Deterministic SQL behavior

The SQL includes stable identifiers when resolving multiple records on the
same date:

- `compensation_id` for compensation records;
- `review_id` for performance reviews; and
- `event_id` for hire, transfer, and event records.

Promotion recency uses the same calendar-month calculation as Python, and
approximate age uses the same snapshot-year minus birth-year rule.

The validator also checks that the snapshot date literals in SQL match
`config/temporal_snapshots.yaml`. A date change in one implementation without
the other therefore fails before query execution.

## Read-only and privacy controls

The SQL runs inside a PostgreSQL read-only transaction. Static validation
rejects write-capable statements such as:

- `INSERT`
- `UPDATE`
- `DELETE`
- `CREATE`
- `DROP`
- `ALTER`
- `TRUNCATE`
- `COPY`

The full employee-level comparison exists temporarily in memory. Saved
evidence is aggregate:

- `validation_checks.csv`
- `column_equivalence.csv`
- `dataset_summary.csv`
- `query_execution.csv`

No employee identifiers or row-level disagreements are exported.

## Validation command

The focused validator can be run after the PostgreSQL data and Python temporal
outputs exist:

```bash
python src/validate_sql_temporal_equivalence.py
```

The complete database path is:

```bash
python scripts/run_project.py postgres
```

On Windows, the historical checkpoint runner is:

```powershell
.\scripts\checkpoints\run_checkpoint63.ps1
```

Successful validation reports:

```text
SQL/PYTHON TEMPORAL EQUIVALENCE VALIDATED SUCCESSFULLY
```

## Clean CI and live integration responsibilities

The ordinary `pytest` suite must run from a clean Git checkout. It therefore
does not read `data/processed/retention_multi_snapshot.csv` or
`data/processed/current_active_scoring_population.csv`, because these are
generated runtime artifacts and are intentionally not committed.

The automated unit tests verify the committed population, schema, comparison,
fingerprint, read-only, and privacy contracts using repository-owned fixtures.
The live PostgreSQL command performs the full 24,082-row comparison:

```bash
python scripts/run_project.py postgres
```

This separation prevents GitHub Actions from depending on one developer's
local outputs while preserving the complete SQL/Python integration gate. The
live validator still fails unless both generated datasets exist, PostgreSQL is
available, and all 52 columns agree across both implementations.

## What a failure means

A failed check means the two implementations no longer construct the same
modeling dataset. The output identifies affected feature names and aggregate
mismatch counts without writing employee-level data.

A failure must be investigated before downstream model steps run. It must not
be resolved by changing the model, reopening the final test, or silently
loosening the comparison tolerance.

## Analytical scope

Checkpoint 63 changes:

- SQL execution;
- equivalence validation;
- deterministic SQL ordering;
- pipeline orchestration;
- automated tests; and
- documentation.

It does not change:

- synthetic records;
- temporal output values;
- selected features;
- model selection;
- probability calibration;
- final-test evidence;
- intervention policy;
- dashboard data; or
- employee review selections.

The achievement is verified implementation equivalence, not improved model
performance.
