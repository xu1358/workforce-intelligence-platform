"""Execute and validate SQL/Python temporal-dataset equivalence.

The PostgreSQL query is executed in a read-only transaction. Its complete
employee-snapshot result is compared in memory with the committed Python
builder outputs. Only aggregate validation evidence is written to disk.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from build_multi_snapshot_retention_dataset import MODEL_COLUMNS
from v2_data_access import (
    TABLE_SPECS,
    load_database_config,
    load_source_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sql_temporal_equivalence.yaml"
TEMPORAL_CONFIG_PATH = PROJECT_ROOT / "config" / "temporal_snapshots.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def strip_sql_comments(sql_text: str) -> str:
    """Remove block and line comments before checking SQL behavior."""

    without_blocks = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)
    return re.sub(r"--[^\r\n]*", "", without_blocks)


def configured_snapshot_dates(config: dict[str, Any]) -> set[str]:
    """Return every snapshot and prediction-end date in the YAML contract."""

    dates = {str(item["snapshot_date"]) for item in config["historical_snapshots"]}
    dates.update(
        str(item["prediction_end_date"]) for item in config["historical_snapshots"]
    )
    dates.add(str(config["current_scoring"]["as_of_date"]))
    return dates


def sql_contract_issues(
    sql_text: str,
    temporal_config: dict[str, Any],
) -> list[str]:
    """Return static contract problems that would make the SQL unsafe or stale."""

    executable = strip_sql_comments(sql_text)
    normalized = executable.strip()
    issues: list[str] = []
    if not normalized.upper().startswith("WITH"):
        issues.append("query must begin with a read-only WITH statement")

    forbidden = re.findall(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|COPY|GRANT|REVOKE)\b",
        executable,
        flags=re.IGNORECASE,
    )
    if forbidden:
        issues.append(f"write-capable SQL tokens found: {sorted(set(forbidden))}")

    relations = {spec.relation for spec in TABLE_SPECS.values()}
    missing_relations = sorted(
        relation
        for relation in relations
        if not re.search(rf"\b{re.escape(relation)}\b", executable)
    )
    if missing_relations:
        issues.append(f"analytical views not queried: {missing_relations}")

    base_relations = {
        "employees",
        "departments",
        "locations",
        "job_roles",
        "compensation_history",
        "performance_reviews",
        "training_records",
        "employee_events",
    }
    direct_base_reads = sorted(
        relation
        for relation in base_relations
        if re.search(
            rf"\b(?:FROM|JOIN)\s+{re.escape(relation)}\b",
            executable,
            flags=re.IGNORECASE,
        )
    )
    if direct_base_reads:
        issues.append(f"direct base-table reads found: {direct_base_reads}")

    actual_dates = set(re.findall(r"\bDATE\s+'(\d{4}-\d{2}-\d{2})'", executable))
    expected_dates = configured_snapshot_dates(temporal_config)
    if actual_dates != expected_dates:
        issues.append(
            "SQL snapshot dates differ from temporal_snapshots.yaml: "
            f"sql={sorted(actual_dates)}, config={sorted(expected_dates)}"
        )

    semicolon_count = executable.count(";")
    if semicolon_count != 1 or not normalized.endswith(";"):
        issues.append("query must contain exactly one terminated statement")

    return issues


def load_python_outputs(
    project_root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load and combine the two Python-builder temporal outputs."""

    frames = [
        pd.read_csv(project_root / relative_path)
        for relative_path in config["python_outputs"].values()
    ]
    return pd.concat(frames, ignore_index=True)


def execute_read_only_query(
    sql_text: str,
    *,
    project_root: Path,
    connection_factory: Callable[..., Any] | None = None,
) -> tuple[pd.DataFrame, bool, float]:
    """Execute SQL in one read-only PostgreSQL transaction."""

    if connection_factory is None:
        import psycopg2

        connection_factory = psycopg2.connect

    source_config = load_source_config(
        project_root / "config" / "v2_postgresql_source.yaml"
    )
    connection = connection_factory(**load_database_config(project_root, source_config))
    started = time.monotonic()
    try:
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor() as cursor:
            cursor.execute("SHOW transaction_read_only;")
            read_only = str(cursor.fetchone()[0]).lower() in {"on", "true"}
            cursor.execute(sql_text)
            columns = [description.name for description in cursor.description]
            frame = pd.DataFrame(cursor.fetchall(), columns=columns)
        connection.rollback()
    finally:
        connection.close()

    return frame, read_only, time.monotonic() - started


def normalize_date(values: pd.Series, column: str) -> pd.Series:
    """Normalize a date column without hiding invalid non-null values."""

    parsed = pd.to_datetime(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        raise ValueError(f"{column} contains invalid date values.")
    return parsed.dt.strftime("%Y-%m-%d")


def normalize_boolean(values: pd.Series, column: str) -> pd.Series:
    """Normalize nullable boolean representations from CSV and PostgreSQL."""

    mapping = {
        "true": "true",
        "false": "false",
        "1": "true",
        "0": "false",
    }

    def convert(value: Any) -> str | None:
        if pd.isna(value):
            return None
        normalized = mapping.get(str(value).strip().lower())
        if normalized is None:
            raise ValueError(f"{column} contains invalid boolean value {value!r}.")
        return normalized

    return values.map(convert)


def normalize_numeric(values: pd.Series, column: str) -> pd.Series:
    """Convert numeric values without converting invalid text to missing."""

    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"{column} contains invalid numeric values.")
    return numeric.astype("float64")


def compare_column(
    python_values: pd.Series,
    sql_values: pd.Series,
    column: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare one aligned Python/SQL column."""

    date_columns = set(config["date_columns"])
    boolean_columns = set(config["boolean_columns"])
    exact_numeric = set(config["exact_numeric_columns"])
    tolerant_numeric = set(config["tolerant_numeric_columns"])
    absolute_tolerance = float(config["comparison"]["absolute_tolerance"])
    relative_tolerance = float(config["comparison"]["relative_tolerance"])

    null_mismatch = python_values.isna() != sql_values.isna()
    max_absolute_difference: float | None = None

    if column in exact_numeric | tolerant_numeric:
        left = normalize_numeric(python_values, column)
        right = normalize_numeric(sql_values, column)
        both_present = left.notna() & right.notna()
        if column in exact_numeric:
            equal = np.isclose(
                left.fillna(0.0),
                right.fillna(0.0),
                rtol=0.0,
                atol=0.0,
            )
            comparison_type = "exact_numeric"
        else:
            equal = np.isclose(
                left.fillna(0.0),
                right.fillna(0.0),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            comparison_type = "tolerant_numeric"
        mismatch = null_mismatch | (both_present & ~equal)
        if both_present.any():
            max_absolute_difference = float(
                (left[both_present] - right[both_present]).abs().max()
            )
    elif column in date_columns:
        left = normalize_date(python_values, column)
        right = normalize_date(sql_values, column)
        mismatch = null_mismatch | (left.fillna("<NULL>") != right.fillna("<NULL>"))
        comparison_type = "date"
    elif column in boolean_columns:
        left = normalize_boolean(python_values, column)
        right = normalize_boolean(sql_values, column)
        mismatch = null_mismatch | (left.fillna("<NULL>") != right.fillna("<NULL>"))
        comparison_type = "boolean"
    else:
        left = python_values.map(lambda value: None if pd.isna(value) else str(value))
        right = sql_values.map(lambda value: None if pd.isna(value) else str(value))
        mismatch = null_mismatch | (left.fillna("<NULL>") != right.fillna("<NULL>"))
        comparison_type = "exact_text"

    return {
        "feature": column,
        "comparison_type": comparison_type,
        "rows_compared": len(python_values),
        "null_pattern_mismatches": int(null_mismatch.sum()),
        "value_mismatches": int(mismatch.sum()),
        "maximum_absolute_difference": max_absolute_difference,
    }


def align_frames(
    python_frame: pd.DataFrame,
    sql_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Sort both implementations by the committed employee-snapshot key."""

    key = config["row_key"]
    python_keys = python_frame.loc[:, key].sort_values(key).reset_index(drop=True)
    sql_keys = sql_frame.loc[:, key].sort_values(key).reset_index(drop=True)
    keys_match = python_keys.equals(sql_keys)

    python_aligned = python_frame.sort_values(key).reset_index(drop=True)
    sql_aligned = sql_frame.sort_values(key).reset_index(drop=True)
    return python_aligned, sql_aligned, keys_match


def canonical_fingerprint(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    """Fingerprint an aligned frame after contract-aware normalization."""

    exact_numeric = set(config["exact_numeric_columns"])
    tolerant_numeric = set(config["tolerant_numeric_columns"])
    date_columns = set(config["date_columns"])
    boolean_columns = set(config["boolean_columns"])
    decimal_places = int(config["comparison"]["fingerprint_decimal_places"])
    canonical = pd.DataFrame(index=frame.index)

    for column in frame.columns:
        values = frame[column]
        if column in exact_numeric:
            numeric = normalize_numeric(values, column)
            canonical[column] = numeric.map(
                lambda value: "<NULL>" if pd.isna(value) else format(value, ".12g")
            )
        elif column in tolerant_numeric:
            numeric = normalize_numeric(values, column).round(decimal_places)
            canonical[column] = numeric.map(
                lambda value: (
                    "<NULL>" if pd.isna(value) else f"{value:.{decimal_places}f}"
                )
            )
        elif column in date_columns:
            canonical[column] = normalize_date(values, column).fillna("<NULL>")
        elif column in boolean_columns:
            canonical[column] = normalize_boolean(values, column).fillna("<NULL>")
        else:
            canonical[column] = values.map(
                lambda value: "<NULL>" if pd.isna(value) else str(value)
            )

    payload = canonical.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_frames(
    python_frame: pd.DataFrame,
    sql_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare every row and feature from both temporal implementations."""

    python_columns = list(python_frame.columns)
    sql_columns = list(sql_frame.columns)
    columns_match = python_columns == sql_columns == MODEL_COLUMNS
    if set(python_columns) != set(sql_columns):
        missing_from_sql = sorted(set(python_columns) - set(sql_columns))
        extra_in_sql = sorted(set(sql_columns) - set(python_columns))
        raise ValueError(
            "SQL/Python columns differ: "
            f"missing_from_sql={missing_from_sql}, extra_in_sql={extra_in_sql}"
        )
    sql_frame = sql_frame.loc[:, python_columns]

    python_aligned, sql_aligned, keys_match = align_frames(
        python_frame,
        sql_frame,
        config,
    )
    if len(python_aligned) != len(sql_aligned) or not keys_match:
        column_comparison = pd.DataFrame(
            columns=[
                "feature",
                "comparison_type",
                "rows_compared",
                "null_pattern_mismatches",
                "value_mismatches",
                "maximum_absolute_difference",
            ]
        )
    else:
        column_comparison = pd.DataFrame(
            [
                compare_column(
                    python_aligned[column],
                    sql_aligned[column],
                    column,
                    config,
                )
                for column in python_columns
            ]
        )

    dataset_rows = []
    dataset_values = sorted(
        set(python_frame["dataset_type"]) | set(sql_frame["dataset_type"])
    )
    for dataset_type in dataset_values:
        python_rows = int(python_frame["dataset_type"].eq(dataset_type).sum())
        sql_rows = int(sql_frame["dataset_type"].eq(dataset_type).sum())
        dataset_rows.append(
            {
                "dataset_type": dataset_type,
                "python_rows": python_rows,
                "sql_rows": sql_rows,
                "row_count_match": python_rows == sql_rows,
            }
        )
    dataset_summary = pd.DataFrame(dataset_rows)

    comparable = (
        len(python_aligned) == len(sql_aligned)
        and keys_match
        and not column_comparison.empty
    )
    python_fingerprint = (
        canonical_fingerprint(python_aligned, config) if comparable else None
    )
    sql_fingerprint = canonical_fingerprint(sql_aligned, config) if comparable else None
    summary = {
        "columns_match": columns_match,
        "keys_match": keys_match,
        "python_rows": len(python_frame),
        "sql_rows": len(sql_frame),
        "total_null_pattern_mismatches": int(
            column_comparison["null_pattern_mismatches"].sum()
        )
        if not column_comparison.empty
        else None,
        "total_value_mismatches": int(column_comparison["value_mismatches"].sum())
        if not column_comparison.empty
        else None,
        "python_fingerprint": python_fingerprint,
        "sql_fingerprint": sql_fingerprint,
        "fingerprints_match": (
            python_fingerprint is not None and python_fingerprint == sql_fingerprint
        ),
    }
    return column_comparison, dataset_summary, summary


def add_check(
    records: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> None:
    """Append one reviewer-facing validation check."""

    records.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def build_checks(
    *,
    config: dict[str, Any],
    contract_issues: list[str],
    sql_frame: pd.DataFrame,
    column_comparison: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    summary: dict[str, Any],
    read_only: bool,
) -> pd.DataFrame:
    """Build the complete SQL/Python equivalence validation table."""

    checks: list[dict[str, Any]] = []
    expected = config["expected"]
    expected_dataset_rows = expected["dataset_rows"]
    actual_dataset_rows = {
        row["dataset_type"]: int(row["sql_rows"])
        for row in dataset_summary.to_dict(orient="records")
    }
    actual_sequences = sorted(
        int(value) for value in sql_frame["snapshot_sequence"].unique()
    )
    comparison_contract = config["comparison"]
    exact_columns = set(config["exact_numeric_columns"]) | {
        column
        for column in MODEL_COLUMNS
        if column
        not in (
            set(config["exact_numeric_columns"])
            | set(config["tolerant_numeric_columns"])
        )
    }
    exact_mismatches = int(
        column_comparison.loc[
            column_comparison["feature"].isin(exact_columns),
            "value_mismatches",
        ].sum()
    )
    numeric_mismatches = int(
        column_comparison.loc[
            column_comparison["comparison_type"].eq("tolerant_numeric"),
            "value_mismatches",
        ].sum()
    )
    null_mismatches = int(column_comparison["null_pattern_mismatches"].sum())
    governance = config["governance"]
    governance_ok = (
        governance["execute_in_read_only_transaction"]
        and governance["equivalence_validation_only"]
        and not governance["changes_synthetic_records"]
        and not governance["changes_temporal_dataset_values"]
        and not governance["changes_model_results"]
        and not governance["reopens_final_test"]
        and not governance["changes_frozen_policy"]
        and not governance["changes_dashboard_data"]
    )

    add_check(
        checks,
        "Temporal SQL is executable and read-only",
        not contract_issues and read_only,
        {"contract_issues": contract_issues, "transaction_read_only": read_only},
        "0 contract issues; PostgreSQL read-only transaction",
        "The SQL is an executed analytical query, not an unaudited reference.",
    )
    add_check(
        checks,
        "SQL uses every least-privilege Version 2 view",
        not contract_issues and len(TABLE_SPECS) == expected["analytical_views"],
        sorted(spec.relation for spec in TABLE_SPECS.values()),
        f"{expected['analytical_views']} analytical views; 0 base-table reads",
        "Both implementations share the governed PostgreSQL source boundary.",
    )
    add_check(
        checks,
        "SQL output column contract matches Python",
        comparison_contract["require_identical_columns"]
        and bool(summary["columns_match"])
        and len(sql_frame.columns) == expected["columns"],
        list(sql_frame.columns),
        {
            "columns": MODEL_COLUMNS,
            "count": expected["columns"],
        },
        "Column names and order must match the model-input contract.",
    )
    add_check(
        checks,
        "SQL and Python employee-snapshot keys match",
        comparison_contract["require_identical_row_keys"]
        and bool(summary["keys_match"])
        and summary["python_rows"] == expected["total_rows"]
        and summary["sql_rows"] == expected["total_rows"],
        {
            "python_rows": summary["python_rows"],
            "sql_rows": summary["sql_rows"],
        },
        {
            "rows": expected["total_rows"],
            "unique_key": config["row_key"],
        },
        "Neither implementation may add, omit, or duplicate a modeled row.",
    )
    add_check(
        checks,
        "Every dataset population reconciles",
        actual_dataset_rows == expected_dataset_rows
        and bool(dataset_summary["row_count_match"].all())
        and actual_sequences == expected["snapshot_sequences"],
        {
            "dataset_rows": actual_dataset_rows,
            "snapshot_sequences": actual_sequences,
        },
        {
            "dataset_rows": expected_dataset_rows,
            "snapshot_sequences": expected["snapshot_sequences"],
        },
        "Historical and current populations must agree independently.",
    )
    add_check(
        checks,
        "Null patterns agree for every feature",
        comparison_contract["require_identical_null_patterns"] and null_mismatches == 0,
        null_mismatches,
        0,
        "Missingness is part of the feature contract and cannot be hidden.",
    )
    add_check(
        checks,
        "Exact SQL/Python values agree",
        exact_mismatches == 0,
        exact_mismatches,
        0,
        "Identifiers, labels, dates, categories, booleans, and counts are exact.",
    )
    add_check(
        checks,
        "Computed numeric features agree within tolerance",
        numeric_mismatches == 0,
        {
            "mismatches": numeric_mismatches,
            "absolute_tolerance": config["comparison"]["absolute_tolerance"],
            "relative_tolerance": config["comparison"]["relative_tolerance"],
        },
        "0 mismatches within committed tolerance",
        "Independent SQL and Python arithmetic may differ only by floating point.",
    )
    add_check(
        checks,
        "Canonical full-result fingerprints match",
        bool(summary["fingerprints_match"])
        and summary["python_fingerprint"] == expected["canonical_result_sha256"]
        and summary["sql_fingerprint"] == expected["canonical_result_sha256"],
        {
            "python": summary["python_fingerprint"],
            "sql": summary["sql_fingerprint"],
        },
        expected["canonical_result_sha256"],
        "The aggregate digest covers all 24,082 rows and 52 columns.",
    )
    add_check(
        checks,
        "Saved equivalence evidence is aggregate only",
        not config["comparison"]["export_employee_level_differences"],
        config["comparison"]["export_employee_level_differences"],
        False,
        "No employee-level disagreement or modeling row is exported.",
    )
    add_check(
        checks,
        "Frozen analytical governance remains intact",
        governance_ok,
        governance,
        "Read-only equivalence validation; all downstream change flags false",
        "The checkpoint verifies two implementations without reopening results.",
    )
    return pd.DataFrame(checks)


def main() -> None:
    """Run SQL/Python temporal equivalence validation."""

    config = load_yaml(CONFIG_PATH)
    temporal_config = load_yaml(TEMPORAL_CONFIG_PATH)
    sql_path = PROJECT_ROOT / config["sql_query"]
    sql_text = sql_path.read_text(encoding="utf-8")
    contract_issues = sql_contract_issues(sql_text, temporal_config)
    if contract_issues:
        raise ValueError(
            "Temporal SQL contract failed before execution: "
            + "; ".join(contract_issues)
        )

    python_frame = load_python_outputs(PROJECT_ROOT, config)
    sql_frame, read_only, elapsed_seconds = execute_read_only_query(
        sql_text,
        project_root=PROJECT_ROOT,
    )
    column_comparison, dataset_summary, summary = compare_frames(
        python_frame,
        sql_frame,
        config,
    )
    checks = build_checks(
        config=config,
        contract_issues=contract_issues,
        sql_frame=sql_frame,
        column_comparison=column_comparison,
        dataset_summary=dataset_summary,
        summary=summary,
        read_only=read_only,
    )

    execution_summary = pd.DataFrame(
        [
            {
                "query": config["sql_query"],
                "transaction_read_only": read_only,
                "elapsed_seconds": elapsed_seconds,
                "rows_returned": len(sql_frame),
                "columns_returned": len(sql_frame.columns),
                "employee_level_rows_exported": 0,
            }
        ]
    )
    output_directory = PROJECT_ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    checks.to_csv(output_directory / "validation_checks.csv", index=False)
    column_comparison.to_csv(
        output_directory / "column_equivalence.csv",
        index=False,
    )
    dataset_summary.to_csv(
        output_directory / "dataset_summary.csv",
        index=False,
    )
    execution_summary.to_csv(
        output_directory / "query_execution.csv",
        index=False,
    )

    print("\nSQL/PYTHON TEMPORAL DATASET SUMMARY")
    print(dataset_summary.to_string(index=False))
    print(
        "\nColumns compared: "
        f"{len(column_comparison)}; "
        "value mismatches: "
        f"{int(column_comparison['value_mismatches'].sum())}; "
        "null-pattern mismatches: "
        f"{int(column_comparison['null_pattern_mismatches'].sum())}"
    )
    print(f"SQL execution time: {elapsed_seconds:.2f} seconds")
    print("\nSQL TEMPORAL EQUIVALENCE VALIDATION")
    print(checks.to_string(index=False))

    failures = checks.loc[checks["status"].eq("FAIL")]
    if not failures.empty:
        mismatched_features = column_comparison.loc[
            column_comparison["value_mismatches"].gt(0),
            [
                "feature",
                "value_mismatches",
                "maximum_absolute_difference",
            ],
        ].to_dict(orient="records")
        raise ValueError(
            "SQL/Python temporal equivalence validation failed:\n"
            + failures.to_string(index=False)
            + f"\nMismatched features: {mismatched_features}"
        )

    print(f"\nSaved aggregate equivalence evidence to: {output_directory}")
    print("Employee-level comparison rows saved: 0")
    print("\nSQL/PYTHON TEMPORAL EQUIVALENCE VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
