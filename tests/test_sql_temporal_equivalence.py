"""Contracts for executable SQL/Python temporal equivalence."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import yaml

from build_multi_snapshot_retention_dataset import MODEL_COLUMNS
from validate_sql_temporal_equivalence import (
    compare_frames,
    execute_read_only_query,
    sql_contract_issues,
)
from v2_data_access import TABLE_SPECS


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one test contract."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def one_row_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Build one complete model-contract row for comparison tests."""

    exact_numeric = set(config["exact_numeric_columns"])
    tolerant_numeric = set(config["tolerant_numeric_columns"])
    date_columns = set(config["date_columns"])
    boolean_columns = set(config["boolean_columns"])
    values: dict[str, Any] = {}

    for column in MODEL_COLUMNS:
        if column in exact_numeric:
            values[column] = 1
        elif column in tolerant_numeric:
            values[column] = 1.5
        elif column in date_columns:
            values[column] = "2023-06-30"
        elif column in boolean_columns:
            values[column] = True
        else:
            values[column] = "value"

    values.update(
        {
            "employee_id": 100001,
            "dataset_type": "historical",
            "snapshot_sequence": 1,
            "snapshot_date": "2023-06-30",
            "prediction_end_date": "2024-06-30",
            "attrition_next_12m": 0,
        }
    )
    return pd.DataFrame([values], columns=MODEL_COLUMNS)


def test_sql_is_read_only_and_queries_all_analytical_views(
    project_root: Path,
) -> None:
    """The large SQL artifact must be executable through the governed views."""

    temporal_config = load_yaml(project_root / "config" / "temporal_snapshots.yaml")
    sql_text = (
        project_root / "sql" / "retention_modeling_multi_snapshot.sql"
    ).read_text(encoding="utf-8")

    assert sql_contract_issues(sql_text, temporal_config) == []
    assert all(spec.relation in sql_text for spec in TABLE_SPECS.values())


def test_equivalent_sql_and_python_rows_pass_with_numeric_tolerance(
    project_root: Path,
) -> None:
    """Equivalent database types and insignificant arithmetic noise must pass."""

    config = load_yaml(project_root / "config" / "sql_temporal_equivalence.yaml")
    python_frame = one_row_frame(config)
    sql_frame = python_frame.copy()
    sql_frame["snapshot_date"] = [date(2023, 6, 30)]
    sql_frame["prediction_end_date"] = [date(2024, 6, 30)]
    sql_frame["base_salary"] = [Decimal("1.5000000001")]

    comparison, datasets, summary = compare_frames(
        python_frame,
        sql_frame,
        config,
    )

    assert summary["columns_match"]
    assert summary["keys_match"]
    assert summary["total_value_mismatches"] == 0
    assert summary["total_null_pattern_mismatches"] == 0
    assert summary["fingerprints_match"]
    assert datasets["row_count_match"].all()
    assert comparison["value_mismatches"].sum() == 0


def test_exact_feature_difference_fails_equivalence(
    project_root: Path,
) -> None:
    """A changed target or exact feature must not be hidden by normalization."""

    config = load_yaml(project_root / "config" / "sql_temporal_equivalence.yaml")
    python_frame = one_row_frame(config)
    sql_frame = python_frame.copy()
    sql_frame.loc[0, "attrition_next_12m"] = 1

    comparison, _, summary = compare_frames(
        python_frame,
        sql_frame,
        config,
    )
    target = comparison.loc[comparison["feature"].eq("attrition_next_12m")].iloc[0]

    assert target["value_mismatches"] == 1
    assert summary["total_value_mismatches"] == 1
    assert not summary["fingerprints_match"]


def test_committed_population_contract_is_complete_and_portable(
    project_root: Path,
) -> None:
    """The committed contract must be testable without generated local files."""

    config = load_yaml(project_root / "config" / "sql_temporal_equivalence.yaml")
    expected = config["expected"]
    output_paths = {name: Path(path) for name, path in config["python_outputs"].items()}
    fingerprint = str(expected["canonical_result_sha256"])

    assert expected["dataset_rows"] == {
        "historical": 16673,
        "current_scoring": 7409,
    }
    assert expected["total_rows"] == sum(expected["dataset_rows"].values())
    assert expected["columns"] == len(MODEL_COLUMNS)
    assert expected["snapshot_sequences"] == [1, 2, 3, 4]
    assert config["row_key"] == ["snapshot_sequence", "employee_id"]
    assert set(output_paths) == {"historical", "current_scoring"}
    assert all(
        path.parts[:2] == ("data", "processed") and path.suffix == ".csv"
        for path in output_paths.values()
    )
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")


class FakeCursor:
    """Minimal cursor that records the read-only contract."""

    def __init__(self) -> None:
        self.description: list[SimpleNamespace] = []
        self._result: list[tuple[Any, ...]] = []
        self.commands: list[str] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, command: str) -> None:
        self.commands.append(command)
        if command.startswith("SHOW"):
            self._result = [("on",)]
            self.description = [SimpleNamespace(name="transaction_read_only")]
        else:
            self._result = [(1,)]
            self.description = [SimpleNamespace(name="value")]

    def fetchone(self) -> tuple[Any, ...]:
        return self._result[0]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._result


class FakeConnection:
    """Minimal connection proving transaction settings and cleanup."""

    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.readonly: bool | None = None
        self.autocommit: bool | None = None
        self.rolled_back = False
        self.closed = False

    def set_session(self, *, readonly: bool, autocommit: bool) -> None:
        self.readonly = readonly
        self.autocommit = autocommit

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def test_sql_execution_enforces_read_only_transaction(
    project_root: Path,
    monkeypatch: Any,
) -> None:
    """Runtime execution must be read-only and close its database connection."""

    connection = FakeConnection()
    monkeypatch.setattr(
        "validate_sql_temporal_equivalence.load_database_config",
        lambda *_: {},
    )

    frame, read_only, _ = execute_read_only_query(
        "WITH example AS (SELECT 1) SELECT 1;",
        project_root=project_root,
        connection_factory=lambda **_: connection,
    )

    assert read_only
    assert connection.readonly is True
    assert connection.autocommit is False
    assert connection.rolled_back
    assert connection.closed
    assert frame.to_dict(orient="records") == [{"value": 1}]


def test_equivalence_outputs_are_aggregate_only(project_root: Path) -> None:
    """Configuration must prohibit employee-level difference exports."""

    config = load_yaml(project_root / "config" / "sql_temporal_equivalence.yaml")

    assert config["comparison"]["export_employee_level_differences"] is False
    assert config["governance"]["equivalence_validation_only"] is True
