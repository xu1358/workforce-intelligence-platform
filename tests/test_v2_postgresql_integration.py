"""Contracts for the PostgreSQL-backed Version 2 analytical path."""

from __future__ import annotations

from argparse import Namespace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_project import (
    POSTGRES_STEPS,
    V2_CSV_DATASET_STEP,
    V2_POSTGRES_DATASET_STEP,
    V2_POSTGRES_VALIDATION_STEP,
    VERSION_2_ANALYSIS_STEPS,
    build_pipeline_steps,
)
from v2_data_access import (
    TABLE_SPECS,
    frame_fingerprint,
    normalize_frame,
)
from validate_v2_postgresql_integration import canonical_csv_sha256


def load_contract(project_root: Path) -> dict:
    """Load the Version 2 PostgreSQL source contract."""

    with (project_root / "config" / "v2_postgresql_source.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def pipeline_args(*, skip_postgres: bool) -> Namespace:
    """Build the portable runner arguments used by integration tests."""

    return Namespace(
        skip_quality=True,
        skip_data_generation=True,
        skip_postgres=skip_postgres,
        include_legacy_v1=False,
        skip_external_benchmark=True,
    )


def test_source_contract_maps_all_eight_analytical_views(
    project_root: Path,
) -> None:
    """Every temporal source must have one explicit database view."""

    contract = load_contract(project_root)

    assert len(TABLE_SPECS) == 8
    assert set(TABLE_SPECS) == set(contract["source_tables"])
    for filename, spec in TABLE_SPECS.items():
        configured = contract["source_tables"][filename]
        assert configured["relation"] == spec.relation
        assert configured["primary_key"] == spec.primary_key


def test_employee_view_excludes_direct_names_and_unused_ids() -> None:
    """The analytical employee projection should follow least privilege."""

    columns = set(TABLE_SPECS["employees.csv"].columns)

    assert {
        "employee_id",
        "hire_date",
        "termination_date",
        "department_id",
        "location_id",
        "job_role_id",
    }.issubset(columns)
    assert not {
        "first_name",
        "last_name",
        "manager_id",
        "termination_type",
    }.intersection(columns)


def test_sql_file_creates_every_declared_view(project_root: Path) -> None:
    """The committed SQL artifact must materialize the Python contract."""

    sql_text = (project_root / "sql" / "create_v2_analytics_views.sql").read_text(
        encoding="utf-8"
    )

    assert sql_text.count("CREATE OR REPLACE VIEW") == 8
    assert all(spec.relation in sql_text for spec in TABLE_SPECS.values())
    assert "first_name" not in sql_text
    assert "last_name" not in sql_text
    assert "reviewer_id" not in sql_text


def test_normalization_equalizes_csv_and_postgresql_types() -> None:
    """Dates and NUMERIC values should hash identically across backends."""

    spec = TABLE_SPECS["compensation_history.csv"]
    csv_like = pd.DataFrame(
        {
            "compensation_id": [2, 1],
            "employee_id": [20, 10],
            "effective_date": ["2025-01-01", "2024-01-01"],
            "base_salary": [90000.0, 80000.0],
            "bonus_target": [0.1, 0.05],
            "equity_value": [1000.0, 0.0],
            "change_reason": ["Merit", "Hire"],
        }
    )
    postgres_like = pd.DataFrame(
        {
            "compensation_id": [1, 2],
            "employee_id": [10, 20],
            "effective_date": [date(2024, 1, 1), date(2025, 1, 1)],
            "base_salary": [Decimal("80000.00"), Decimal("90000.00")],
            "bonus_target": [Decimal("0.05"), Decimal("0.10")],
            "equity_value": [Decimal("0.00"), Decimal("1000.00")],
            "change_reason": ["Hire", "Merit"],
        }
    )

    assert frame_fingerprint(csv_like, spec) == frame_fingerprint(
        postgres_like,
        spec,
    )
    assert normalize_frame(csv_like, spec)["compensation_id"].tolist() == [
        1,
        2,
    ]


def test_temporal_output_hash_ignores_platform_line_endings(
    tmp_path: Path,
) -> None:
    """Equivalent CSV types and line endings must hash identically."""

    unix_path = tmp_path / "unix.csv"
    windows_path = tmp_path / "windows.csv"
    unix_path.write_bytes(b"employee_id,salary,value\n1,100000,alpha\n2,90000,beta\n")
    windows_path.write_bytes(
        b"employee_id,salary,value\r\n1,100000.0,alpha\r\n2,90000.00,beta\r\n"
    )

    assert canonical_csv_sha256(unix_path) == canonical_csv_sha256(windows_path)


def test_default_pipeline_builds_v2_from_postgresql_before_analysis() -> None:
    """The full pipeline must use PostgreSQL at the V2 source boundary."""

    steps = build_pipeline_steps(pipeline_args(skip_postgres=False))
    postgres_end = max(steps.index(step) for step in POSTGRES_STEPS)
    builder = steps.index(V2_POSTGRES_DATASET_STEP)
    validator = steps.index(V2_POSTGRES_VALIDATION_STEP)
    first_analysis = steps.index(VERSION_2_ANALYSIS_STEPS[0])

    assert postgres_end < builder < validator < first_analysis
    assert V2_CSV_DATASET_STEP not in steps


def test_skip_postgres_uses_explicit_csv_fallback() -> None:
    """Reviewers without a database retain a complete file-backed path."""

    steps = build_pipeline_steps(pipeline_args(skip_postgres=True))

    assert steps[:2] == [V2_CSV_DATASET_STEP, VERSION_2_ANALYSIS_STEPS[0]]
    assert V2_POSTGRES_DATASET_STEP not in steps
    assert V2_POSTGRES_VALIDATION_STEP not in steps
