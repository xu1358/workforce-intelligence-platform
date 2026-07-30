"""Version 2 source access for CSV and PostgreSQL analytical views."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "v2_postgresql_source.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class TableSpec:
    """One least-privilege Version 2 analytical relation."""

    filename: str
    relation: str
    primary_key: str
    columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()


TABLE_SPECS = {
    "employees.csv": TableSpec(
        filename="employees.csv",
        relation="analytics_v2_employees",
        primary_key="employee_id",
        columns=(
            "employee_id",
            "hire_date",
            "termination_date",
            "department_id",
            "location_id",
            "job_role_id",
            "employment_type",
            "birth_year",
            "education_level",
            "organizational_level",
        ),
        date_columns=("hire_date", "termination_date"),
        numeric_columns=(
            "employee_id",
            "department_id",
            "location_id",
            "job_role_id",
            "birth_year",
        ),
    ),
    "departments.csv": TableSpec(
        filename="departments.csv",
        relation="analytics_v2_departments",
        primary_key="department_id",
        columns=(
            "department_id",
            "department_name",
            "department_group",
        ),
        numeric_columns=("department_id",),
    ),
    "locations.csv": TableSpec(
        filename="locations.csv",
        relation="analytics_v2_locations",
        primary_key="location_id",
        columns=("location_id", "city", "region", "location_type"),
        numeric_columns=("location_id",),
    ),
    "job_roles.csv": TableSpec(
        filename="job_roles.csv",
        relation="analytics_v2_job_roles",
        primary_key="job_role_id",
        columns=("job_role_id", "job_title", "job_family", "job_level"),
        numeric_columns=("job_role_id",),
    ),
    "compensation_history.csv": TableSpec(
        filename="compensation_history.csv",
        relation="analytics_v2_compensation_history",
        primary_key="compensation_id",
        columns=(
            "compensation_id",
            "employee_id",
            "effective_date",
            "base_salary",
            "bonus_target",
            "equity_value",
            "change_reason",
        ),
        date_columns=("effective_date",),
        numeric_columns=(
            "compensation_id",
            "employee_id",
            "base_salary",
            "bonus_target",
            "equity_value",
        ),
    ),
    "performance_reviews.csv": TableSpec(
        filename="performance_reviews.csv",
        relation="analytics_v2_performance_reviews",
        primary_key="review_id",
        columns=(
            "review_id",
            "employee_id",
            "review_date",
            "performance_rating",
            "goal_completion",
            "promotion_recommended",
        ),
        date_columns=("review_date",),
        numeric_columns=(
            "review_id",
            "employee_id",
            "performance_rating",
            "goal_completion",
        ),
    ),
    "training_records.csv": TableSpec(
        filename="training_records.csv",
        relation="analytics_v2_training_records",
        primary_key="training_record_id",
        columns=(
            "training_record_id",
            "employee_id",
            "completion_date",
            "completion_status",
            "training_hours",
            "score",
        ),
        date_columns=("completion_date",),
        numeric_columns=(
            "training_record_id",
            "employee_id",
            "training_hours",
            "score",
        ),
    ),
    "employee_events.csv": TableSpec(
        filename="employee_events.csv",
        relation="analytics_v2_employee_events",
        primary_key="event_id",
        columns=(
            "event_id",
            "employee_id",
            "event_date",
            "event_type",
            "new_value",
            "notes",
        ),
        date_columns=("event_date",),
        numeric_columns=("event_id", "employee_id"),
    ),
}


def load_source_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the Version 2 PostgreSQL source contract."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_database_config(
    project_root: Path = PROJECT_ROOT,
    source_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Load PostgreSQL connection settings without exposing credentials."""

    config = source_config or load_source_config()
    env_path = project_root / config["database"]["env_file"]
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing {env_path.name}. Copy .env.example and add local "
            "PostgreSQL credentials."
        )

    from dotenv import load_dotenv

    load_dotenv(env_path)
    environment = config["database"]["connection_environment"]
    values = {
        key: os.getenv(environment_name)
        for key, environment_name in environment.items()
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing PostgreSQL settings: {missing}")

    return {key: str(value) for key, value in values.items()}


def normalize_frame(frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Normalize source types so CSV and PostgreSQL have one contract."""

    missing = [column for column in spec.columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{spec.filename} is missing columns: {missing}")

    normalized = frame.loc[:, list(spec.columns)].copy()
    for column in spec.date_columns:
        normalized[column] = pd.to_datetime(
            normalized[column],
            errors="coerce",
        )
    for column in spec.numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    return normalized.sort_values(spec.primary_key).reset_index(drop=True)


def frame_fingerprint(frame: pd.DataFrame, spec: TableSpec) -> str:
    """Return a stable content fingerprint across database and CSV types."""

    normalized = normalize_frame(frame, spec)
    canonical = normalized.copy()

    for column in canonical.columns:
        if column in spec.date_columns:
            canonical[column] = canonical[column].dt.strftime("%Y-%m-%d")
        elif column in spec.numeric_columns:
            canonical[column] = canonical[column].map(
                lambda value: (
                    "<NULL>"
                    if pd.isna(value)
                    else format(float(value), ".12g")
                )
            )
        else:
            canonical[column] = canonical[column].map(
                lambda value: "<NULL>" if pd.isna(value) else str(value)
            )

    canonical = canonical.fillna("<NULL>")
    payload = canonical.to_csv(index=False, lineterminator="\n")
    return sha256(payload.encode("utf-8")).hexdigest()


class V2DataSource:
    """Load the V2 temporal builder's inputs from CSV or PostgreSQL."""

    def __init__(
        self,
        backend: str,
        *,
        project_root: Path = PROJECT_ROOT,
        connection: Any | None = None,
    ) -> None:
        if backend not in {"csv", "postgresql"}:
            raise ValueError(
                f"Unsupported Version 2 source backend: {backend}"
            )

        self.backend = backend
        self.project_root = project_root
        self.source_config = load_source_config(
            project_root / "config" / "v2_postgresql_source.yaml"
        )
        self.schema = self.source_config["database"]["schema"]
        self.connection = connection
        self._owns_connection = False
        self.audit_records: list[dict[str, Any]] = []

    def __enter__(self) -> V2DataSource:
        if self.backend == "postgresql" and self.connection is None:
            import psycopg2

            self.connection = psycopg2.connect(
                **load_database_config(
                    self.project_root,
                    self.source_config,
                )
            )
            self._owns_connection = True
        return self

    def __exit__(self, *_: object) -> None:
        if self._owns_connection and self.connection is not None:
            self.connection.close()
            self.connection = None
            self._owns_connection = False

    def load_table(self, filename: str) -> pd.DataFrame:
        """Load one configured analytical table."""

        if filename not in TABLE_SPECS:
            raise KeyError(f"Unsupported Version 2 source table: {filename}")

        spec = TABLE_SPECS[filename]
        configured = self.source_config["source_tables"][filename]
        if (
            configured["relation"] != spec.relation
            or configured["primary_key"] != spec.primary_key
        ):
            raise ValueError(
                f"Source configuration does not match code contract for "
                f"{filename}."
            )

        if self.backend == "csv":
            path = self.project_root / "data" / "raw" / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {filename}. Run the data-generation path first."
                )
            frame = pd.read_csv(path, usecols=list(spec.columns))
            relation = path.relative_to(self.project_root).as_posix()
        else:
            if self.connection is None:
                raise RuntimeError(
                    "PostgreSQL source must be used inside its context manager."
                )
            from psycopg2 import sql

            query = sql.SQL("SELECT {} FROM {}.{} ORDER BY {};").format(
                sql.SQL(", ").join(
                    sql.Identifier(column) for column in spec.columns
                ),
                sql.Identifier(self.schema),
                sql.Identifier(spec.relation),
                sql.Identifier(spec.primary_key),
            )
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                frame = pd.DataFrame(
                    cursor.fetchall(),
                    columns=list(spec.columns),
                )
            relation = f"{self.schema}.{spec.relation}"

        normalized = normalize_frame(frame, spec)
        self.audit_records.append(
            {
                "backend": self.backend,
                "source_file": filename,
                "relation": relation,
                "primary_key": spec.primary_key,
                "rows": len(normalized),
                "columns": len(normalized.columns),
                "content_sha256": frame_fingerprint(normalized, spec),
            }
        )
        return normalized

    def provenance_frame(self) -> pd.DataFrame:
        """Return aggregate query provenance without employee-level data."""

        return pd.DataFrame(self.audit_records)
