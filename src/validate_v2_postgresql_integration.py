"""Validate PostgreSQL as an executable Version 2 analytical source."""

from __future__ import annotations

import csv
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

try:
    from v2_data_access import (
        TABLE_SPECS,
        V2DataSource,
        frame_fingerprint,
        load_source_config,
    )
except ModuleNotFoundError:
    from src.v2_data_access import (
        TABLE_SPECS,
        V2DataSource,
        frame_fingerprint,
        load_source_config,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "v2_postgresql_integration"
TEMPORAL_VALIDATION_DIR = (
    PROJECT_ROOT / "data" / "processed" / "temporal_dataset_validation"
)


def load_runner_module(project_root: Path = PROJECT_ROOT) -> Any:
    """Load the portable runner without relying on repository sys.path."""

    module_path = project_root / "scripts" / "run_project.py"
    spec = importlib.util.spec_from_file_location(
        "v2_postgresql_runner_contract",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load project runner: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_csv_sha256(path: Path) -> str:
    """Hash CSV values after normalizing numeric types and line endings."""

    frame = pd.read_csv(path)
    canonical = pd.DataFrame(index=frame.index)

    for column in frame.columns:
        values = frame[column]
        nonmissing = values.loc[values.notna()]
        numeric = pd.to_numeric(nonmissing, errors="coerce")
        is_numeric = nonmissing.empty or bool(numeric.notna().all())

        if is_numeric:
            converted = pd.to_numeric(values, errors="coerce")
            canonical[column] = converted.map(
                lambda value: (
                    "<NULL>" if pd.isna(value) else format(float(value), ".12g")
                )
            )
        else:
            canonical[column] = values.map(
                lambda value: "<NULL>" if pd.isna(value) else str(value)
            )

    payload = canonical.to_csv(index=False, lineterminator="\n")
    return sha256(payload.encode("utf-8")).hexdigest()


def add_check(
    records: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> None:
    """Append one validation result."""

    records.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def compare_sources(
    csv_source: V2DataSource,
    postgres_source: V2DataSource,
) -> pd.DataFrame:
    """Compare every V2 PostgreSQL view with its CSV source."""

    records: list[dict[str, Any]] = []
    for filename, spec in TABLE_SPECS.items():
        csv_frame = csv_source.load_table(filename)
        postgres_frame = postgres_source.load_table(filename)
        csv_hash = frame_fingerprint(csv_frame, spec)
        postgres_hash = frame_fingerprint(postgres_frame, spec)
        records.append(
            {
                "source_file": filename,
                "postgresql_relation": (f"{postgres_source.schema}.{spec.relation}"),
                "primary_key": spec.primary_key,
                "csv_rows": len(csv_frame),
                "postgresql_rows": len(postgres_frame),
                "row_count_match": len(csv_frame) == len(postgres_frame),
                "column_order_match": (
                    list(csv_frame.columns) == list(postgres_frame.columns)
                ),
                "csv_duplicate_keys": int(
                    csv_frame[spec.primary_key].duplicated().sum()
                ),
                "postgresql_duplicate_keys": int(
                    postgres_frame[spec.primary_key].duplicated().sum()
                ),
                "csv_content_sha256": csv_hash,
                "postgresql_content_sha256": postgres_hash,
                "content_match": csv_hash == postgres_hash,
            }
        )

    return pd.DataFrame(records)


def validate_temporal_outputs(
    project_root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Reconcile PostgreSQL-built temporal outputs to frozen fingerprints."""

    provenance_path = (
        project_root
        / "data"
        / "processed"
        / "temporal_dataset_validation"
        / "source_provenance.csv"
    )
    provenance = pd.read_csv(provenance_path)
    rows: list[dict[str, Any]] = []

    for relative_path, expected in config["expected_temporal_outputs"].items():
        path = project_root / relative_path
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            row_count = sum(1 for _ in reader)
        actual_hash = canonical_csv_sha256(path)
        rows.append(
            {
                "file": relative_path,
                "rows": row_count,
                "expected_rows": expected["rows"],
                "columns": len(header),
                "expected_columns": expected["columns"],
                "canonical_content_sha256": actual_hash,
                "expected_canonical_content_sha256": expected[
                    "canonical_content_sha256"
                ],
                "matches_frozen_output": (
                    row_count == expected["rows"]
                    and len(header) == expected["columns"]
                    and actual_hash == expected["canonical_content_sha256"]
                ),
            }
        )

    if set(provenance["backend"]) != {"postgresql"}:
        raise ValueError("Temporal source provenance does not identify PostgreSQL.")

    return pd.DataFrame(rows)


def build_checks(
    parity: pd.DataFrame,
    temporal: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build reviewer-facing PostgreSQL integration checks."""

    records: list[dict[str, Any]] = []
    validation = config["validation"]
    source_count = len(parity)
    add_check(
        records,
        "Every Version 2 source table is queried",
        source_count == validation["expected_source_table_count"],
        source_count,
        validation["expected_source_table_count"],
        "The temporal builder reads eight normalized analytical views.",
    )
    add_check(
        records,
        "PostgreSQL and CSV row counts reconcile",
        bool(parity["row_count_match"].all()),
        parity.loc[~parity["row_count_match"], "source_file"].tolist(),
        "0 mismatched tables",
        "Database ingestion cannot silently drop or duplicate records.",
    )
    add_check(
        records,
        "PostgreSQL and CSV column contracts reconcile",
        bool(parity["column_order_match"].all()),
        parity.loc[~parity["column_order_match"], "source_file"].tolist(),
        "0 mismatched tables",
        "The database path supplies exactly the fields expected by Version 2.",
    )
    duplicate_total = int(
        parity["csv_duplicate_keys"].sum() + parity["postgresql_duplicate_keys"].sum()
    )
    add_check(
        records,
        "Source primary keys remain unique",
        duplicate_total == 0,
        duplicate_total,
        0,
        "Every relational source preserves its declared grain.",
    )
    add_check(
        records,
        "PostgreSQL and CSV content fingerprints reconcile",
        bool(parity["content_match"].all()),
        parity.loc[~parity["content_match"], "source_file"].tolist(),
        "0 mismatched tables",
        "Canonical hashes compare all values used by Version 2.",
    )
    add_check(
        records,
        "PostgreSQL-built temporal datasets remain frozen",
        bool(temporal["matches_frozen_output"].all()),
        temporal.loc[~temporal["matches_frozen_output"], "file"].tolist(),
        "0 changed analytical datasets",
        "Database integration changes the source boundary, not model inputs.",
    )
    exposed_columns = {
        column for spec in TABLE_SPECS.values() for column in spec.columns
    }
    direct_identifiers = sorted(
        {"first_name", "last_name", "manager_id", "reviewer_id"} & exposed_columns
    )
    add_check(
        records,
        "Analytical views follow least-privilege projection",
        not direct_identifiers,
        direct_identifiers,
        "No direct names, manager IDs, or reviewer IDs",
        "The V2 query boundary excludes fields not needed for modeling.",
    )
    runner = load_runner_module()
    default_steps = runner.build_pipeline_steps(
        type(
            "Args",
            (),
            {
                "skip_quality": False,
                "skip_data_generation": False,
                "skip_postgres": False,
                "include_legacy_v1": False,
                "skip_external_benchmark": False,
            },
        )()
    )
    postgres_end = max(default_steps.index(step) for step in runner.POSTGRES_STEPS)
    builder_position = default_steps.index(runner.V2_POSTGRES_DATASET_STEP)
    validator_position = default_steps.index(runner.V2_POSTGRES_VALIDATION_STEP)
    add_check(
        records,
        "Default Version 2 pipeline uses PostgreSQL before modeling",
        postgres_end < builder_position < validator_position,
        {
            "postgres_preparation_end": postgres_end + 1,
            "postgres_builder": builder_position + 1,
            "parity_validation": validator_position + 1,
        },
        "PostgreSQL preparation -> V2 builder -> parity validation",
        "The database is no longer a detached post-modeling side path.",
    )
    governance = config["governance"]
    governance_ok = (
        governance["database_integration_only"]
        and not governance["changes_synthetic_records"]
        and not governance["changes_temporal_dataset_values"]
        and not governance["changes_model_results"]
        and not governance["reopens_final_test"]
        and not governance["changes_frozen_policy"]
        and not governance["changes_dashboard_data"]
    )
    add_check(
        records,
        "Frozen analytical governance is preserved",
        governance_ok,
        governance,
        "Database integration only",
        "No model, final-test, policy, or dashboard decision is reopened.",
    )

    return pd.DataFrame(records)


def main() -> None:
    """Run the PostgreSQL source and temporal-output parity audit."""

    config = load_source_config()
    with (
        V2DataSource("csv", project_root=PROJECT_ROOT) as csv_source,
        V2DataSource(
            "postgresql",
            project_root=PROJECT_ROOT,
        ) as postgres_source,
    ):
        parity = compare_sources(csv_source, postgres_source)
        query_provenance = postgres_source.provenance_frame()

    temporal = validate_temporal_outputs(PROJECT_ROOT, config)
    checks = build_checks(parity, temporal, config)
    failures = checks.loc[checks["status"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(
            "Version 2 PostgreSQL integration validation failed:\n"
            + failures.to_string(index=False)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parity.to_csv(OUTPUT_DIR / "source_table_parity.csv", index=False)
    temporal.to_csv(OUTPUT_DIR / "temporal_output_parity.csv", index=False)
    query_provenance.to_csv(
        OUTPUT_DIR / "postgresql_query_provenance.csv",
        index=False,
    )
    checks.to_csv(OUTPUT_DIR / "validation_checks.csv", index=False)

    print("\nVERSION 2 POSTGRESQL SOURCE PARITY")
    print(
        parity[
            [
                "source_file",
                "postgresql_relation",
                "csv_rows",
                "postgresql_rows",
                "content_match",
            ]
        ].to_string(index=False)
    )
    print("\nPOSTGRESQL-BUILT TEMPORAL OUTPUT PARITY")
    print(
        temporal[
            [
                "file",
                "rows",
                "columns",
                "matches_frozen_output",
            ]
        ].to_string(index=False)
    )
    print("\nVERSION 2 POSTGRESQL INTEGRATION VALIDATION")
    print(checks.to_string(index=False))
    print(f"\nSaved PostgreSQL integration outputs to: {OUTPUT_DIR}")
    print("\nVERSION 2 POSTGRESQL INTEGRATION VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
