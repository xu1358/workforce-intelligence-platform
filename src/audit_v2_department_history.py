"""Audit Version 2 department continuity and transfer-event semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


plt.switch_backend("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "v2_department_history.yaml"
PANEL_COLUMNS = [
    "employee_id",
    "snapshot_sequence",
    "snapshot_date",
    "department_name",
    "city",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the Version 2 department-history contract."""

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Department-history configuration must be a mapping.")
    return config


def load_inputs(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load only the fields required for the aggregate history audit."""

    inputs = config["inputs"]
    employees = pd.read_csv(
        project_root / inputs["employees"],
        usecols=["employee_id", "department_id", "location_id"],
    )
    events = pd.read_csv(
        project_root / inputs["employee_events"],
        usecols=[
            "event_id",
            "employee_id",
            "event_date",
            "event_type",
            "old_value",
            "new_value",
            "notes",
        ],
    )
    departments = pd.read_csv(
        project_root / inputs["departments"],
        usecols=["department_id", "department_name"],
    )
    locations = pd.read_csv(
        project_root / inputs["locations"],
        usecols=["location_id", "city"],
    )
    panel = pd.read_csv(
        project_root / inputs["temporal_panel"],
        usecols=PANEL_COLUMNS,
    )
    return employees, events, departments, locations, panel


def parse_hire_assignments(events: pd.DataFrame) -> pd.DataFrame:
    """Parse immutable hire department and initial location from event notes."""

    hires = events[events["event_type"].eq("Hire")].copy()
    hires["hire_department_id"] = pd.to_numeric(
        hires["notes"].str.extract(r"department_id=(\d+)")[0],
        errors="coerce",
    ).astype("Int64")
    hires["hire_location_id"] = pd.to_numeric(
        hires["notes"].str.extract(r"location_id=(\d+)")[0],
        errors="coerce",
    ).astype("Int64")
    return hires[
        [
            "employee_id",
            "event_date",
            "hire_department_id",
            "hire_location_id",
        ]
    ]


def classify_transfer_semantics(events: pd.DataFrame) -> pd.DataFrame:
    """Classify transfer events from their explicit generator notes."""

    transfers = events[events["event_type"].eq("Transfer")].copy()
    notes = transfers["notes"].fillna("")
    transfers["transfer_semantic"] = "Other or unspecified"
    transfers.loc[
        notes.str.contains("location_id", case=False),
        "transfer_semantic",
    ] = "Location"
    transfers.loc[
        notes.str.contains("department_id", case=False),
        "transfer_semantic",
    ] = "Department"
    return transfers


def build_event_type_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Count event types without saving employee-level records."""

    return (
        events.groupby("event_type", dropna=False)
        .agg(events=("event_id", "size"), unique_employees=("employee_id", "nunique"))
        .reset_index()
        .sort_values("events", ascending=False)
    )


def build_department_continuity(
    employees: pd.DataFrame,
    departments: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize department continuity by current department."""

    panel_counts = (
        panel.groupby("employee_id")["department_name"]
        .nunique()
        .rename("panel_department_count")
        .reset_index()
    )
    current = employees.merge(departments, on="department_id", how="left")
    current = current.merge(panel_counts, on="employee_id", how="left")
    current["appears_in_panel"] = current["panel_department_count"].notna()
    current["multiple_panel_departments"] = current[
        "panel_department_count"
    ].fillna(0).gt(1)
    return (
        current.groupby("department_name", dropna=False)
        .agg(
            source_employees=("employee_id", "size"),
            panel_employees=("appears_in_panel", "sum"),
            employees_with_multiple_panel_departments=(
                "multiple_panel_departments",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("source_employees", ascending=False)
    )


def build_location_reconstruction(
    employees: pd.DataFrame,
    locations: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Count historical rows whose snapshot location differs from final state."""

    current_locations = employees.merge(
        locations,
        on="location_id",
        how="left",
    )[["employee_id", "city"]].rename(columns={"city": "current_city"})
    compared = panel.merge(current_locations, on="employee_id", how="left")
    compared["reconstructed_away_from_current"] = compared["city"].ne(
        compared["current_city"]
    )
    return (
        compared.groupby("snapshot_sequence", dropna=False)
        .agg(
            rows=("employee_id", "size"),
            unique_employees=("employee_id", "nunique"),
            reconstructed_rows=("reconstructed_away_from_current", "sum"),
            reconstructed_employees=(
                "employee_id",
                lambda values: values[
                    compared.loc[
                        values.index,
                        "reconstructed_away_from_current",
                    ]
                ].nunique(),
            ),
        )
        .reset_index()
    )


def build_summary(
    employees: pd.DataFrame,
    hires: pd.DataFrame,
    transfers: pd.DataFrame,
    panel: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    """Create the headline Version 2 history-attribution metrics."""

    hire_comparison = employees.merge(hires, on="employee_id", how="left")
    panel_department_counts = panel.groupby("employee_id")[
        "department_name"
    ].nunique()
    current_locations = employees.merge(
        locations,
        on="location_id",
        how="left",
    )[["employee_id", "city"]].rename(columns={"city": "current_city"})
    location_comparison = panel.merge(
        current_locations,
        on="employee_id",
        how="left",
    )
    location_difference = location_comparison["city"].ne(
        location_comparison["current_city"]
    )

    metrics = [
        ("Source employees", len(employees), "rows"),
        ("Hire events", len(hires), "events"),
        (
            "Parsed hire departments",
            int(hires["hire_department_id"].notna().sum()),
            "events",
        ),
        (
            "Hire-to-current department mismatches",
            int(
                hire_comparison["department_id"].ne(
                    hire_comparison["hire_department_id"]
                ).sum()
            ),
            "employees",
        ),
        (
            "Current location differs from hire",
            int(
                hire_comparison["location_id"].ne(
                    hire_comparison["hire_location_id"]
                ).sum()
            ),
            "employees",
        ),
        ("Transfer events", len(transfers), "events"),
        (
            "Employees with transfer events",
            transfers["employee_id"].nunique(),
            "employees",
        ),
        (
            "Location-semantic transfer events",
            int(transfers["transfer_semantic"].eq("Location").sum()),
            "events",
        ),
        (
            "Department-semantic transfer events",
            int(transfers["transfer_semantic"].eq("Department").sum()),
            "events",
        ),
        ("Panel unique employees", panel["employee_id"].nunique(), "employees"),
        (
            "Employees with multiple panel departments",
            int(panel_department_counts.gt(1).sum()),
            "employees",
        ),
        (
            "Historical rows reconstructed away from current location",
            int(location_difference.sum()),
            "rows",
        ),
        (
            "Employees reconstructed away from current location",
            location_comparison.loc[
                location_difference,
                "employee_id",
            ].nunique(),
            "employees",
        ),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "unit"])


def add_check(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> None:
    """Append one validation result."""

    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def metric_value(summary: pd.DataFrame, metric: str) -> int:
    """Return one integer headline metric."""

    row = summary[summary["metric"].eq(metric)]
    if len(row) != 1:
        raise ValueError(f"Expected one summary metric {metric!r}.")
    return int(row.iloc[0]["value"])


def build_checks(
    summary: pd.DataFrame,
    transfers: pd.DataFrame,
    continuity: pd.DataFrame,
    reconstruction: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate department invariance and historical location reconstruction."""

    expected = config["expected"]
    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "Version 2 source and hire populations reconcile",
        metric_value(summary, "Source employees") == expected["employees"]
        and metric_value(summary, "Hire events") == expected["hire_events"]
        and metric_value(summary, "Parsed hire departments")
        == expected["parsed_hire_departments"],
        {
            "employees": metric_value(summary, "Source employees"),
            "hire_events": metric_value(summary, "Hire events"),
            "parsed_hire_departments": metric_value(
                summary,
                "Parsed hire departments",
            ),
        },
        {
            "employees": expected["employees"],
            "hire_events": expected["hire_events"],
            "parsed_hire_departments": expected["parsed_hire_departments"],
        },
        "Every Version 2 employee has a parseable hire department.",
    )
    add_check(
        checks,
        "Hire department remains the current department",
        metric_value(summary, "Hire-to-current department mismatches")
        == expected["hire_to_current_department_mismatches"],
        metric_value(summary, "Hire-to-current department mismatches"),
        expected["hire_to_current_department_mismatches"],
        "Department is invariant by generator design, not reconstructed from transfers.",
    )
    add_check(
        checks,
        "Transfer events are explicitly location-semantic",
        len(transfers) == expected["transfer_events"]
        and transfers["employee_id"].nunique()
        == expected["transfer_employees"]
        and int(transfers["transfer_semantic"].eq("Location").sum())
        == expected["location_semantic_transfer_events"]
        and int(transfers["transfer_semantic"].eq("Department").sum())
        == expected["department_semantic_transfer_events"],
        transfers["transfer_semantic"].value_counts().to_dict(),
        {
            "location": expected["location_semantic_transfer_events"],
            "department": expected["department_semantic_transfer_events"],
        },
        "Notebook 19's hypothesis is re-audited on Version 2 source events.",
    )
    add_check(
        checks,
        "Temporal panel contains no department changes",
        metric_value(summary, "Panel unique employees")
        == expected["panel_unique_employees"]
        and metric_value(
            summary,
            "Employees with multiple panel departments",
        )
        == expected["employees_with_multiple_panel_departments"]
        and int(
            continuity["employees_with_multiple_panel_departments"].sum()
        )
        == 0,
        {
            "panel_employees": metric_value(
                summary,
                "Panel unique employees",
            ),
            "multiple_departments": metric_value(
                summary,
                "Employees with multiple panel departments",
            ),
        },
        {
            "panel_employees": expected["panel_unique_employees"],
            "multiple_departments": 0,
        },
        "No current department is backfilled into a changing department history.",
    )
    add_check(
        checks,
        "Location history is reconstructed rather than backfilled",
        metric_value(
            summary,
            "Historical rows reconstructed away from current location",
        )
        == expected["historical_rows_reconstructed_away_from_current_location"]
        and metric_value(
            summary,
            "Employees reconstructed away from current location",
        )
        == expected["employees_reconstructed_away_from_current_location"]
        and reconstruction["reconstructed_rows"].astype(int).tolist()
        == expected["reconstructed_rows_by_snapshot"],
        {
            "rows": metric_value(
                summary,
                "Historical rows reconstructed away from current location",
            ),
            "employees": metric_value(
                summary,
                "Employees reconstructed away from current location",
            ),
            "by_snapshot": reconstruction["reconstructed_rows"]
            .astype(int)
            .tolist(),
        },
        {
            "rows": expected[
                "historical_rows_reconstructed_away_from_current_location"
            ],
            "employees": expected[
                "employees_reconstructed_away_from_current_location"
            ],
            "by_snapshot": expected["reconstructed_rows_by_snapshot"],
        },
        "Dated location transfers prevent final-location attribution in historical rows.",
    )
    add_check(
        checks,
        "Current and hire locations are not assumed identical",
        metric_value(summary, "Current location differs from hire")
        == expected["current_location_differs_from_hire"],
        metric_value(summary, "Current location differs from hire"),
        expected["current_location_differs_from_hire"],
        "The audit distinguishes immutable department from mutable location.",
    )
    leaked_columns = sorted(
        {
            column
            for frame in [summary, continuity, reconstruction]
            for column in frame.columns
            if column in {"employee_id", "attrition_next_12m"}
        }
    )
    add_check(
        checks,
        "Saved history outputs are aggregate and target-free",
        leaked_columns == [] and "attrition_next_12m" not in PANEL_COLUMNS,
        leaked_columns,
        [],
        "The audit exports neither employee identifiers nor outcome columns.",
    )
    governance = config["governance"]
    add_check(
        checks,
        "History audit preserves Version 2 analytical results",
        governance["version_2_is_authoritative"]
        and governance["department_is_time_invariant_by_generator_design"]
        and governance["transfer_events_represent_location"]
        and not governance["target_column_accessed"]
        and not governance["causal_claims_permitted"]
        and not governance["changes_model_inputs"]
        and not governance["changes_selected_model"]
        and not governance["changes_frozen_policy"],
        governance,
        "Target-free structural audit with no model changes",
        "Checkpoint 66 replaces the missing V2 evidence, not the data design.",
    )
    return pd.DataFrame(checks)


def plot_transfer_semantics(
    transfers: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot location versus department transfer semantics."""

    counts = (
        transfers["transfer_semantic"]
        .value_counts()
        .reindex(["Location", "Department", "Other or unspecified"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=["#2F6B8A", "#C8553D", "#999999"])
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 8,
            f"{value:,}",
            ha="center",
        )
    ax.set_ylabel("Transfer events")
    ax.set_title("Version 2 transfer events represent location changes")
    ax.set_ylim(0, max(counts.max() * 1.15, 10))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_location_reconstruction(
    reconstruction: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot historical rows reconstructed away from current location."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [
        f"Snapshot {value}"
        for value in reconstruction["snapshot_sequence"].astype(int)
    ]
    bars = ax.bar(labels, reconstruction["reconstructed_rows"], color="#4C956C")
    for bar, value in zip(bars, reconstruction["reconstructed_rows"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{int(value)}",
            ha="center",
        )
    ax.set_ylabel("Historical rows")
    ax.set_title("Rows protected from final-location backfill")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def display_table(frame: pd.DataFrame) -> str:
    """Return a readable console table."""

    return frame.to_string(index=False)


def main() -> None:
    """Run the Version 2 department-history audit."""

    config = load_config()
    employees, events, departments, locations, panel = load_inputs(
        PROJECT_ROOT,
        config,
    )
    hires = parse_hire_assignments(events)
    transfers = classify_transfer_semantics(events)
    event_types = build_event_type_summary(events)
    continuity = build_department_continuity(
        employees,
        departments,
        panel,
    )
    reconstruction = build_location_reconstruction(
        employees,
        locations,
        panel,
    )
    summary = build_summary(
        employees,
        hires,
        transfers,
        panel,
        locations,
    )
    checks = build_checks(
        summary,
        transfers,
        continuity,
        reconstruction,
        config,
    )

    output_config = config["outputs"]
    output_directory = PROJECT_ROOT / output_config["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "summary": summary,
        "event_types": event_types,
        "continuity": continuity,
        "location_reconstruction": reconstruction,
        "validation": checks,
    }
    for key, frame in frames.items():
        frame.to_csv(
            output_directory / output_config["files"][key],
            index=False,
        )

    plot_transfer_semantics(
        transfers,
        output_directory / output_config["figures"][0],
    )
    plot_location_reconstruction(
        reconstruction,
        output_directory / output_config["figures"][1],
    )

    print("\nVERSION 2 DEPARTMENT-HISTORY SUMMARY")
    print(display_table(summary))
    print("\nVERSION 2 EVENT-TYPE SUMMARY")
    print(display_table(event_types))
    print("\nHISTORICAL LOCATION RECONSTRUCTION")
    print(display_table(reconstruction))
    print("\nVERSION 2 DEPARTMENT-HISTORY VALIDATION")
    print(display_table(checks))

    failures = checks[checks["status"].ne("PASS")]
    if not failures.empty:
        raise ValueError(
            "Version 2 department-history validation failed:\n"
            + display_table(failures)
        )

    print(f"\nSaved Version 2 history outputs to: {output_directory}")
    print("Outcome columns accessed: 0; model and policy changes made: 0")
    print("\nVERSION 2 DEPARTMENT-HISTORY AUDIT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
