"""Quantify department and location history attribution bias."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "department_history_bias"
)
FIGURE_DIR = OUTPUT_DIR / "figures"


def assign_historical_location(
    records: pd.DataFrame,
    date_column: str,
    domain: str,
    employees: pd.DataFrame,
    transfers: pd.DataFrame,
    location_map: dict[int, str],
) -> pd.DataFrame:
    """Assign current and event-time locations to records."""

    result = records.copy()

    result["record_date"] = pd.to_datetime(
        result[date_column]
    )

    result = result.merge(
        employees[
            [
                "employee_id",
                "location_id",
            ]
        ].rename(
            columns={
                "location_id": (
                    "current_location_id"
                )
            }
        ),
        on="employee_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        transfers[
            [
                "employee_id",
                "transfer_date",
                "old_location_id",
                "new_location_id",
            ]
        ],
        on="employee_id",
        how="left",
        validate="many_to_one",
    )

    result["historical_location_id"] = (
        result["current_location_id"]
    )

    before_transfer = (
        result["transfer_date"].notna()
        & result["record_date"].lt(
            result["transfer_date"]
        )
    )

    result.loc[
        before_transfer,
        "historical_location_id",
    ] = result.loc[
        before_transfer,
        "old_location_id",
    ]

    result["current_location_id"] = (
        result["current_location_id"].astype(int)
    )

    result["historical_location_id"] = (
        result["historical_location_id"]
        .astype(int)
    )

    result["current_location"] = (
        result["current_location_id"].map(
            location_map
        )
    )

    result["historical_location"] = (
        result["historical_location_id"].map(
            location_map
        )
    )

    result["location_reassigned"] = (
        result["current_location_id"].ne(
            result["historical_location_id"]
        )
    )

    result["domain"] = domain

    return result


def compare_location_metric(
    records: pd.DataFrame,
    domain: str,
    metric: str,
    value_column: str | None = None,
    aggregation: str = "count",
) -> pd.DataFrame:
    """Compare a metric under two attribution methods."""

    if aggregation == "count":
        current = (
            records.groupby(
                "current_location"
            )
            .size()
            .rename("current_value")
        )

        historical = (
            records.groupby(
                "historical_location"
            )
            .size()
            .rename("historical_value")
        )

    elif aggregation == "mean":
        if value_column is None:
            raise ValueError(
                "value_column is required for mean."
            )

        current = (
            records.groupby(
                "current_location"
            )[value_column]
            .mean()
            .rename("current_value")
        )

        historical = (
            records.groupby(
                "historical_location"
            )[value_column]
            .mean()
            .rename("historical_value")
        )

    elif aggregation == "sum":
        if value_column is None:
            raise ValueError(
                "value_column is required for sum."
            )

        current = (
            records.groupby(
                "current_location"
            )[value_column]
            .sum()
            .rename("current_value")
        )

        historical = (
            records.groupby(
                "historical_location"
            )[value_column]
            .sum()
            .rename("historical_value")
        )

    else:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    comparison = pd.concat(
        [
            current,
            historical,
        ],
        axis=1,
    ).fillna(0)

    comparison.index.name = "location"

    comparison = comparison.reset_index()

    comparison.insert(0, "domain", domain)
    comparison.insert(1, "metric", metric)

    comparison["difference"] = (
        comparison["historical_value"]
        - comparison["current_value"]
    )

    comparison["absolute_difference"] = (
        comparison["difference"].abs()
    )

    comparison["relative_difference"] = (
        comparison["difference"]
        / comparison["current_value"].replace(
            0,
            np.nan,
        )
    )

    return comparison


def main() -> None:
    """Run the complete attribution-bias analysis."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    employees = pd.read_csv(
        RAW_DIR / "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    events = pd.read_csv(
        RAW_DIR / "employee_events.csv",
        parse_dates=["event_date"],
    )

    departments = pd.read_csv(
        RAW_DIR / "departments.csv"
    )

    locations = pd.read_csv(
        RAW_DIR / "locations.csv"
    )

    performance = pd.read_csv(
        RAW_DIR / "performance_reviews.csv",
        parse_dates=["review_date"],
    )

    compensation = pd.read_csv(
        RAW_DIR / "compensation_history.csv",
        parse_dates=["effective_date"],
    )

    training = pd.read_csv(
        RAW_DIR / "training_records.csv",
        parse_dates=[
            "start_date",
            "completion_date",
        ],
    )

    department_map = (
        departments.set_index(
            "department_id"
        )["department_name"]
        .to_dict()
    )

    location_map = (
        locations.set_index(
            "location_id"
        )["city"]
        .to_dict()
    )

    # --------------------------------------------------
    # Department-history validation
    # --------------------------------------------------

    hire_events = (
        events.loc[
            events["event_type"].eq("Hire"),
            [
                "employee_id",
                "notes",
            ],
        ]
        .copy()
    )

    hire_events["hire_department_id"] = (
        pd.to_numeric(
            hire_events["notes"].str.extract(
                r"department_id=(\d+)"
            )[0]
        )
    )

    department_validation = (
        employees[
            [
                "employee_id",
                "department_id",
            ]
        ]
        .merge(
            hire_events[
                [
                    "employee_id",
                    "hire_department_id",
                ]
            ],
            on="employee_id",
            how="left",
            validate="one_to_one",
        )
    )

    department_validation[
        "department_mismatch"
    ] = (
        department_validation["department_id"]
        .ne(
            department_validation[
                "hire_department_id"
            ]
        )
    )

    department_validation[
        "department_name"
    ] = (
        department_validation["department_id"]
        .map(department_map)
    )

    department_by_group = (
        department_validation.groupby(
            "department_name"
        )
        .agg(
            employees=("employee_id", "size"),
            department_mismatches=(
                "department_mismatch",
                "sum",
            ),
        )
        .reset_index()
    )

    department_by_group[
        "mismatch_rate"
    ] = (
        department_by_group[
            "department_mismatches"
        ]
        / department_by_group["employees"]
    )

    department_bias_summary = pd.DataFrame(
        {
            "metric": [
                "Employees checked",
                "Employees missing hire department",
                "Hire-versus-current department mismatches",
                "Department-transfer events",
                "Department mismatch rate",
            ],
            "value": [
                len(department_validation),
                int(
                    department_validation[
                        "hire_department_id"
                    ].isna().sum()
                ),
                int(
                    department_validation[
                        "department_mismatch"
                    ].sum()
                ),
                int(
                    events["notes"]
                    .fillna("")
                    .str.contains(
                        "department transfer",
                        case=False,
                    )
                    .sum()
                ),
                float(
                    department_validation[
                        "department_mismatch"
                    ].mean()
                ),
            ],
        }
    )

    # --------------------------------------------------
    # Location-transfer reconstruction
    # --------------------------------------------------

    transfers = (
        events.loc[
            events["event_type"].eq("Transfer")
        ]
        .copy()
    )

    transfers["transfer_date"] = pd.to_datetime(
        transfers["event_date"]
    )

    transfers["old_location_id"] = (
        pd.to_numeric(
            transfers["old_value"]
        ).astype(int)
    )

    transfers["new_location_id"] = (
        pd.to_numeric(
            transfers["new_value"]
        ).astype(int)
    )

    transfer_count_by_employee = (
        transfers.groupby("employee_id")
        .size()
    )

    if transfer_count_by_employee.gt(1).any():
        raise ValueError(
            "This reconstruction expects no more "
            "than one location transfer per employee."
        )

    transfer_validation = transfers.merge(
        employees[
            [
                "employee_id",
                "location_id",
            ]
        ],
        on="employee_id",
        how="left",
        validate="many_to_one",
    )

    if not transfer_validation[
        "new_location_id"
    ].eq(
        transfer_validation["location_id"]
    ).all():
        raise ValueError(
            "A transfer destination does not match "
            "the employee's current location."
        )

    transfers["old_location"] = (
        transfers["old_location_id"]
        .map(location_map)
    )

    transfers["new_location"] = (
        transfers["new_location_id"]
        .map(location_map)
    )

    transfer_flows = (
        transfers.groupby(
            [
                "old_location",
                "new_location",
            ]
        )
        .agg(
            transfer_count=(
                "employee_id",
                "size",
            )
        )
        .reset_index()
        .sort_values(
            "transfer_count",
            ascending=False,
        )
    )

    # At the transfer date, the new location is treated
    # as effective. Records strictly before the date use
    # the old location.
    training["record_date"] = (
        training["completion_date"]
        .fillna(training["start_date"])
    )

    domain_inputs = {
        "Performance reviews": (
            performance,
            "review_date",
        ),
        "Compensation records": (
            compensation,
            "effective_date",
        ),
        "Training records": (
            training,
            "record_date",
        ),
        "Promotion events": (
            events.loc[
                events["event_type"].eq(
                    "Promotion"
                )
            ].copy(),
            "event_date",
        ),
        "Manager-change events": (
            events.loc[
                events["event_type"].eq(
                    "Manager Change"
                )
            ].copy(),
            "event_date",
        ),
        "Leave events": (
            events.loc[
                events["event_type"].eq(
                    "Leave"
                )
            ].copy(),
            "event_date",
        ),
        "Termination events": (
            events.loc[
                events["event_type"].eq(
                    "Termination"
                )
            ].copy(),
            "event_date",
        ),
    }

    attributed_records = {}

    for domain, (
        records,
        date_column,
    ) in domain_inputs.items():
        attributed_records[domain] = (
            assign_historical_location(
                records=records,
                date_column=date_column,
                domain=domain,
                employees=employees,
                transfers=transfers,
                location_map=location_map,
            )
        )

    reassignment_rows = []

    for domain, records in (
        attributed_records.items()
    ):
        reassignment_rows.append(
            {
                "domain": domain,
                "total_records": len(records),
                "reassigned_records": int(
                    records[
                        "location_reassigned"
                    ].sum()
                ),
                "reassigned_employees": int(
                    records.loc[
                        records[
                            "location_reassigned"
                        ],
                        "employee_id",
                    ].nunique()
                ),
                "reassignment_rate": (
                    records[
                        "location_reassigned"
                    ].mean()
                ),
            }
        )

    reassignment_summary = (
        pd.DataFrame(reassignment_rows)
        .sort_values(
            "reassignment_rate",
            ascending=False,
        )
    )

    comparisons = []

    for domain, records in (
        attributed_records.items()
    ):
        comparisons.append(
            compare_location_metric(
                records=records,
                domain=domain,
                metric="record_count",
            )
        )

    comparisons.append(
        compare_location_metric(
            records=attributed_records[
                "Performance reviews"
            ],
            domain="Performance reviews",
            metric="average_performance_rating",
            value_column="performance_rating",
            aggregation="mean",
        )
    )

    comparisons.append(
        compare_location_metric(
            records=attributed_records[
                "Compensation records"
            ],
            domain="Compensation records",
            metric="average_base_salary",
            value_column="base_salary",
            aggregation="mean",
        )
    )

    comparisons.append(
        compare_location_metric(
            records=attributed_records[
                "Training records"
            ],
            domain="Training records",
            metric="total_training_hours",
            value_column="training_hours",
            aggregation="sum",
        )
    )

    location_metric_comparison = pd.concat(
        comparisons,
        ignore_index=True,
    )

    sample_frames = []

    for domain, records in (
        attributed_records.items()
    ):
        changed = records.loc[
            records["location_reassigned"],
            [
                "employee_id",
                "record_date",
                "current_location",
                "historical_location",
            ],
        ].head(20)

        if not changed.empty:
            changed = changed.copy()
            changed.insert(0, "domain", domain)
            sample_frames.append(changed)

    if sample_frames:
        reassigned_samples = pd.concat(
            sample_frames,
            ignore_index=True,
        )
    else:
        reassigned_samples = pd.DataFrame(
            columns=[
                "domain",
                "employee_id",
                "record_date",
                "current_location",
                "historical_location",
            ]
        )

    # --------------------------------------------------
    # Save reproducible outputs
    # --------------------------------------------------

    outputs = {
        "department_bias_summary.csv": (
            department_bias_summary
        ),
        "department_validation_by_group.csv": (
            department_by_group
        ),
        "location_transfer_flows.csv": (
            transfer_flows
        ),
        "location_reassignment_summary.csv": (
            reassignment_summary
        ),
        "location_metric_comparison.csv": (
            location_metric_comparison
        ),
        "reassigned_record_samples.csv": (
            reassigned_samples
        ),
    }

    for filename, table in outputs.items():
        table.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    # --------------------------------------------------
    # Chart 1: reassignment rate by domain
    # --------------------------------------------------

    chart_data = reassignment_summary.sort_values(
        "reassignment_rate"
    ).copy()

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.barh(
        chart_data["domain"],
        chart_data["reassignment_rate"] * 100,
        color="#1565C0",
    )

    axis.set_title(
        "Historical Location Attribution Changes "
        "a Minority of Records"
    )
    axis.set_xlabel("Records reassigned (%)")
    axis.set_ylabel("")
    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR
        / "01_reassignment_rate_by_domain.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    # --------------------------------------------------
    # Chart 2: net count shift by location
    # --------------------------------------------------

    count_comparison = (
        location_metric_comparison.loc[
            location_metric_comparison[
                "metric"
            ].eq("record_count")
        ]
        .groupby("location")
        .agg(
            current_value=(
                "current_value",
                "sum",
            ),
            historical_value=(
                "historical_value",
                "sum",
            ),
        )
        .reset_index()
    )

    positions = np.arange(
        len(count_comparison)
    )

    width = 0.38

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        positions - width / 2,
        count_comparison["current_value"],
        width=width,
        label="Current-location attribution",
        color="#90CAF9",
    )

    axis.bar(
        positions + width / 2,
        count_comparison[
            "historical_value"
        ],
        width=width,
        label="Event-time attribution",
        color="#1565C0",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        count_comparison["location"]
    )

    axis.set_title(
        "Current Locations Slightly Redistribute "
        "Historical Analytical Records"
    )
    axis.set_ylabel("Records across audited domains")
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR
        / "02_current_vs_historical_location.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nDEPARTMENT BIAS SUMMARY")
    print(
        department_bias_summary.to_string(
            index=False
        )
    )

    print("\nDEPARTMENT VALIDATION BY GROUP")
    print(
        department_by_group.to_string(
            index=False
        )
    )

    print("\nLOCATION TRANSFER SUMMARY")
    print(
        pd.DataFrame(
            {
                "metric": [
                    "Transfer records",
                    "Employees with transfers",
                    "Employees with multiple transfers",
                    "Destination/current-location mismatches",
                ],
                "value": [
                    len(transfers),
                    transfers[
                        "employee_id"
                    ].nunique(),
                    int(
                        transfer_count_by_employee
                        .gt(1)
                        .sum()
                    ),
                    int(
                        (
                            ~transfer_validation[
                                "new_location_id"
                            ].eq(
                                transfer_validation[
                                    "location_id"
                                ]
                            )
                        ).sum()
                    ),
                ],
            }
        ).to_string(index=False)
    )

    print("\nLOCATION RECORD REASSIGNMENT")
    print(
        reassignment_summary.to_string(
            index=False
        )
    )

    print("\nLARGEST LOCATION METRIC CHANGES")
    print(
        location_metric_comparison.sort_values(
            "absolute_difference",
            ascending=False,
        )
        .head(25)
        .to_string(index=False)
    )

    print(
        "\nOutputs saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()