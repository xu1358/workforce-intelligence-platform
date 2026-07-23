"""Audit the Version 1 synthetic termination-generation process."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audit"
AS_OF_DATE = pd.Timestamp("2026-06-30")


def find_name_column(
    table: pd.DataFrame,
    id_column: str,
) -> str | None:
    """Find a descriptive name column in a reference table."""

    candidates = [
        column
        for column in table.columns
        if column != id_column
        and "name" in column.lower()
    ]

    if candidates:
        return candidates[0]

    return None


def create_group_summary(
    data: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Summarize observed and generator-defined attrition."""

    summary = (
        data.groupby(
            group_column,
            observed=True,
            dropna=False,
        )
        .agg(
            headcount=("employee_id", "size"),
            terminations=("terminated", "sum"),
            observed_attrition_rate=(
                "terminated",
                "mean",
            ),
            expected_attrition_rate=(
                "generator_probability",
                "mean",
            ),
            individual_contributor_share=(
                "is_individual_contributor",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["rate_difference"] = (
        summary["observed_attrition_rate"]
        - summary["expected_attrition_rate"]
    )

    return summary.sort_values(
        "observed_attrition_rate",
        ascending=False,
    )


def main() -> None:
    """Run the complete generator audit."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    employees = pd.read_csv(
        RAW_DATA_DIR / "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    departments = pd.read_csv(
        RAW_DATA_DIR / "departments.csv"
    )

    locations = pd.read_csv(
        RAW_DATA_DIR / "locations.csv"
    )

    events = pd.read_csv(
        RAW_DATA_DIR / "employee_events.csv",
        parse_dates=["event_date"],
    )

    compensation = pd.read_csv(
        RAW_DATA_DIR / "compensation_history.csv",
        parse_dates=["effective_date"],
    )

    performance = pd.read_csv(
        RAW_DATA_DIR / "performance_reviews.csv",
        parse_dates=["review_date"],
    )

    training = pd.read_csv(
        RAW_DATA_DIR / "training_records.csv",
        parse_dates=[
            "start_date",
            "completion_date",
        ],
    )

    employees["terminated"] = (
        employees["employment_status"]
        .eq("Terminated")
    )

    employees["is_individual_contributor"] = (
        employees["organizational_level"]
        .eq("Individual Contributor")
    )

    employees["days_since_hire_at_as_of"] = (
        AS_OF_DATE - employees["hire_date"]
    ).dt.days

    employees["tenure_rule_band"] = pd.cut(
        employees["days_since_hire_at_as_of"],
        bins=[
            -np.inf,
            119,
            364,
            np.inf,
        ],
        labels=[
            "0–119 days: forced active",
            "120–364 days: reduced probability",
            "365+ days: full probability",
        ],
    )

    # Reproduce the probability rules in
    # generate_full_workforce.py.
    employees["generator_probability"] = 0.0

    eligible = (
        employees["is_individual_contributor"]
        & employees["days_since_hire_at_as_of"].ge(120)
    )

    probability = pd.Series(
        0.18,
        index=employees.index,
        dtype=float,
    )

    probability += (
        employees["department_id"]
        .isin([2, 8])
        .astype(float)
        * 0.06
    )

    probability += (
        employees["location_id"]
        .eq(3)
        .astype(float)
        * 0.03
    )

    probability -= (
        employees["days_since_hire_at_as_of"]
        .lt(365)
        .astype(float)
        * 0.05
    )

    probability = probability.clip(
        lower=0.05,
        upper=0.40,
    )

    employees.loc[
        eligible,
        "generator_probability",
    ] = probability.loc[eligible]

    department_name_column = find_name_column(
        departments,
        "department_id",
    )

    if department_name_column is not None:
        employees = employees.merge(
            departments[
                [
                    "department_id",
                    department_name_column,
                ]
            ],
            on="department_id",
            how="left",
        )

        employees = employees.rename(
            columns={
                department_name_column: (
                    "department_name"
                )
            }
        )
    else:
        employees["department_name"] = (
            employees["department_id"]
            .astype(str)
        )

    location_name_column = (
        "city"
        if "city" in locations.columns
        else find_name_column(
            locations,
            "location_id",
        )
    )

    if location_name_column is not None:
        employees = employees.merge(
            locations[
                [
                    "location_id",
                    location_name_column,
                ]
            ],
            on="location_id",
            how="left",
        )

        employees = employees.rename(
            columns={
                location_name_column: "location_name"
            }
        )
    else:
        employees["location_name"] = (
            employees["location_id"]
            .astype(str)
        )

    overall_summary = pd.DataFrame(
        {
            "metric": [
                "Total employees",
                "Terminated employees",
                "Observed attrition rate",
                "Expected terminations from generator",
                "Expected attrition rate",
                "Individual contributors",
                "Permanently active leaders",
                "Individual contributors under 120 days",
                "Voluntary terminations",
                "Involuntary terminations",
            ],
            "value": [
                len(employees),
                int(employees["terminated"].sum()),
                employees["terminated"].mean(),
                employees[
                    "generator_probability"
                ].sum(),
                employees[
                    "generator_probability"
                ].mean(),
                int(
                    employees[
                        "is_individual_contributor"
                    ].sum()
                ),
                int(
                    (
                        ~employees[
                            "is_individual_contributor"
                        ]
                    ).sum()
                ),
                int(
                    (
                        employees[
                            "is_individual_contributor"
                        ]
                        & employees[
                            "days_since_hire_at_as_of"
                        ].lt(120)
                    ).sum()
                ),
                int(
                    employees[
                        "termination_type"
                    ]
                    .eq("Voluntary")
                    .sum()
                ),
                int(
                    employees[
                        "termination_type"
                    ]
                    .eq("Involuntary")
                    .sum()
                ),
            ],
        }
    )

    organizational_summary = create_group_summary(
        employees,
        "organizational_level",
    )

    tenure_summary = create_group_summary(
        employees,
        "tenure_rule_band",
    )

    department_summary = create_group_summary(
        employees,
        "department_name",
    )

    location_summary = create_group_summary(
        employees,
        "location_name",
    )

    terminated = employees.loc[
        employees["terminated"]
    ].copy()

    terminated["termination_month"] = (
        terminated["termination_date"]
        .dt.to_period("M")
        .astype(str)
    )

    termination_month_summary = (
        terminated.groupby(
            "termination_month",
            dropna=False,
        )
        .agg(
            terminations=("employee_id", "size")
        )
        .reset_index()
        .sort_values("termination_month")
    )

    performance = performance.sort_values(
        [
            "employee_id",
            "review_date",
        ]
    )

    performance_features = (
        performance.groupby("employee_id")
        .agg(
            review_count=(
                "performance_rating",
                "size",
            ),
            average_performance_rating=(
                "performance_rating",
                "mean",
            ),
            latest_performance_rating=(
                "performance_rating",
                "last",
            ),
        )
        .reset_index()
    )

    compensation = compensation.sort_values(
        [
            "employee_id",
            "effective_date",
        ]
    )

    compensation_features = (
        compensation.groupby("employee_id")
        .agg(
            first_base_salary=(
                "base_salary",
                "first",
            ),
            latest_base_salary=(
                "base_salary",
                "last",
            ),
            compensation_record_count=(
                "base_salary",
                "size",
            ),
        )
        .reset_index()
    )

    compensation_features[
        "compensation_growth_rate"
    ] = (
        compensation_features[
            "latest_base_salary"
        ]
        - compensation_features[
            "first_base_salary"
        ]
    ) / compensation_features[
        "first_base_salary"
    ]

    training["completed_training"] = (
        training["completion_status"]
        .eq("Completed")
        .astype(int)
    )

    training_features = (
        training.groupby("employee_id")
        .agg(
            training_record_count=(
                "training_record_id",
                "size",
            ),
            completed_training_count=(
                "completed_training",
                "sum",
            ),
            total_training_hours=(
                "training_hours",
                "sum",
            ),
        )
        .reset_index()
    )

    events["event_type_lower"] = (
        events["event_type"]
        .fillna("")
        .str.lower()
    )

    events["promotion_event"] = (
        events["event_type_lower"]
        .str.contains("promotion")
        .astype(int)
    )

    events["manager_change_event"] = (
        events["event_type_lower"]
        .str.contains("manager")
        .astype(int)
    )

    events["transfer_event"] = (
        events["event_type_lower"]
        .str.contains("transfer")
        .astype(int)
    )

    event_features = (
        events.groupby("employee_id")
        .agg(
            event_count=("event_id", "size"),
            promotion_event_count=(
                "promotion_event",
                "sum",
            ),
            manager_change_event_count=(
                "manager_change_event",
                "sum",
            ),
            transfer_event_count=(
                "transfer_event",
                "sum",
            ),
        )
        .reset_index()
    )

    behavioral = employees[
        [
            "employee_id",
            "terminated",
            "days_since_hire_at_as_of",
        ]
    ].copy()

    for feature_table in [
        performance_features,
        compensation_features,
        training_features,
        event_features,
    ]:
        behavioral = behavioral.merge(
            feature_table,
            on="employee_id",
            how="left",
        )

    count_columns = [
        "review_count",
        "compensation_record_count",
        "training_record_count",
        "completed_training_count",
        "total_training_hours",
        "event_count",
        "promotion_event_count",
        "manager_change_event_count",
        "transfer_event_count",
    ]

    behavioral[count_columns] = (
        behavioral[count_columns].fillna(0)
    )

    behavioral_summary = (
        behavioral.groupby("terminated")
        .agg(
            employees=("employee_id", "size"),
            average_days_since_hire=(
                "days_since_hire_at_as_of",
                "mean",
            ),
            average_performance_rating=(
                "average_performance_rating",
                "mean",
            ),
            average_latest_performance_rating=(
                "latest_performance_rating",
                "mean",
            ),
            average_compensation_growth=(
                "compensation_growth_rate",
                "mean",
            ),
            average_completed_training_count=(
                "completed_training_count",
                "mean",
            ),
            average_training_hours=(
                "total_training_hours",
                "mean",
            ),
            average_promotion_events=(
                "promotion_event_count",
                "mean",
            ),
            average_manager_change_events=(
                "manager_change_event_count",
                "mean",
            ),
            average_transfer_events=(
                "transfer_event_count",
                "mean",
            ),
        )
        .reset_index()
    )

    behavioral_summary["group"] = (
        behavioral_summary["terminated"].map(
            {
                False: "Active",
                True: "Terminated",
            }
        )
    )

    behavioral_summary = behavioral_summary.drop(
        columns="terminated"
    )

    generator_rules = pd.DataFrame(
        {
            "factor": [
                "Organizational level",
                "Days since hire below 120",
                "Base probability",
                "Department 2 or 8",
                "Location 3",
                "Days since hire from 120 to 364",
                "Behavioral history",
                "Termination date",
                "Termination type",
            ],
            "current_rule": [
                "Only individual contributors are eligible",
                "Forced active",
                "18 percentage points",
                "Add 6 percentage points",
                "Add 3 percentage points",
                "Subtract 5 percentage points",
                "Not used by choose_termination()",
                "Uniform draw from hire date + 60 days to as-of date",
                "75% voluntary and 25% involuntary",
            ],
        }
    )

    outputs = {
        "generator_audit_summary.csv": overall_summary,
        "attrition_by_organizational_level.csv": (
            organizational_summary
        ),
        "attrition_by_tenure_rule.csv": tenure_summary,
        "attrition_by_department.csv": (
            department_summary
        ),
        "attrition_by_location.csv": location_summary,
        "termination_by_month.csv": (
            termination_month_summary
        ),
        "behavioral_associations.csv": (
            behavioral_summary
        ),
        "generator_defined_rules.csv": generator_rules,
    }

    for filename, table in outputs.items():
        table.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    print("\nGENERATOR AUDIT SUMMARY")
    print(overall_summary.to_string(index=False))

    print("\nATTRITION BY ORGANIZATIONAL LEVEL")
    print(
        organizational_summary.to_string(
            index=False
        )
    )

    print("\nATTRITION BY TENURE RULE")
    print(tenure_summary.to_string(index=False))

    print("\nATTRITION BY DEPARTMENT")
    print(
        department_summary.to_string(
            index=False
        )
    )

    print("\nATTRITION BY LOCATION")
    print(location_summary.to_string(index=False))

    print("\nBEHAVIORAL ASSOCIATIONS")
    print(behavioral_summary.to_string(index=False))

    print(
        "\nAudit files saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()