from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DASHBOARD_DIR = (
    PROCESSED_DIR
    / "dashboard"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)


# ---------------------------------------------------------
# Expected files
# ---------------------------------------------------------

EXPECTED_RAW_FILES = [
    "departments.csv",
    "locations.csv",
    "job_roles.csv",
    "training_programs.csv",
    "employees.csv",
    "compensation_history.csv",
    "performance_reviews.csv",
    "training_records.csv",
    "employee_events.csv",
    "candidates.csv",
    "job_requisitions.csv",
    "applications.csv",
]


EXPECTED_PROCESSED_FILES = [
    "retention_modeling_dataset.csv",
    "baseline_retention_predictions.csv",
    "retention_model_comparison.csv",
    "selected_retention_predictions.csv",
    "retention_feature_importance.csv",
    "retention_feature_group_importance.csv",
    "retention_threshold_analysis.csv",
    "retention_risk_segments.csv",
]


EXPECTED_DASHBOARD_FILES = [
    "overview_kpis.csv",
    "headcount_by_department.csv",
    "headcount_by_location.csv",
    "recruiting_funnel.csv",
    "requisition_metrics.csv",
    "retention_risk_summary.csv",
    "retention_risk_employees.csv",
    "model_performance.csv",
    "model_summary.csv",
]


EXPECTED_MODEL_FILES = [
    "baseline_retention_model.joblib",
    "selected_retention_model.joblib",
]


# ---------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------

def validate_file_group(
    directory: Path,
    filenames: list[str],
    group_name: str,
) -> list[dict]:
    """Check that expected files exist and are not empty."""

    results = []

    for filename in filenames:

        path = (
            directory
            / filename
        )

        exists = (
            path.exists()
        )

        nonempty = (
            exists
            and path.stat().st_size > 0
        )

        results.append(
            {
                "check": (
                    f"{group_name}: {filename}"
                ),
                "passed": (
                    exists
                    and nonempty
                ),
                "details": (
                    str(path)
                ),
            }
        )

    return results


def add_check(
    results: list[dict],
    name: str,
    condition: bool,
    details: str,
) -> None:
    """Add one logical validation result."""

    results.append(
        {
            "check": name,
            "passed": bool(
                condition
            ),
            "details": details,
        }
    )


# ---------------------------------------------------------
# Main validation
# ---------------------------------------------------------

def main() -> None:
    """Validate final end-to-end project outputs."""

    results = []

    # -----------------------------------------------------
    # File existence
    # -----------------------------------------------------

    results.extend(
        validate_file_group(
            RAW_DIR,
            EXPECTED_RAW_FILES,
            "Raw data",
        )
    )

    results.extend(
        validate_file_group(
            PROCESSED_DIR,
            EXPECTED_PROCESSED_FILES,
            "Processed data",
        )
    )

    results.extend(
        validate_file_group(
            DASHBOARD_DIR,
            EXPECTED_DASHBOARD_FILES,
            "Dashboard data",
        )
    )

    results.extend(
        validate_file_group(
            MODEL_DIR,
            EXPECTED_MODEL_FILES,
            "Model artifact",
        )
    )

    # -----------------------------------------------------
    # Load important datasets
    # -----------------------------------------------------

    employees = pd.read_csv(
        RAW_DIR
        / "employees.csv"
    )

    candidates = pd.read_csv(
        RAW_DIR
        / "candidates.csv"
    )

    applications = pd.read_csv(
        RAW_DIR
        / "applications.csv"
    )

    retention = pd.read_csv(
        PROCESSED_DIR
        / "retention_modeling_dataset.csv"
    )

    model_comparison = pd.read_csv(
        PROCESSED_DIR
        / "retention_model_comparison.csv"
    )

    overview = pd.read_csv(
        DASHBOARD_DIR
        / "overview_kpis.csv"
    )

    department = pd.read_csv(
        DASHBOARD_DIR
        / "headcount_by_department.csv"
    )

    location = pd.read_csv(
        DASHBOARD_DIR
        / "headcount_by_location.csv"
    )

    risk_employees = pd.read_csv(
        DASHBOARD_DIR
        / "retention_risk_employees.csv"
    )

    model_performance = pd.read_csv(
        DASHBOARD_DIR
        / "model_performance.csv"
    )

    model_summary = pd.read_csv(
        DASHBOARD_DIR
        / "model_summary.csv"
    )

    # -----------------------------------------------------
    # Raw data checks
    # -----------------------------------------------------

    add_check(
        results,
        "Employees contain 10,000 rows",
        len(employees) == 10_000,
        f"Actual rows: {len(employees):,}",
    )

    add_check(
        results,
        "Employee IDs are unique",
        employees[
            "employee_id"
        ].is_unique,
        (
            "Unique IDs: "
            f"{employees['employee_id'].nunique():,}"
        ),
    )

    add_check(
        results,
        "Candidates contain 40,000 rows",
        len(candidates) == 40_000,
        f"Actual rows: {len(candidates):,}",
    )

    add_check(
        results,
        "Candidate IDs are unique",
        candidates[
            "candidate_id"
        ].is_unique,
        (
            "Unique IDs: "
            f"{candidates['candidate_id'].nunique():,}"
        ),
    )

    hired_count = int(
        (
            applications[
                "application_status"
            ]
            == "Hired"
        )
        .sum()
    )

    add_check(
        results,
        "Applications contain 10,000 hires",
        hired_count == 10_000,
        f"Actual hires: {hired_count:,}",
    )

    hired_employee_count = (
        applications.loc[
            applications[
                "application_status"
            ]
            == "Hired",
            "employee_id",
        ]
        .nunique()
    )

    add_check(
        results,
        "All employees map to a hired application",
        hired_employee_count == 10_000,
        (
            "Employees mapped to hires: "
            f"{hired_employee_count:,}"
        ),
    )

    # -----------------------------------------------------
    # Retention dataset checks
    # -----------------------------------------------------

    add_check(
        results,
        "Retention dataset is not empty",
        len(retention) > 0,
        f"Rows: {len(retention):,}",
    )

    target_values = set(
        retention[
            "attrition_next_12m"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    add_check(
        results,
        "Retention target contains both classes",
        target_values == {
            0,
            1,
        },
        f"Classes: {sorted(target_values)}",
    )

    add_check(
        results,
        "Retention employee IDs are unique",
        retention[
            "employee_id"
        ].is_unique,
        (
            "Unique IDs: "
            f"{retention['employee_id'].nunique():,}"
        ),
    )

    # -----------------------------------------------------
    # Model checks
    # -----------------------------------------------------

    add_check(
        results,
        "Three models were compared",
        len(model_comparison) == 3,
        (
            "Models: "
            + ", ".join(
                model_comparison[
                    "model"
                ].astype(str)
            )
        ),
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    valid_model_metrics = (
        model_comparison[
            metric_columns
        ]
        .apply(
            lambda column: column.between(
                0,
                1,
            )
        )
        .all()
        .all()
    )

    add_check(
        results,
        "Model metrics are between zero and one",
        valid_model_metrics,
        (
            "Checked: "
            + ", ".join(
                metric_columns
            )
        ),
    )

    selected_flags = (
        model_performance[
            "selected_model"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "true"
        )
    )

    selected_model_count = int(
        selected_flags.sum()
    )

    add_check(
        results,
        "Exactly one model is selected",
        selected_model_count == 1,
        (
            "Selected model rows: "
            f"{selected_model_count}"
        ),
    )

    add_check(
        results,
        "Model summary contains one row",
        len(model_summary) == 1,
        f"Rows: {len(model_summary)}",
    )

    if len(model_summary) == 1:

        selected_model_name = str(
            model_summary.loc[
                0,
                "selected_model",
            ]
        )

        add_check(
            results,
            "Selected model appears in comparison",
            selected_model_name
            in model_comparison[
                "model"
            ].astype(str).values,
            (
                "Selected model: "
                f"{selected_model_name}"
            ),
        )

    # -----------------------------------------------------
    # Dashboard checks
    # -----------------------------------------------------

    add_check(
        results,
        "Overview contains one row",
        len(overview) == 1,
        f"Rows: {len(overview)}",
    )

    if len(overview) == 1:

        active_headcount = int(
            overview.loc[
                0,
                "active_headcount",
            ]
        )

        department_total = int(
            department[
                "active_headcount"
            ]
            .sum()
        )

        location_total = int(
            location[
                "active_headcount"
            ]
            .sum()
        )

        add_check(
            results,
            "Department headcount matches overview",
            department_total
            == active_headcount,
            (
                f"Department: {department_total:,}; "
                f"overview: {active_headcount:,}"
            ),
        )

        add_check(
            results,
            "Location headcount matches overview",
            location_total
            == active_headcount,
            (
                f"Location: {location_total:,}; "
                f"overview: {active_headcount:,}"
            ),
        )

        snapshot_population = int(
            overview.loc[
                0,
                "retention_snapshot_population",
            ]
        )

        add_check(
            results,
            "Dashboard risk population matches retention dataset",
            snapshot_population
            == len(retention)
            == len(risk_employees),
            (
                f"Overview: {snapshot_population:,}; "
                f"retention: {len(retention):,}; "
                f"risk rows: {len(risk_employees):,}"
            ),
        )

    risk_segments = set(
        risk_employees[
            "risk_segment"
        ]
        .dropna()
        .unique()
    )

    add_check(
        results,
        "Dashboard contains three risk segments",
        risk_segments
        == {
            "Low",
            "Medium",
            "High",
        },
        (
            "Segments: "
            f"{sorted(risk_segments)}"
        ),
    )

    valid_probabilities = (
        risk_employees[
            "attrition_probability"
        ]
        .between(
            0,
            1,
        )
        .all()
    )

    add_check(
        results,
        "Dashboard probabilities are valid",
        valid_probabilities,
        (
            "Minimum: "
            f"{risk_employees['attrition_probability'].min():.4f}; "
            "maximum: "
            f"{risk_employees['attrition_probability'].max():.4f}"
        ),
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    results_frame = pd.DataFrame(
        results
    )

    print(
        "\nFinal project validation results:\n"
    )

    print(
        results_frame[
            [
                "check",
                "passed",
                "details",
            ]
        ]
        .to_string(
            index=False
        )
    )

    failed_checks = (
        results_frame[
            ~results_frame[
                "passed"
            ]
        ]
    )

    if not failed_checks.empty:

        raise ValueError(
            "\nOne or more final project "
            "validation checks failed."
        )

    print(
        "\nAll final project validation "
        "checks passed."
    )


if __name__ == "__main__":
    main()