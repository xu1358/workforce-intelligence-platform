"""Build the current-state Version 2 retention dashboard data.

Checkpoint 47 replaces the legacy full-snapshot retention view with the
current active workforce plan produced by Checkpoint 46. It does not refit,
retune, or rescore the model. It filters workforce records at the declared
as-of date, reconciles them to the frozen policy output, and writes
dashboard-ready tables plus explicit timeline and governance metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DIR = PROCESSED_DIR / "dashboard"

CONFIG_PATH = PROJECT_ROOT / "config" / "dashboard_current_state.yaml"
EMPLOYEE_PATH = RAW_DIR / "employees.csv"
LOCATION_PATH = RAW_DIR / "locations.csv"
CURRENT_SCORE_PATH = PROCESSED_DIR / "current_retention_policy_scores.csv"
CURRENT_SUMMARY_PATH = PROCESSED_DIR / "current_retention_policy_summary.csv"
CURRENT_GROUP_PATH = (
    PROCESSED_DIR / "current_retention_policy_group_summary.csv"
)
POLICY_DECISION_PATH = PROCESSED_DIR / "retention_policy_decision.csv"
FINAL_METRIC_PATH = PROCESSED_DIR / "retention_final_test_model_metrics.csv"
CALIBRATION_SELECTION_PATH = (
    PROCESSED_DIR / "retention_calibration_selection.csv"
)
MODEL_SELECTION_PATH = PROCESSED_DIR / "model_selection_decision_v2.csv"

RISK_EMPLOYEE_PATH = DASHBOARD_DIR / "retention_risk_employees.csv"
RISK_SUMMARY_PATH = DASHBOARD_DIR / "retention_risk_summary.csv"
POLICY_SUMMARY_PATH = DASHBOARD_DIR / "current_policy_summary.csv"
POLICY_GROUP_PATH = DASHBOARD_DIR / "current_policy_by_department.csv"
MODEL_PERFORMANCE_PATH = DASHBOARD_DIR / "model_performance.csv"
MODEL_SUMMARY_PATH = DASHBOARD_DIR / "model_summary.csv"
METADATA_PATH = DASHBOARD_DIR / "dashboard_metadata.csv"
VALIDATION_PATH = (
    PROCESSED_DIR / "dashboard_current_state_validation.csv"
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML configuration file."""

    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    if not isinstance(values, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping.")

    return values


def require_files(paths: list[Path]) -> None:
    """Fail with one readable message when an input is missing."""

    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 47 input files:\n- "
            + "\n- ".join(missing)
        )


def selected_row(
    frame: pd.DataFrame,
    flag_column: str,
) -> pd.Series:
    """Return the one row selected by a boolean decision column."""

    selected = frame.loc[frame[flag_column].eq(True)].copy()

    if len(selected) != 1:
        raise ValueError(
            f"Expected one selected row in {flag_column}; found "
            f"{len(selected)}."
        )

    return selected.iloc[0]


def active_as_of(
    employees: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return employees active at the explicit dashboard as-of date."""

    frame = employees.copy()
    frame["hire_date"] = pd.to_datetime(
        frame["hire_date"],
        errors="raise",
    )
    frame["termination_date"] = pd.to_datetime(
        frame["termination_date"],
        errors="coerce",
    )

    active_mask = (
        frame["hire_date"].le(as_of_date)
        & (
            frame["termination_date"].isna()
            | frame["termination_date"].gt(as_of_date)
        )
        & frame["employment_status"].eq("Active")
    )

    return frame.loc[active_mask].copy()


def assign_probability_bands(
    probabilities: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.Series, float, float]:
    """Create descriptive score bands for the current active population."""

    band_config = config["probability_bands"]
    labels = band_config["labels"]
    high_threshold = float(
        probabilities.quantile(float(band_config["high_quantile"]))
    )
    medium_threshold = float(
        probabilities.quantile(float(band_config["medium_quantile"]))
    )

    values = np.select(
        [
            probabilities.ge(high_threshold),
            probabilities.ge(medium_threshold),
        ],
        [
            labels["high"],
            labels["medium"],
        ],
        default=labels["low"],
    )

    return (
        pd.Series(values, index=probabilities.index, dtype="object"),
        medium_threshold,
        high_threshold,
    )


def build_retention_risk_data(
    employees: pd.DataFrame,
    locations: pd.DataFrame,
    current_scores: pd.DataFrame,
    current_summary: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build active-only employee, summary, and reconciliation tables."""

    dashboard_config = config["dashboard"]
    expected_as_of = pd.Timestamp(
        dashboard_config["expected_as_of_date"]
    )

    if len(current_summary) != 1:
        raise ValueError("Current policy summary must contain one row.")

    summary_as_of = pd.Timestamp(
        str(current_summary.iloc[0]["snapshot_date"])
    )

    if summary_as_of != expected_as_of:
        raise ValueError(
            "Current policy summary date does not match the configured "
            "dashboard as-of date."
        )

    active = active_as_of(employees, expected_as_of)
    eligible_levels = set(
        config["population"]["eligible_organizational_levels"]
    )
    eligible_active = active.loc[
        active["organizational_level"].isin(eligible_levels)
    ].copy()

    current_scores = current_scores.copy()
    current_scores["snapshot_date"] = current_scores[
        "snapshot_date"
    ].astype(str)

    score_ids = set(current_scores["employee_id"].astype(int))
    eligible_ids = set(eligible_active["employee_id"].astype(int))
    active_ids = set(active["employee_id"].astype(int))
    all_employee_ids = set(employees["employee_id"].astype(int))

    inactive_score_ids = score_ids - active_ids
    missing_eligible_ids = eligible_ids - score_ids
    unexpected_score_ids = score_ids - eligible_ids
    unknown_score_ids = score_ids - all_employee_ids

    if (
        inactive_score_ids
        or missing_eligible_ids
        or unexpected_score_ids
        or unknown_score_ids
    ):
        raise ValueError(
            "Current policy scores do not reconcile to the active, "
            "model-eligible workforce."
        )

    employee_context = eligible_active[
        [
            "employee_id",
            "employment_status",
            "location_id",
            "organizational_level",
            "employment_type",
        ]
    ].merge(
        locations[
            [
                "location_id",
                "city",
                "region",
            ]
        ],
        on="location_id",
        how="left",
        validate="many_to_one",
    )

    risk = current_scores.merge(
        employee_context,
        on="employee_id",
        how="inner",
        suffixes=("", "_employee"),
        validate="one_to_one",
    )

    for column in ["organizational_level", "employment_type"]:
        employee_column = f"{column}_employee"

        if not risk[column].eq(risk[employee_column]).all():
            raise ValueError(
                f"Current score {column} does not match employees.csv."
            )

        risk = risk.drop(columns=[employee_column])

    (
        risk["probability_band"],
        medium_threshold,
        high_threshold,
    ) = assign_probability_bands(
        risk["attrition_probability"],
        config,
    )

    risk["review_status"] = np.where(
        risk["selected_for_human_review"].eq(True),
        "Selected for human review",
        "Not selected",
    )
    risk["as_of_date"] = expected_as_of.date().isoformat()
    risk["probability_label"] = dashboard_config[
        "probability_label"
    ]
    risk["automatic_employment_action_permitted"] = False

    output_columns = [
        "employee_id",
        "as_of_date",
        "snapshot_date",
        "employment_status",
        "department_name",
        "city",
        "region",
        "organizational_level",
        "employment_type",
        "job_level",
        "base_salary",
        "attrition_probability",
        "probability_label",
        "probability_band",
        "replacement_cost_usd",
        "predicted_avoidable_cost_usd",
        "predicted_net_value_usd",
        "expected_value_rank",
        "selected_for_human_review",
        "review_status",
        "automatic_employment_action_permitted",
    ]
    risk = (
        risk[output_columns]
        .sort_values(
            ["selected_for_human_review", "expected_value_rank"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    summary = (
        risk.groupby(
            ["review_status", "probability_band"],
            as_index=False,
        )
        .agg(
            employee_count=("employee_id", "count"),
            average_attrition_probability=(
                "attrition_probability",
                "mean",
            ),
            average_predicted_net_value_usd=(
                "predicted_net_value_usd",
                "mean",
            ),
        )
    )
    summary["population_percent"] = (
        100.0 * summary["employee_count"] / len(risk)
    )
    summary["as_of_date"] = expected_as_of.date().isoformat()

    reconciliation = pd.DataFrame(
        [
            {
                "as_of_date": expected_as_of.date().isoformat(),
                "active_workforce_employees": len(active),
                "model_eligible_active_employees": len(eligible_active),
                "protected_active_employees": (
                    len(active) - len(eligible_active)
                ),
                "dashboard_scored_employees": len(risk),
                "selected_for_human_review": int(
                    risk["selected_for_human_review"].sum()
                ),
                "inactive_employees_in_dashboard": len(
                    inactive_score_ids
                ),
                "missing_eligible_employees": len(missing_eligible_ids),
                "unexpected_scored_employees": len(
                    unexpected_score_ids
                ),
                "medium_probability_threshold": medium_threshold,
                "high_probability_threshold": high_threshold,
            }
        ]
    )

    return risk, summary, reconciliation


def build_model_tables(
    final_metrics: pd.DataFrame,
    model_selection: pd.DataFrame,
    calibration_selection: pd.DataFrame,
    policy_decision: pd.DataFrame,
    current_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact Version 2 model and policy evidence tables."""

    model_row = selected_row(model_selection, "selected_model")
    calibration_row = selected_row(
        calibration_selection,
        "selected_method",
    )

    if len(final_metrics) != 1 or len(policy_decision) != 1:
        raise ValueError(
            "Final metrics and policy decision must each contain one row."
        )

    final_row = final_metrics.iloc[0]
    policy_row = policy_decision.iloc[0]
    current_row = current_summary.iloc[0]

    performance = final_metrics[
        [
            "evaluation_period",
            "rows",
            "positive_cases",
            "positive_rate",
            "mean_predicted_probability",
            "mean_calibration_gap",
            "brier_score",
            "pr_auc",
            "roc_auc",
            "top_decile_count",
            "top_decile_precision",
            "top_decile_capture",
            "top_decile_lift",
        ]
    ].copy()
    performance["evidence_status"] = (
        "Once-only out-of-time final test"
    )

    summary = pd.DataFrame(
        [
            {
                "selected_model": model_row["model"],
                "selected_calibration_method": calibration_row["method"],
                "probability_display_label": calibration_row[
                    "recommended_dashboard_label"
                ],
                "selected_policy": policy_row[
                    "selected_policy_display_name"
                ],
                "validation_snapshot": policy_row[
                    "validation_snapshot"
                ],
                "final_test_snapshot": policy_row["test_snapshot"],
                "final_test_pr_auc": final_row["pr_auc"],
                "final_test_roc_auc": final_row["roc_auc"],
                "final_test_brier_score": final_row["brier_score"],
                "maximum_employees": policy_row["maximum_employees"],
                "budget_usd": policy_row["budget_usd"],
                "current_as_of_date": current_row["snapshot_date"],
                "current_eligible_employees": current_row[
                    "eligible_employees"
                ],
                "current_selected_for_human_review": current_row[
                    "selected_for_human_review"
                ],
            }
        ]
    )

    return performance, summary


def build_metadata(
    config: dict[str, Any],
    policy_decision: pd.DataFrame,
) -> pd.DataFrame:
    """Create the explicit UI timeline and governance messages."""

    dashboard = config["dashboard"]
    governance = config["governance"]
    policy_row = policy_decision.iloc[0]

    return pd.DataFrame(
        [
            {
                "as_of_date": dashboard["expected_as_of_date"],
                "workforce_status_label": dashboard[
                    "workforce_status_label"
                ],
                "probability_label": dashboard["probability_label"],
                "score_horizon_months": dashboard[
                    "score_horizon_months"
                ],
                "model_status_label": dashboard["model_status_label"],
                "final_test_snapshot": policy_row["test_snapshot"],
                "selected_policy": policy_row[
                    "selected_policy_display_name"
                ],
                "synthetic_data_notice": governance[
                    "synthetic_data_notice"
                ],
                "current_score_caveat": governance[
                    "current_score_caveat"
                ],
                "legacy_score_caveat": governance[
                    "legacy_score_caveat"
                ],
                "use_notice": governance["use_notice"],
                "human_review_required": governance[
                    "human_review_required"
                ],
                "automatic_employment_action_permitted": governance[
                    "automatic_employment_action_permitted"
                ],
            }
        ]
    )


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> None:
    """Append one validation result."""

    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def validate_dashboard(
    risk: pd.DataFrame,
    reconciliation: pd.DataFrame,
    metadata: pd.DataFrame,
    model_performance: pd.DataFrame,
    policy_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate timeline, status, model evidence, and safety semantics."""

    checks: list[dict[str, Any]] = []
    expected_as_of = config["dashboard"]["expected_as_of_date"]
    expected_selected = int(
        config["validation"]["expected_selected_for_review"]
    )
    recon = reconciliation.iloc[0]
    metadata_row = metadata.iloc[0]
    summary_row = policy_summary.iloc[0]

    add_check(
        checks,
        "Dashboard as-of date is explicit and exact",
        (
            risk["as_of_date"].astype(str).unique().tolist()
            == [expected_as_of]
            and str(metadata_row["as_of_date"]) == expected_as_of
        ),
        risk["as_of_date"].astype(str).unique().tolist(),
        [expected_as_of],
        "Every employee row and the UI metadata use one current date.",
    )
    add_check(
        checks,
        "Dashboard contains current active employees only",
        (
            risk["employment_status"].eq("Active").all()
            and int(recon["inactive_employees_in_dashboard"]) == 0
        ),
        int(recon["inactive_employees_in_dashboard"]),
        0,
        "Terminated employees cannot appear as current retention risks.",
    )
    add_check(
        checks,
        "Active model-eligible population reconciles",
        (
            int(recon["model_eligible_active_employees"])
            == len(risk)
            == int(summary_row["eligible_employees"])
        ),
        len(risk),
        int(recon["model_eligible_active_employees"]),
        "Current scores must cover all and only eligible active employees.",
    )
    add_check(
        checks,
        "Protected hierarchy employees are excluded",
        int(recon["protected_active_employees"]) > 0,
        int(recon["protected_active_employees"]),
        "> 0 and absent from risk rows",
        "Department heads and senior managers remain in workforce totals.",
    )
    add_check(
        checks,
        "Employee dashboard keys are unique",
        risk["employee_id"].is_unique,
        int(risk["employee_id"].duplicated().sum()),
        0,
        "Each current employee receives one dashboard row.",
    )
    add_check(
        checks,
        "Frozen review capacity is preserved",
        (
            int(risk["selected_for_human_review"].sum())
            == expected_selected
            == int(summary_row["selected_for_human_review"])
        ),
        int(risk["selected_for_human_review"].sum()),
        expected_selected,
        "Checkpoint 47 displays but does not retune the 700-person plan.",
    )
    add_check(
        checks,
        "Current probabilities are finite and bounded",
        (
            np.isfinite(risk["attrition_probability"]).all()
            and risk["attrition_probability"].between(0.0, 1.0).all()
        ),
        (
            f"min={risk['attrition_probability'].min():.6f}; "
            f"max={risk['attrition_probability'].max():.6f}"
        ),
        "All finite and inside [0, 1]",
        "The approved calibrated probability scale is preserved.",
    )
    target_columns = {
        "attrition_next_12m",
        "actual_attrition",
        "termination_date",
        "termination_type",
    }
    present_targets = sorted(target_columns & set(risk.columns))
    add_check(
        checks,
        "No future outcome columns are displayed",
        not present_targets,
        present_targets,
        [],
        "Current outcomes after the as-of date are unknown.",
    )
    identifying_columns = {
        "first_name",
        "last_name",
        "full_name",
        "email",
    }
    present_identifiers = sorted(identifying_columns & set(risk.columns))
    add_check(
        checks,
        "Direct personal names are omitted",
        not present_identifiers,
        present_identifiers,
        [],
        "The synthetic review table remains anonymized by employee ID.",
    )
    add_check(
        checks,
        "Out-of-time model evidence is retained",
        (
            len(model_performance) == 1
            and model_performance["evidence_status"].eq(
                "Once-only out-of-time final test"
            ).all()
        ),
        model_performance["evaluation_period"].tolist(),
        ["2025 final test"],
        "Performance metrics are not calculated from current unknown outcomes.",
    )
    add_check(
        checks,
        "Legacy in-sample score caveat is visible metadata",
        bool(str(metadata_row["legacy_score_caveat"]).strip()),
        "Present",
        "Present",
        "The UI distinguishes legacy prioritization from current scoring.",
    )
    add_check(
        checks,
        "Current projection caveat is visible metadata",
        bool(str(metadata_row["current_score_caveat"]).strip()),
        "Present",
        "Present",
        "The UI states that current rows are not a new performance test.",
    )
    add_check(
        checks,
        "Human review is required",
        bool(metadata_row["human_review_required"]),
        bool(metadata_row["human_review_required"]),
        True,
        "Risk scores support review rather than automatic action.",
    )
    add_check(
        checks,
        "Automatic employment action is prohibited",
        not bool(
            metadata_row["automatic_employment_action_permitted"]
        ),
        bool(
            metadata_row["automatic_employment_action_permitted"]
        ),
        False,
        "The dashboard cannot authorize an employment decision.",
    )

    validation = pd.DataFrame(checks)
    failed = validation.loc[validation["status"].eq("FAIL")]

    if not failed.empty:
        raise ValueError(
            "Checkpoint 47 validation failed:\n"
            + failed[["check", "observed", "requirement"]].to_string(
                index=False
            )
        )

    return validation


def main() -> None:
    """Build and validate all Checkpoint 47 dashboard tables."""

    required = [
        CONFIG_PATH,
        EMPLOYEE_PATH,
        LOCATION_PATH,
        CURRENT_SCORE_PATH,
        CURRENT_SUMMARY_PATH,
        CURRENT_GROUP_PATH,
        POLICY_DECISION_PATH,
        FINAL_METRIC_PATH,
        CALIBRATION_SELECTION_PATH,
        MODEL_SELECTION_PATH,
    ]
    require_files(required)
    config = load_yaml(CONFIG_PATH)

    employees = pd.read_csv(EMPLOYEE_PATH)
    locations = pd.read_csv(LOCATION_PATH)
    current_scores = pd.read_csv(CURRENT_SCORE_PATH)
    current_summary = pd.read_csv(CURRENT_SUMMARY_PATH)
    current_group = pd.read_csv(CURRENT_GROUP_PATH)
    policy_decision = pd.read_csv(POLICY_DECISION_PATH)
    final_metrics = pd.read_csv(FINAL_METRIC_PATH)
    calibration_selection = pd.read_csv(
        CALIBRATION_SELECTION_PATH
    )
    model_selection = pd.read_csv(MODEL_SELECTION_PATH)

    risk, risk_summary, reconciliation = (
        build_retention_risk_data(
            employees,
            locations,
            current_scores,
            current_summary,
            config,
        )
    )
    model_performance, model_summary = build_model_tables(
        final_metrics,
        model_selection,
        calibration_selection,
        policy_decision,
        current_summary,
    )
    metadata = build_metadata(config, policy_decision)

    current_group = current_group.copy()
    current_group["as_of_date"] = config["dashboard"][
        "expected_as_of_date"
    ]
    current_group["selection_rate_percent"] = (
        100.0 * current_group["selection_rate"]
    )

    validation = validate_dashboard(
        risk,
        reconciliation,
        metadata,
        model_performance,
        current_summary,
        config,
    )

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    risk.to_csv(RISK_EMPLOYEE_PATH, index=False)
    risk_summary.to_csv(RISK_SUMMARY_PATH, index=False)
    current_summary.to_csv(POLICY_SUMMARY_PATH, index=False)
    current_group.to_csv(POLICY_GROUP_PATH, index=False)
    model_performance.to_csv(MODEL_PERFORMANCE_PATH, index=False)
    model_summary.to_csv(MODEL_SUMMARY_PATH, index=False)
    metadata.to_csv(METADATA_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)

    print("\nCURRENT DASHBOARD POPULATION")
    print(reconciliation.to_string(index=False))
    print("\nCURRENT HUMAN-REVIEW PLAN")
    print(current_summary.to_string(index=False))
    print("\nDASHBOARD TIMELINE AND SAFETY VALIDATION")
    print(validation.to_string(index=False))
    print(
        "\nSaved current dashboard data to: "
        f"{DASHBOARD_DIR}"
    )
    print(
        "\nCURRENT-STATE DASHBOARD DATA COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
