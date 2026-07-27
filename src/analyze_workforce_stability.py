"""Translate the frozen retention policy into workforce-planning scenarios.

Checkpoint 48 reframes current Version 2 scores around manufacturing
workforce stability and backfill planning. It does not retrain the model,
change probabilities, select a new policy, or inspect future outcomes.
Probability-weighted departures and operational units are scenario estimates,
not exact staffing forecasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DIR = PROCESSED_DIR / "dashboard"

CONFIG_PATH = PROJECT_ROOT / "config" / "workforce_stability.yaml"
CURRENT_POPULATION_PATH = (
    PROCESSED_DIR / "current_active_scoring_population.csv"
)
CURRENT_SCORE_PATH = (
    PROCESSED_DIR / "current_retention_policy_scores.csv"
)
CURRENT_POLICY_SUMMARY_PATH = (
    PROCESSED_DIR / "current_retention_policy_summary.csv"
)
POLICY_DECISION_PATH = (
    PROCESSED_DIR / "retention_policy_decision.csv"
)

DEPARTMENT_OUTPUT_PATH = (
    PROCESSED_DIR / "workforce_stability_by_department.csv"
)
FOCUS_ROLE_OUTPUT_PATH = (
    PROCESSED_DIR / "manufacturing_backfill_by_role.csv"
)
SUMMARY_OUTPUT_PATH = (
    PROCESSED_DIR / "manufacturing_stability_summary.csv"
)
VALIDATION_OUTPUT_PATH = (
    PROCESSED_DIR / "workforce_stability_validation.csv"
)

DASHBOARD_DEPARTMENT_PATH = (
    DASHBOARD_DIR / "workforce_stability_by_department.csv"
)
DASHBOARD_ROLE_PATH = (
    DASHBOARD_DIR / "manufacturing_backfill_by_role.csv"
)
DASHBOARD_SUMMARY_PATH = (
    DASHBOARD_DIR / "manufacturing_stability_summary.csv"
)
DASHBOARD_METADATA_PATH = (
    DASHBOARD_DIR / "workforce_stability_metadata.csv"
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""

    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    if not isinstance(values, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping.")

    return values


def require_files(paths: list[Path]) -> None:
    """Raise one readable error when required inputs are missing."""

    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 48 input files:\n- "
            + "\n- ".join(missing)
        )


def prepare_population(
    current_population: pd.DataFrame,
    scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile the current active population with frozen policy scores."""

    analysis = config["analysis"]
    expected_date = str(analysis["expected_as_of_date"])
    eligible_levels = set(
        config["population"]["eligible_organizational_levels"]
    )

    population = current_population.copy()
    population["snapshot_date"] = population["snapshot_date"].astype(str)
    scores = scores.copy()
    scores["snapshot_date"] = scores["snapshot_date"].astype(str)

    if population["snapshot_date"].unique().tolist() != [expected_date]:
        raise ValueError(
            "Current population does not use the configured as-of date."
        )

    if scores["snapshot_date"].unique().tolist() != [expected_date]:
        raise ValueError(
            "Current scores do not use the configured as-of date."
        )

    eligible_population = population.loc[
        population["organizational_level"].isin(eligible_levels)
    ].copy()

    eligible_ids = set(
        eligible_population["employee_id"].astype(int)
    )
    score_ids = set(scores["employee_id"].astype(int))

    if eligible_ids != score_ids:
        raise ValueError(
            "Current policy scores do not exactly cover the model-eligible "
            "active population."
        )

    context_columns = [
        "employee_id",
        "snapshot_date",
        "department_name",
        "department_group",
        "job_title",
        "job_family",
        "organizational_level",
        "employment_type",
        "job_level",
        "base_salary",
    ]
    eligible_context = eligible_population[context_columns].copy()
    score_columns = [
        "employee_id",
        "snapshot_date",
        "department_name",
        "organizational_level",
        "employment_type",
        "job_level",
        "base_salary",
        "attrition_probability",
        "replacement_cost_usd",
        "predicted_avoidable_cost_usd",
        "predicted_net_value_usd",
        "expected_value_rank",
        "selected_for_human_review",
    ]
    scores = scores[score_columns].copy()

    frame = eligible_context.merge(
        scores,
        on=["employee_id", "snapshot_date"],
        how="inner",
        suffixes=("_context", ""),
        validate="one_to_one",
    )

    for column in [
        "department_name",
        "organizational_level",
        "employment_type",
        "job_level",
    ]:
        context_column = f"{column}_context"

        if not frame[column].eq(frame[context_column]).all():
            raise ValueError(
                f"Current score {column} does not match the source context."
            )

        frame = frame.drop(columns=[context_column])

    salary_difference = (
        pd.to_numeric(
            frame["base_salary"],
            errors="raise",
        )
        - pd.to_numeric(
            frame["base_salary_context"],
            errors="raise",
        )
    ).abs().max()

    if float(salary_difference) > 0.000001:
        raise ValueError(
            "Current score salary does not match the source context."
        )

    frame = frame.drop(columns=["base_salary_context"])
    return population, frame


def policy_assumptions(
    policy_decision: pd.DataFrame,
    policy_summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float | int | str | bool]:
    """Extract the committed planning assumptions."""

    if len(policy_decision) != 1 or len(policy_summary) != 1:
        raise ValueError(
            "Policy decision and current summary must each have one row."
        )

    decision = policy_decision.iloc[0]
    summary = policy_summary.iloc[0]
    expected_date = str(config["analysis"]["expected_as_of_date"])

    if str(summary["snapshot_date"]) != expected_date:
        raise ValueError(
            "Current policy summary date does not match Checkpoint 48."
        )

    assumptions: dict[str, float | int | str | bool] = {
        "as_of_date": expected_date,
        "policy_name": str(
            decision["selected_policy_display_name"]
        ),
        "policy_frozen_before_test": bool(
            decision["policy_frozen_before_test"]
        ),
        "test_evaluated_once": bool(
            decision["test_evaluated_once"]
        ),
        "maximum_employees": int(decision["maximum_employees"]),
        "budget_usd": float(decision["budget_usd"]),
        "intervention_cost_usd": float(
            decision["intervention_cost_usd"]
        ),
        "success_probability": float(
            decision["success_probability"]
        ),
        "replacement_multiplier": float(
            decision["replacement_cost_salary_multiplier"]
        ),
        "vacancy_days": float(decision["vacancy_days"]),
        "time_to_productivity_days": float(
            decision["time_to_productivity_days"]
        ),
        "training_hours": float(
            decision["training_hours_per_replacement"]
        ),
        "coverage_hours": float(
            decision["coverage_hours_per_replacement"]
        ),
        "eligible_employees": int(summary["eligible_employees"]),
        "selected_employees": int(
            summary["selected_for_human_review"]
        ),
        "projected_prevented_departures": float(
            summary["projected_expected_prevented_departures"]
        ),
        "projected_spend_usd": float(
            summary["projected_intervention_spend_usd"]
        ),
        "projected_avoided_cost_usd": float(
            summary["projected_expected_avoided_cost_usd"]
        ),
        "projected_net_value_usd": float(
            summary["projected_expected_net_value_usd"]
        ),
        "human_review_required": bool(
            summary["human_review_required"]
        ),
        "automatic_action_permitted": bool(
            summary["automatic_employment_action_permitted"]
        ),
    }

    return assumptions


def add_planning_columns(
    frame: pd.DataFrame,
    assumptions: dict[str, float | int | str | bool],
) -> pd.DataFrame:
    """Calculate employee-level scenario contributions."""

    enriched = frame.copy()
    success_probability = float(
        assumptions["success_probability"]
    )
    intervention_cost = float(
        assumptions["intervention_cost_usd"]
    )

    selected = enriched["selected_for_human_review"].eq(True)
    enriched["expected_departure_contribution"] = enriched[
        "attrition_probability"
    ]
    enriched["expected_prevented_departure_contribution"] = (
        enriched["attrition_probability"]
        * success_probability
        * selected.astype(float)
    )
    enriched["residual_backfill_contribution"] = (
        enriched["expected_departure_contribution"]
        - enriched["expected_prevented_departure_contribution"]
    )
    enriched["replacement_cost_exposure_usd"] = (
        enriched["attrition_probability"]
        * enriched["replacement_cost_usd"]
    )
    enriched["planned_avoided_cost_usd"] = np.where(
        selected,
        enriched["predicted_avoidable_cost_usd"],
        0.0,
    )
    enriched["planned_intervention_spend_usd"] = np.where(
        selected,
        intervention_cost,
        0.0,
    )
    enriched["planned_net_value_usd"] = (
        enriched["planned_avoided_cost_usd"]
        - enriched["planned_intervention_spend_usd"]
    )
    enriched["residual_replacement_cost_exposure_usd"] = (
        enriched["replacement_cost_exposure_usd"]
        - enriched["planned_avoided_cost_usd"]
    )
    enriched["residual_vacancy_days"] = (
        enriched["residual_backfill_contribution"]
        * float(assumptions["vacancy_days"])
    )
    enriched["residual_time_to_productivity_days"] = (
        enriched["residual_backfill_contribution"]
        * float(assumptions["time_to_productivity_days"])
    )
    enriched["residual_training_hours"] = (
        enriched["residual_backfill_contribution"]
        * float(assumptions["training_hours"])
    )
    enriched["residual_coverage_hours"] = (
        enriched["residual_backfill_contribution"]
        * float(assumptions["coverage_hours"])
    )

    return enriched


def aggregate_plan(
    frame: pd.DataFrame,
    group_columns: list[str],
    total_active: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate probability-weighted planning metrics."""

    plan = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            model_eligible_employees=("employee_id", "count"),
            expected_departures_12m=(
                "expected_departure_contribution",
                "sum",
            ),
            selected_for_human_review=(
                "selected_for_human_review",
                "sum",
            ),
            expected_prevented_departures=(
                "expected_prevented_departure_contribution",
                "sum",
            ),
            residual_expected_backfills=(
                "residual_backfill_contribution",
                "sum",
            ),
            replacement_cost_exposure_usd=(
                "replacement_cost_exposure_usd",
                "sum",
            ),
            planned_avoided_cost_usd=(
                "planned_avoided_cost_usd",
                "sum",
            ),
            planned_intervention_spend_usd=(
                "planned_intervention_spend_usd",
                "sum",
            ),
            planned_net_value_usd=(
                "planned_net_value_usd",
                "sum",
            ),
            residual_replacement_cost_exposure_usd=(
                "residual_replacement_cost_exposure_usd",
                "sum",
            ),
            residual_vacancy_days=(
                "residual_vacancy_days",
                "sum",
            ),
            residual_time_to_productivity_days=(
                "residual_time_to_productivity_days",
                "sum",
            ),
            residual_training_hours=(
                "residual_training_hours",
                "sum",
            ),
            residual_coverage_hours=(
                "residual_coverage_hours",
                "sum",
            ),
        )
    )

    if total_active is not None:
        plan = plan.merge(
            total_active,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )
    else:
        plan["active_workforce_employees"] = plan[
            "model_eligible_employees"
        ]

    plan["protected_active_employees"] = (
        plan["active_workforce_employees"]
        - plan["model_eligible_employees"]
    )
    plan["expected_departure_rate"] = (
        plan["expected_departures_12m"]
        / plan["model_eligible_employees"]
    )
    plan["selection_rate"] = (
        plan["selected_for_human_review"]
        / plan["model_eligible_employees"]
    )
    plan["intervention_offset_rate"] = np.where(
        plan["expected_departures_12m"].gt(0),
        (
            plan["expected_prevented_departures"]
            / plan["expected_departures_12m"]
        ),
        0.0,
    )

    return plan


def build_outputs(
    population: pd.DataFrame,
    frame: pd.DataFrame,
    assumptions: dict[str, float | int | str | bool],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build department, manufacturing-role, summary, and metadata tables."""

    enriched = add_planning_columns(frame, assumptions)
    active_by_department = (
        population.groupby(
            ["department_name", "department_group"],
            as_index=False,
        )
        .agg(
            active_workforce_employees=("employee_id", "count"),
        )
    )
    department = aggregate_plan(
        enriched,
        ["department_name", "department_group"],
        active_by_department,
    )
    focus_department = str(config["analysis"]["focus_department"])
    department["focus_department"] = department[
        "department_name"
    ].eq(focus_department)
    department["as_of_date"] = str(
        config["analysis"]["expected_as_of_date"]
    )
    department = department.sort_values(
        [
            "focus_department",
            "residual_expected_backfills",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    focus_frame = enriched.loc[
        enriched["department_name"].eq(focus_department)
    ].copy()
    active_focus_by_role = (
        population.loc[
            population["department_name"].eq(focus_department)
        ]
        .groupby(["job_title", "job_family"], as_index=False)
        .agg(
            active_workforce_employees=("employee_id", "count"),
        )
    )
    focus_role = aggregate_plan(
        focus_frame,
        ["job_title", "job_family"],
        active_focus_by_role,
    )
    focus_role["department_name"] = focus_department
    focus_role["as_of_date"] = str(
        config["analysis"]["expected_as_of_date"]
    )
    focus_role = focus_role.sort_values(
        "residual_expected_backfills",
        ascending=False,
    ).reset_index(drop=True)

    focus_row = department.loc[
        department["department_name"].eq(focus_department)
    ]

    if len(focus_row) != 1:
        raise ValueError(
            "The configured focus department must appear exactly once."
        )

    focus = focus_row.iloc[0]
    summary = pd.DataFrame(
        [
            {
                "as_of_date": assumptions["as_of_date"],
                "planning_horizon_months": config["analysis"][
                    "planning_horizon_months"
                ],
                "focus_department": focus_department,
                "active_workforce_employees": focus[
                    "active_workforce_employees"
                ],
                "model_eligible_employees": focus[
                    "model_eligible_employees"
                ],
                "protected_active_employees": focus[
                    "protected_active_employees"
                ],
                "expected_departures_12m": focus[
                    "expected_departures_12m"
                ],
                "selected_for_human_review": focus[
                    "selected_for_human_review"
                ],
                "expected_prevented_departures": focus[
                    "expected_prevented_departures"
                ],
                "residual_expected_backfills": focus[
                    "residual_expected_backfills"
                ],
                "replacement_cost_exposure_usd": focus[
                    "replacement_cost_exposure_usd"
                ],
                "planned_intervention_spend_usd": focus[
                    "planned_intervention_spend_usd"
                ],
                "planned_avoided_cost_usd": focus[
                    "planned_avoided_cost_usd"
                ],
                "planned_net_value_usd": focus[
                    "planned_net_value_usd"
                ],
                "residual_vacancy_days": focus[
                    "residual_vacancy_days"
                ],
                "residual_time_to_productivity_days": focus[
                    "residual_time_to_productivity_days"
                ],
                "residual_training_hours": focus[
                    "residual_training_hours"
                ],
                "residual_coverage_hours": focus[
                    "residual_coverage_hours"
                ],
                "policy_name": assumptions["policy_name"],
                "human_review_required": assumptions[
                    "human_review_required"
                ],
                "automatic_action_permitted": assumptions[
                    "automatic_action_permitted"
                ],
            }
        ]
    )

    metadata = pd.DataFrame(
        [
            {
                "as_of_date": assumptions["as_of_date"],
                "planning_label": config["analysis"][
                    "planning_label"
                ],
                "backfill_label": config["analysis"]["backfill_label"],
                "focus_department": focus_department,
                "policy_name": assumptions["policy_name"],
                "success_probability": assumptions[
                    "success_probability"
                ],
                "intervention_cost_usd": assumptions[
                    "intervention_cost_usd"
                ],
                "vacancy_days_per_replacement": assumptions[
                    "vacancy_days"
                ],
                "time_to_productivity_days_per_replacement": assumptions[
                    "time_to_productivity_days"
                ],
                "training_hours_per_replacement": assumptions[
                    "training_hours"
                ],
                "coverage_hours_per_replacement": assumptions[
                    "coverage_hours"
                ],
                "estimate_notice": config["interpretation"][
                    "estimate_notice"
                ],
                "intervention_notice": config["interpretation"][
                    "intervention_notice"
                ],
                "operational_notice": config["interpretation"][
                    "operational_notice"
                ],
                "governance_notice": config["interpretation"][
                    "governance_notice"
                ],
            }
        ]
    )

    return department, focus_role, summary, metadata


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


def validate_outputs(
    population: pd.DataFrame,
    frame: pd.DataFrame,
    department: pd.DataFrame,
    focus_role: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
    assumptions: dict[str, float | int | str | bool],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate reconciliation, formulas, timeline, and interpretation."""

    checks: list[dict[str, Any]] = []
    tolerance = float(config["validation"]["numeric_tolerance"])
    expected_date = str(config["analysis"]["expected_as_of_date"])
    focus_department = str(config["analysis"]["focus_department"])
    expected_selected = int(
        config["validation"]["expected_selected_employees"]
    )
    focus = summary.iloc[0]

    add_check(
        checks,
        "Current as-of date is exact",
        (
            department["as_of_date"].unique().tolist()
            == [expected_date]
            and focus_role["as_of_date"].unique().tolist()
            == [expected_date]
        ),
        department["as_of_date"].unique().tolist(),
        [expected_date],
        "All planning outputs describe one current workforce date.",
    )
    add_check(
        checks,
        "Current active population is unique",
        population["employee_id"].is_unique,
        int(population["employee_id"].duplicated().sum()),
        0,
        "Each active employee contributes once to workforce totals.",
    )
    add_check(
        checks,
        "Scored population is unique",
        frame["employee_id"].is_unique,
        int(frame["employee_id"].duplicated().sum()),
        0,
        "Each model-eligible employee contributes one current score.",
    )
    add_check(
        checks,
        "Department active headcount reconciles",
        int(department["active_workforce_employees"].sum())
        == len(population),
        int(department["active_workforce_employees"].sum()),
        len(population),
        "Department totals include eligible and protected active levels.",
    )
    add_check(
        checks,
        "Department eligible headcount reconciles",
        int(department["model_eligible_employees"].sum())
        == len(frame),
        int(department["model_eligible_employees"].sum()),
        len(frame),
        "Planning probabilities cover only model-eligible active levels.",
    )
    add_check(
        checks,
        "Frozen selected population reconciles",
        (
            int(department["selected_for_human_review"].sum())
            == expected_selected
            == int(assumptions["selected_employees"])
        ),
        int(department["selected_for_human_review"].sum()),
        expected_selected,
        "Checkpoint 48 does not change the top-700 review capacity.",
    )
    departure_difference = abs(
        float(department["expected_departures_12m"].sum())
        - float(frame["attrition_probability"].sum())
    )
    add_check(
        checks,
        "Expected departures reconcile to probabilities",
        departure_difference <= tolerance,
        departure_difference,
        f"<= {tolerance}",
        "Expected count equals the sum of individual probabilities.",
    )
    prevented_difference = abs(
        float(department["expected_prevented_departures"].sum())
        - float(assumptions["projected_prevented_departures"])
    )
    add_check(
        checks,
        "Expected prevented departures reconcile",
        prevented_difference <= tolerance,
        prevented_difference,
        f"<= {tolerance}",
        "The planning view reuses the frozen policy effectiveness.",
    )
    residual_difference = (
        department["residual_expected_backfills"]
        - (
            department["expected_departures_12m"]
            - department["expected_prevented_departures"]
        )
    ).abs().max()
    add_check(
        checks,
        "Residual backfill formula reconciles",
        float(residual_difference) <= tolerance,
        float(residual_difference),
        f"<= {tolerance}",
        "Residual demand equals expected departures minus expected prevention.",
    )
    spend_difference = abs(
        float(department["planned_intervention_spend_usd"].sum())
        - float(assumptions["projected_spend_usd"])
    )
    add_check(
        checks,
        "Intervention spend reconciles",
        spend_difference <= tolerance,
        spend_difference,
        f"<= {tolerance}",
        "Selected employees use the committed per-employee cost.",
    )
    avoided_difference = abs(
        float(department["planned_avoided_cost_usd"].sum())
        - float(assumptions["projected_avoided_cost_usd"])
    )
    add_check(
        checks,
        "Avoided cost reconciles",
        avoided_difference <= tolerance,
        avoided_difference,
        f"<= {tolerance}",
        "Department values sum to the current policy scenario.",
    )
    net_difference = abs(
        float(department["planned_net_value_usd"].sum())
        - float(assumptions["projected_net_value_usd"])
    )
    add_check(
        checks,
        "Net value reconciles",
        net_difference <= tolerance,
        net_difference,
        f"<= {tolerance}",
        "Expected avoided cost minus spend matches Checkpoint 46.",
    )
    vacancy_difference = (
        department["residual_vacancy_days"]
        - (
            department["residual_expected_backfills"]
            * float(assumptions["vacancy_days"])
        )
    ).abs().max()
    training_difference = (
        department["residual_training_hours"]
        - (
            department["residual_expected_backfills"]
            * float(assumptions["training_hours"])
        )
    ).abs().max()
    coverage_difference = (
        department["residual_coverage_hours"]
        - (
            department["residual_expected_backfills"]
            * float(assumptions["coverage_hours"])
        )
    ).abs().max()
    add_check(
        checks,
        "Operational-unit formulas reconcile",
        max(
            float(vacancy_difference),
            float(training_difference),
            float(coverage_difference),
        )
        <= tolerance,
        max(
            float(vacancy_difference),
            float(training_difference),
            float(coverage_difference),
        ),
        f"<= {tolerance}",
        "Backfill units use the committed base-scenario assumptions.",
    )
    add_check(
        checks,
        "Manufacturing focus is present",
        (
            str(focus["focus_department"]) == focus_department
            and len(focus_role) > 0
        ),
        str(focus["focus_department"]),
        focus_department,
        "The business framing is anchored on manufacturing stability.",
    )
    role_reconciliation = {
        "active": int(
            focus_role["active_workforce_employees"].sum()
        ),
        "eligible": int(
            focus_role["model_eligible_employees"].sum()
        ),
        "selected": int(
            focus_role["selected_for_human_review"].sum()
        ),
    }
    add_check(
        checks,
        "Manufacturing role totals reconcile",
        (
            role_reconciliation["active"]
            == int(focus["active_workforce_employees"])
            and role_reconciliation["eligible"]
            == int(focus["model_eligible_employees"])
            and role_reconciliation["selected"]
            == int(focus["selected_for_human_review"])
        ),
        role_reconciliation,
        {
            "active": int(focus["active_workforce_employees"]),
            "eligible": int(focus["model_eligible_employees"]),
            "selected": int(focus["selected_for_human_review"]),
        },
        "Role-level planning must sum to the Manufacturing summary.",
    )
    add_check(
        checks,
        "Residual planning values are nonnegative",
        (
            department["residual_expected_backfills"].ge(0).all()
            and department[
                "residual_replacement_cost_exposure_usd"
            ].ge(0).all()
        ),
        (
            f"minimum_backfills="
            f"{department['residual_expected_backfills'].min():.6f}; "
            f"minimum_cost="
            f"{department['residual_replacement_cost_exposure_usd'].min():.2f}"
        ),
        "Both minimums >= 0",
        "The scenario cannot prevent more departures or cost than exposed.",
    )
    target_columns = {
        "attrition_next_12m",
        "actual_attrition",
        "termination_date",
        "termination_type",
    }
    displayed_columns = (
        set(department.columns)
        | set(focus_role.columns)
        | set(summary.columns)
    )
    present_targets = sorted(target_columns & displayed_columns)
    add_check(
        checks,
        "No current future outcomes are used",
        not present_targets,
        present_targets,
        [],
        "Current planning uses scores and assumptions, not unknown outcomes.",
    )
    add_check(
        checks,
        "Policy remains frozen",
        (
            bool(assumptions["policy_frozen_before_test"])
            and bool(assumptions["test_evaluated_once"])
        ),
        (
            f"frozen={assumptions['policy_frozen_before_test']}; "
            f"test_once={assumptions['test_evaluated_once']}"
        ),
        "frozen=True; test_once=True",
        "Checkpoint 48 cannot retune the tested decision rule.",
    )
    add_check(
        checks,
        "Human review remains required",
        (
            bool(assumptions["human_review_required"])
            and not bool(assumptions["automatic_action_permitted"])
        ),
        (
            f"human_review={assumptions['human_review_required']}; "
            f"automatic_action="
            f"{assumptions['automatic_action_permitted']}"
        ),
        "human_review=True; automatic_action=False",
        "Workforce planning does not authorize employment action.",
    )
    add_check(
        checks,
        "Scenario caveats are present",
        all(
            bool(str(metadata.iloc[0][column]).strip())
            for column in [
                "estimate_notice",
                "intervention_notice",
                "operational_notice",
                "governance_notice",
            ]
        ),
        "All present",
        "All present",
        "The dashboard must not present expected counts as exact forecasts.",
    )

    validation = pd.DataFrame(checks)
    failed = validation.loc[validation["status"].eq("FAIL")]

    if not failed.empty:
        raise ValueError(
            "Checkpoint 48 validation failed:\n"
            + failed[["check", "observed", "requirement"]].to_string(
                index=False
            )
        )

    return validation


def main() -> None:
    """Build, validate, and save the workforce-stability scenario."""

    required = [
        CONFIG_PATH,
        CURRENT_POPULATION_PATH,
        CURRENT_SCORE_PATH,
        CURRENT_POLICY_SUMMARY_PATH,
        POLICY_DECISION_PATH,
    ]
    require_files(required)
    config = load_yaml(CONFIG_PATH)
    current_population = pd.read_csv(CURRENT_POPULATION_PATH)
    scores = pd.read_csv(CURRENT_SCORE_PATH)
    policy_summary = pd.read_csv(CURRENT_POLICY_SUMMARY_PATH)
    policy_decision = pd.read_csv(POLICY_DECISION_PATH)

    population, frame = prepare_population(
        current_population,
        scores,
        config,
    )
    assumptions = policy_assumptions(
        policy_decision,
        policy_summary,
        config,
    )
    department, focus_role, summary, metadata = build_outputs(
        population,
        frame,
        assumptions,
        config,
    )
    validation = validate_outputs(
        population,
        frame,
        department,
        focus_role,
        summary,
        metadata,
        assumptions,
        config,
    )

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    department.to_csv(DEPARTMENT_OUTPUT_PATH, index=False)
    focus_role.to_csv(FOCUS_ROLE_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    validation.to_csv(VALIDATION_OUTPUT_PATH, index=False)
    department.to_csv(DASHBOARD_DEPARTMENT_PATH, index=False)
    focus_role.to_csv(DASHBOARD_ROLE_PATH, index=False)
    summary.to_csv(DASHBOARD_SUMMARY_PATH, index=False)
    metadata.to_csv(DASHBOARD_METADATA_PATH, index=False)

    print("\nDEPARTMENT WORKFORCE-STABILITY SCENARIO")
    display_columns = [
        "department_name",
        "active_workforce_employees",
        "model_eligible_employees",
        "expected_departures_12m",
        "selected_for_human_review",
        "expected_prevented_departures",
        "residual_expected_backfills",
        "planned_net_value_usd",
    ]
    print(department[display_columns].to_string(index=False))
    print("\nMANUFACTURING STABILITY SUMMARY")
    print(summary.to_string(index=False))
    print("\nMANUFACTURING BACKFILL BY ROLE")
    role_columns = [
        "job_title",
        "active_workforce_employees",
        "expected_departures_12m",
        "selected_for_human_review",
        "expected_prevented_departures",
        "residual_expected_backfills",
        "residual_training_hours",
        "residual_coverage_hours",
    ]
    print(focus_role[role_columns].to_string(index=False))
    print("\nWORKFORCE-STABILITY VALIDATION")
    print(validation.to_string(index=False))
    print(
        "\nSaved workforce-stability outputs to: "
        f"{PROCESSED_DIR}"
    )
    print(
        "\nMANUFACTURING WORKFORCE-STABILITY ANALYSIS "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
