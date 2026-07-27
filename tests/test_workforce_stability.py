"""Tests for current workforce-stability planning calculations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.analyze_workforce_stability import (
    add_planning_columns,
    aggregate_plan,
    build_outputs,
    policy_assumptions,
    prepare_population,
)


def current_population_and_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create matching current population and score records."""

    population = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "snapshot_date": ["2026-06-30"] * 4,
            "department_name": [
                "Manufacturing",
                "Manufacturing",
                "Manufacturing",
                "Engineering",
            ],
            "department_group": [
                "Operations",
                "Operations",
                "Operations",
                "Product and Technology",
            ],
            "job_title": [
                "Production Technician",
                "Manufacturing Engineer",
                "Manufacturing Engineer",
                "Software Engineer",
            ],
            "job_family": [
                "Manufacturing",
                "Engineering",
                "Engineering",
                "Engineering",
            ],
            "organizational_level": [
                "Individual Contributor",
                "Team Manager",
                "Senior Manager",
                "Individual Contributor",
            ],
            "employment_type": [
                "Hourly",
                "Salaried",
                "Salaried",
                "Salaried",
            ],
            "job_level": [1, 3, 3, 2],
            "base_salary": [60_000.0, 100_000.0, 120_000.0, 90_000.0],
        }
    )
    scores = pd.DataFrame(
        {
            "employee_id": [1, 2, 4],
            "snapshot_date": ["2026-06-30"] * 3,
            "department_name": [
                "Manufacturing",
                "Manufacturing",
                "Engineering",
            ],
            "organizational_level": [
                "Individual Contributor",
                "Team Manager",
                "Individual Contributor",
            ],
            "employment_type": ["Hourly", "Salaried", "Salaried"],
            "job_level": [1, 3, 2],
            "base_salary": [60_000.0, 100_000.0, 90_000.0],
            "attrition_probability": [0.40, 0.20, 0.10],
            "replacement_cost_usd": [60_000.0, 100_000.0, 90_000.0],
            "predicted_avoidable_cost_usd": [
                6_000.0,
                5_000.0,
                2_250.0,
            ],
            "predicted_net_value_usd": [
                3_500.0,
                2_500.0,
                -250.0,
            ],
            "expected_value_rank": [1, 2, 3],
            "selected_for_human_review": [True, False, False],
        }
    )
    return population, scores


def planning_assumptions() -> dict[str, Any]:
    """Return base planning assumptions for hand calculations."""

    return {
        "as_of_date": "2026-06-30",
        "policy_name": "Budget-constrained expected value",
        "policy_frozen_before_test": True,
        "test_evaluated_once": True,
        "maximum_employees": 700,
        "budget_usd": 1_750_000.0,
        "intervention_cost_usd": 2_500.0,
        "success_probability": 0.25,
        "replacement_multiplier": 1.0,
        "vacancy_days": 60.0,
        "time_to_productivity_days": 90.0,
        "training_hours": 80.0,
        "coverage_hours": 180.0,
        "eligible_employees": 3,
        "selected_employees": 1,
        "projected_prevented_departures": 0.10,
        "projected_spend_usd": 2_500.0,
        "projected_avoided_cost_usd": 6_000.0,
        "projected_net_value_usd": 3_500.0,
        "human_review_required": True,
        "automatic_action_permitted": False,
    }


def test_prepare_population_reconciles_exact_score_coverage(
    stability_config: dict[str, Any],
) -> None:
    """All and only eligible active employees should receive scores."""

    population, scores = current_population_and_scores()
    complete, eligible = prepare_population(
        population,
        scores,
        stability_config,
    )

    assert len(complete) == 4
    assert set(eligible["employee_id"]) == {1, 2, 4}
    assert 3 not in set(eligible["employee_id"])


def test_prepare_population_rejects_missing_score(
    stability_config: dict[str, Any],
) -> None:
    """Missing eligible employees must fail reconciliation."""

    population, scores = current_population_and_scores()

    with pytest.raises(ValueError, match="exactly cover"):
        prepare_population(
            population,
            scores.iloc[:-1].copy(),
            stability_config,
        )


def test_policy_assumptions_extract_committed_governance(
    stability_config: dict[str, Any],
) -> None:
    """Planning assumptions must retain the frozen-policy safety fields."""

    decision = pd.DataFrame(
        [
            {
                "selected_policy_display_name": (
                    "Budget-constrained expected value"
                ),
                "policy_frozen_before_test": True,
                "test_evaluated_once": True,
                "maximum_employees": 700,
                "budget_usd": 1_750_000.0,
                "intervention_cost_usd": 2_500.0,
                "success_probability": 0.25,
                "replacement_cost_salary_multiplier": 1.0,
                "vacancy_days": 60.0,
                "time_to_productivity_days": 90.0,
                "training_hours_per_replacement": 80.0,
                "coverage_hours_per_replacement": 180.0,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-06-30",
                "eligible_employees": 3,
                "selected_for_human_review": 1,
                "projected_expected_prevented_departures": 0.10,
                "projected_intervention_spend_usd": 2_500.0,
                "projected_expected_avoided_cost_usd": 6_000.0,
                "projected_expected_net_value_usd": 3_500.0,
                "human_review_required": True,
                "automatic_employment_action_permitted": False,
            }
        ]
    )

    result = policy_assumptions(
        decision,
        summary,
        stability_config,
    )

    assert result["maximum_employees"] == 700
    assert result["human_review_required"] is True
    assert result["automatic_action_permitted"] is False


def test_planning_columns_match_hand_calculation() -> None:
    """Expected departures, prevention, and value should reconcile per row."""

    _, scores = current_population_and_scores()
    result = add_planning_columns(scores, planning_assumptions())

    assert result["expected_departure_contribution"].sum() == pytest.approx(
        0.70
    )
    assert result[
        "expected_prevented_departure_contribution"
    ].sum() == pytest.approx(0.10)
    assert result["residual_backfill_contribution"].sum() == pytest.approx(
        0.60
    )
    assert result["planned_intervention_spend_usd"].sum() == 2_500.0
    assert result["planned_avoided_cost_usd"].sum() == 6_000.0
    assert result["planned_net_value_usd"].sum() == 3_500.0


def test_aggregate_plan_includes_protected_headcount(
    stability_config: dict[str, Any],
) -> None:
    """Department totals should reconcile eligible and protected employees."""

    population, scores = current_population_and_scores()
    _, eligible = prepare_population(
        population,
        scores,
        stability_config,
    )
    enriched = add_planning_columns(eligible, planning_assumptions())
    active = (
        population.groupby(
            ["department_name", "department_group"],
            as_index=False,
        )
        .agg(active_workforce_employees=("employee_id", "count"))
    )
    result = aggregate_plan(
        enriched,
        ["department_name", "department_group"],
        active,
    )
    manufacturing = result.loc[
        result["department_name"].eq("Manufacturing")
    ].iloc[0]

    assert manufacturing["active_workforce_employees"] == 3
    assert manufacturing["model_eligible_employees"] == 2
    assert manufacturing["protected_active_employees"] == 1
    assert manufacturing["expected_departures_12m"] == pytest.approx(0.60)
    assert manufacturing["expected_prevented_departures"] == pytest.approx(
        0.10
    )
    assert manufacturing["residual_expected_backfills"] == pytest.approx(
        0.50
    )


def test_build_outputs_reconciles_manufacturing_roles(
    stability_config: dict[str, Any],
) -> None:
    """Role totals and summary should match the focus department."""

    population, scores = current_population_and_scores()
    _, eligible = prepare_population(
        population,
        scores,
        stability_config,
    )
    department, roles, summary, metadata = build_outputs(
        population,
        eligible,
        planning_assumptions(),
        stability_config,
    )
    manufacturing = department.loc[
        department["department_name"].eq("Manufacturing")
    ].iloc[0]

    assert roles["active_workforce_employees"].sum() == 3
    assert roles["model_eligible_employees"].sum() == 2
    assert summary.iloc[0]["active_workforce_employees"] == 3
    assert summary.iloc[0]["expected_departures_12m"] == pytest.approx(
        manufacturing["expected_departures_12m"]
    )
    assert metadata.iloc[0]["focus_department"] == "Manufacturing"
    assert "not exact" in metadata.iloc[0]["estimate_notice"]
