"""Tests for retention economics and frozen policy calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.calculate_retention_economics import (
    build_assumption_table,
    build_population_summary,
    build_scenario_table,
)
from src.optimize_retention_policy import (
    calculate_policy_metrics,
    choose_cost_threshold,
    freeze_policy,
    individual_economics,
    policy_flags,
    reference_economics,
    stable_descending_order,
)


def economic_populations() -> pd.DataFrame:
    """Return small validation and current salary populations."""

    return pd.DataFrame(
        {
            "population": [
                "Validation",
                "Validation",
                "Current active",
                "Current active",
            ],
            "snapshot_date": [
                "2024-06-30",
                "2024-06-30",
                "2026-06-30",
                "2026-06-30",
            ],
            "employee_id": [1, 2, 3, 4],
            "department_name": [
                "Engineering",
                "Manufacturing",
                "Engineering",
                "Manufacturing",
            ],
            "base_salary": [80_000.0, 100_000.0, 90_000.0, 110_000.0],
        }
    )


def reference_assumptions(
    retention_policy: dict[str, Any],
    cost_policy: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the committed base economic assumptions."""

    return reference_economics(retention_policy, cost_policy)


def test_assumption_table_contains_all_nine_named_scenarios(
    cost_policy: dict[str, Any],
) -> None:
    """Three assumption families should each contribute three rows."""

    table = build_assumption_table(cost_policy)

    assert len(table) == 9
    assert set(table["assumption_family"]) == {
        "Replacement impact",
        "Intervention cost",
        "Intervention effectiveness",
    }
    base = table.loc[
        table["assumption_family"].eq("Replacement impact")
        & table["scenario"].eq("Base")
    ].iloc[0]
    assert base["coverage_hours_per_replacement"] == 180.0


def test_scenario_table_evaluates_full_cross_product(
    cost_policy: dict[str, Any],
) -> None:
    """Every population should receive all 27 economic scenarios."""

    table = build_scenario_table(
        economic_populations(),
        cost_policy,
    )

    assert len(table) == 54
    assert table.groupby("population").size().eq(27).all()
    assert not table["model_probabilities_used"].any()
    assert not table["operating_policy_selected"].any()
    assert not table["reserved_test_outcomes_used"].any()


def test_base_scenario_formulas_reconcile(
    cost_policy: dict[str, Any],
) -> None:
    """Replacement, avoidable-value, and coverage formulas must be exact."""

    table = build_scenario_table(
        economic_populations(),
        cost_policy,
    )
    row = table.loc[
        table["population"].eq("Validation")
        & table["replacement_impact_scenario"].eq("Base")
        & table["intervention_cost_scenario"].eq("Standard")
        & table["effectiveness_scenario"].eq("Base")
    ].iloc[0]

    assert row["average_base_salary"] == 90_000.0
    assert row["average_replacement_cost_usd"] == 90_000.0
    assert row["average_maximum_avoidable_cost_usd"] == 22_500.0
    assert row["coverage_hours_per_prevented_departure"] == 180.0


def test_population_summary_uses_reference_multiplier(
    cost_policy: dict[str, Any],
) -> None:
    """Department summaries should use the named reference scenario."""

    summary = build_population_summary(
        economic_populations(),
        cost_policy,
    )
    engineering_validation = summary.loc[
        summary["population"].eq("Validation")
        & summary["department_name"].eq("Engineering")
    ].iloc[0]

    assert engineering_validation[
        "reference_replacement_cost_multiplier"
    ] == 1.0
    assert engineering_validation[
        "average_reference_replacement_cost_usd"
    ] == 80_000.0


def test_individual_economics_formula(
    retention_policy: dict[str, Any],
    cost_policy: dict[str, Any],
) -> None:
    """Expected net value should subtract intervention cost once."""

    economics = reference_assumptions(retention_policy, cost_policy)
    replacement, avoidable, net = individual_economics(
        np.array([0.40, 0.10]),
        np.array([100_000.0, 80_000.0]),
        economics,
    )

    assert replacement.tolist() == [100_000.0, 80_000.0]
    assert avoidable.tolist() == [10_000.0, 2_000.0]
    assert net.tolist() == [7_500.0, -500.0]


def test_stable_order_breaks_ties_by_employee_id() -> None:
    """Equal values should use ascending employee ID as tie-break."""

    order = stable_descending_order(
        np.array([1.0, 2.0, 2.0]),
        np.array([3, 2, 1]),
    )

    assert order.tolist() == [2, 1, 0]


def test_cost_threshold_selects_nobody_when_value_is_negative(
    retention_policy: dict[str, Any],
    cost_policy: dict[str, Any],
) -> None:
    """The validation threshold must allow a no-intervention result."""

    economics = reference_assumptions(retention_policy, cost_policy)
    validation = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "attrition_probability": [0.01, 0.02],
            "base_salary": [50_000.0, 50_000.0],
        }
    )

    threshold, count, table = choose_cost_threshold(
        validation,
        economics,
    )

    assert threshold == 1.0
    assert count == 0
    assert len(table) == 2


def test_frozen_policy_hash_is_deterministic(
    retention_policy: dict[str, Any],
    cost_policy: dict[str, Any],
) -> None:
    """The same validation evidence must produce the same frozen policy."""

    economics = reference_assumptions(retention_policy, cost_policy)
    validation = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "attrition_probability": [0.40, 0.30, 0.20],
            "base_salary": [100_000.0, 90_000.0, 80_000.0],
        }
    )

    first, _ = freeze_policy(
        validation,
        retention_policy,
        economics,
    )
    second, _ = freeze_policy(
        validation,
        retention_policy,
        economics,
    )

    assert first["policy_sha256"] == second["policy_sha256"]
    assert first["maximum_employees"] == 700
    assert first["test_target_accessed_during_selection"] is False
    assert first["post_test_changes_allowed"] is False


def test_policy_flags_respect_capacity_budget_and_positive_value() -> None:
    """Expected-value selection must stop at all frozen constraints."""

    frame = pd.DataFrame(
        {
            "employee_id": [4, 3, 2, 1],
            "attrition_probability": [0.40, 0.20, 0.10, 0.05],
            "base_salary": [100_000.0] * 4,
        }
    )
    frozen = {
        "replacement_cost_salary_multiplier": 1.0,
        "success_probability": 0.25,
        "intervention_cost_usd": 2_500.0,
        "fixed_threshold": 0.50,
        "cost_optimal_threshold": 0.15,
        "top_fraction": 0.50,
        "top_count": 3,
        "maximum_employees": 2,
        "budget_usd": 5_000.0,
        "require_positive_expected_net_value": True,
    }

    flags = policy_flags(frame, frozen)

    assert flags["fixed_threshold"].sum() == 0
    assert flags["cost_optimal_threshold"].sum() == 2
    assert flags["top_fraction"].sum() == 2
    assert flags["top_count"].sum() == 3
    assert flags["budget_expected_value"].tolist() == [
        True,
        True,
        False,
        False,
    ]


def test_policy_metrics_reconcile_expected_and_observed_value() -> None:
    """Policy reporting formulas should match a hand-calculated example."""

    frame = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "attrition_probability": [0.40, 0.20, 0.10],
            "base_salary": [100_000.0] * 3,
            "actual_attrition": [1, 0, 1],
        }
    )
    selected = np.array([True, False, True])
    economics = {
        "replacement_cost_salary_multiplier": 1.0,
        "success_probability": 0.25,
        "intervention_cost_usd": 2_500.0,
    }

    result = calculate_policy_metrics(
        frame,
        selected,
        "example",
        "Example policy",
        "validation",
        economics,
    )

    assert result["selected_count"] == 2
    assert result["selected_positive_cases"] == 2
    assert result["precision"] == 1.0
    assert result["capture_rate"] == 1.0
    assert result["intervention_spend_usd"] == 5_000.0
    assert result["expected_net_value_usd"] == pytest.approx(7_500.0)
    assert result["outcome_aligned_net_value_usd"] == 45_000.0
