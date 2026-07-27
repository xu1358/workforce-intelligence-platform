"""Cross-file tests for committed Version 2 configuration contracts."""

from __future__ import annotations

from typing import Any


def test_hazard_baseline_probabilities_form_distribution(
    hazard_config: dict[str, Any],
) -> None:
    """Monthly stay and exit probabilities must sum to one."""

    baseline = hazard_config["baseline_probabilities"]

    assert abs(sum(float(value) for value in baseline.values()) - 1.0) < 1e-9
    assert all(0.0 <= float(value) <= 1.0 for value in baseline.values())


def test_current_as_of_date_aligns_across_configs(
    hazard_config: dict[str, Any],
    temporal_config: dict[str, Any],
    retention_policy: dict[str, Any],
    stability_config: dict[str, Any],
) -> None:
    """Every current-state component must describe the same date."""

    dates = {
        hazard_config["simulation"]["as_of_date"],
        temporal_config["current_scoring"]["as_of_date"],
        retention_policy["model"]["current_snapshot"],
        stability_config["analysis"]["expected_as_of_date"],
    }

    assert dates == {"2026-06-30"}


def test_historical_snapshots_are_ordered_and_nonoverlapping(
    temporal_config: dict[str, Any],
) -> None:
    """Backtest windows must move forward without overlapping outcomes."""

    snapshots = temporal_config["historical_snapshots"]
    sequences = [int(item["sequence"]) for item in snapshots]
    snapshot_dates = [str(item["snapshot_date"]) for item in snapshots]
    end_dates = [str(item["prediction_end_date"]) for item in snapshots]

    assert sequences == [1, 2, 3]
    assert snapshot_dates == sorted(snapshot_dates)
    assert all(
        end_dates[index] <= snapshot_dates[index + 1]
        for index in range(len(snapshots) - 1)
    )


def test_feature_policy_periods_align_with_split_strategy(
    feature_policy: dict[str, Any],
    split_strategy: dict[str, Any],
) -> None:
    """Feature selection and model evaluation must share period boundaries."""

    configured = split_strategy["primary_temporal_split"]

    assert feature_policy["development_snapshots"] == [
        configured["train"]["snapshot_date"],
        configured["validation"]["snapshot_date"],
    ]
    assert (
        feature_policy["reserved_holdout_snapshot"]
        == configured["test"]["snapshot_date"]
    )


def test_model_population_levels_align_across_configs(
    feature_policy: dict[str, Any],
    cost_policy: dict[str, Any],
    stability_config: dict[str, Any],
) -> None:
    """Scoring, economics, and planning must use the same eligible levels."""

    feature_levels = set(
        feature_policy["model_population"][
            "eligible_organizational_levels"
        ]
    )
    cost_levels = set(
        cost_policy["population"]["eligible_organizational_levels"]
    )
    stability_levels = set(
        stability_config["population"][
            "eligible_organizational_levels"
        ]
    )

    assert feature_levels == cost_levels == stability_levels
    assert feature_levels == {
        "Individual Contributor",
        "Team Manager",
    }


def test_protected_levels_align_across_configs(
    hazard_config: dict[str, Any],
    feature_policy: dict[str, Any],
    stability_config: dict[str, Any],
) -> None:
    """Protected hierarchy levels must remain consistently excluded."""

    assert set(hazard_config["hierarchy"]["protected_levels"]) == set(
        feature_policy["model_population"][
            "excluded_organizational_levels"
        ]
    )
    assert set(
        stability_config["population"][
            "protected_organizational_levels"
        ]
    ) == {"Department Head", "Senior Manager"}


def test_cost_scenario_cross_product_matches_contract(
    cost_policy: dict[str, Any],
) -> None:
    """All low/base/high economic assumptions must remain represented."""

    scenario_count = (
        len(cost_policy["replacement_impact_scenarios"])
        * len(cost_policy["intervention_cost_scenarios"])
        * len(cost_policy["intervention_effectiveness_scenarios"])
    )

    assert scenario_count == 27
    assert (
        scenario_count
        == cost_policy["validation"]["expected_combined_scenarios"]
    )


def test_frozen_policy_capacity_matches_budget(
    retention_policy: dict[str, Any],
) -> None:
    """The 700-person reference policy must fit its explicit budget."""

    candidate = retention_policy["candidate_policies"][
        "budget_expected_value"
    ]

    assert candidate["maximum_employees"] == 700
    assert candidate["budget_usd"] == 1_750_000


def test_policy_freeze_and_once_only_test_rules_are_enabled(
    retention_policy: dict[str, Any],
) -> None:
    """The final test cannot influence policy selection or retuning."""

    assert retention_policy["policy_selection"][
        "freeze_before_test_access"
    ]
    assert retention_policy["policy_selection"][
        "prohibit_post_test_policy_changes"
    ]
    assert retention_policy["test_evaluation"][
        "access_test_target_once_after_policy_freeze"
    ]


def test_current_plan_requires_human_review(
    retention_policy: dict[str, Any],
    stability_config: dict[str, Any],
) -> None:
    """Risk scores may support review but cannot authorize employment action."""

    assert retention_policy["current_plan"]["require_human_review"]
    assert retention_policy["current_plan"][
        "prohibit_automatic_employment_action"
    ]
    assert "human review" in (
        stability_config["interpretation"]["governance_notice"].lower()
    )
    assert "automatic" in (
        stability_config["interpretation"]["governance_notice"].lower()
    )
