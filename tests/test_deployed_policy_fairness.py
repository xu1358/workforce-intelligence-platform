"""Contracts for deployed-policy subgroup fairness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from audit_retention_policy_fairness import (
    POLICY_FLAG_COLUMN,
    PROXY_FLAG_COLUMN,
    apply_selection_rules,
    build_group_metrics,
    build_overall_summary,
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one test contract."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def small_policy_population() -> pd.DataFrame:
    """Build a population where probability and expected value disagree."""

    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "snapshot_date": ["2024-06-30"] * 4,
            "attrition_probability": [0.40, 0.30, 0.20, 0.10],
            "actual_attrition": [1, 0, 1, 0],
            "base_salary": [20_000.0, 30_000.0, 200_000.0, 300_000.0],
            "age_band": ["Under 30", "Under 30", "30-39", "30-39"],
            "education_level": ["A", "A", "B", "B"],
            "region": ["West", "West", "South", "South"],
            "employment_type": ["Hourly", "Hourly", "Salaried", "Salaried"],
            "department_name": ["Support", "Support", "Engineering", "Engineering"],
            "job_level": ["Level 1", "Level 1", "Level 3", "Level 3"],
            "organizational_level": [
                "Individual Contributor",
                "Individual Contributor",
                "Team Manager",
                "Team Manager",
            ],
        }
    )


def small_frozen_policy() -> dict[str, Any]:
    """Return the fields consumed by the shared policy implementation."""

    return {
        "selected_policy": "budget_expected_value",
        "fixed_threshold": 0.50,
        "cost_optimal_threshold": 0.20,
        "top_fraction": 0.10,
        "top_count": 2,
        "maximum_employees": 2,
        "budget_usd": 5_000.0,
        "require_positive_expected_net_value": True,
        "replacement_cost_salary_multiplier": 1.0,
        "success_probability": 0.25,
        "intervention_cost_usd": 2_500.0,
    }


def small_config() -> dict[str, Any]:
    """Return the audit settings needed by pure-function tests."""

    return {
        "source_policy": {
            "selected_policy_key": "budget_expected_value",
        },
        "probability_proxy": {
            "fraction": 0.50,
            "label": "Probability-only top 50% diagnostic",
        },
        "group_attributes": [
            "age_band",
            "education_level",
            "region",
            "employment_type",
            "department_name",
            "job_level",
            "organizational_level",
        ],
        "comparison_eligibility": {
            "minimum_group_rows": 1,
            "minimum_positive_cases": 0,
            "minimum_negative_cases": 0,
            "minimum_selected_rows": 0,
        },
    }


def test_expected_value_policy_differs_from_probability_proxy() -> None:
    """The audit must expose rather than hide selection-rule disagreement."""

    audited = apply_selection_rules(
        small_policy_population(),
        small_frozen_policy(),
        small_config(),
    )

    assert audited[PROXY_FLAG_COLUMN].tolist() == [True, True, False, False]
    assert audited[POLICY_FLAG_COLUMN].tolist() == [False, False, True, True]
    assert not np.array_equal(
        audited[PROXY_FLAG_COLUMN],
        audited[POLICY_FLAG_COLUMN],
    )


def test_group_metrics_use_deployed_policy_flag() -> None:
    """Subgroup selection rates must be calculated from expected value."""

    audited = apply_selection_rules(
        small_policy_population(),
        small_frozen_policy(),
        small_config(),
    )
    groups = build_group_metrics(audited, small_config())
    employment = groups.loc[groups["attribute"].eq("employment_type")].set_index(
        "group"
    )

    assert employment.loc["Hourly", "selection_rate"] == 0.0
    assert employment.loc["Salaried", "selection_rate"] == 1.0


def test_overall_summary_names_both_rules() -> None:
    """Reviewer output must never label the proxy as deployed."""

    config = small_config()
    audited = apply_selection_rules(
        small_policy_population(),
        small_frozen_policy(),
        config,
    )
    summary = build_overall_summary(audited, config)

    assert summary["selection_rule"].tolist() == [
        "Frozen budget-constrained expected value",
        "Probability-only top 50% diagnostic",
    ]
    assert summary["cross_rule_overlap_count"].eq(0).all()


def test_committed_contract_prohibits_policy_replacement(
    project_root: Path,
) -> None:
    """The correction must audit rather than modify the frozen policy."""

    config = load_yaml(project_root / "config" / "retention_policy_fairness.yaml")
    governance = config["governance"]

    assert (
        config["source_policy"]["selection_function"]
        == "optimize_retention_policy.policy_flags"
    )
    assert config["probability_proxy"]["explicitly_not_deployed"] is True
    assert governance["accesses_reserved_test_target"] is False
    assert governance["changes_model_results"] is False
    assert governance["changes_frozen_policy"] is False
    assert governance["export_employee_level_rows"] is False


def test_old_fairness_code_labels_probability_proxy(
    project_root: Path,
) -> None:
    """Checkpoint 44 must not present its top fraction as the policy."""

    source = (project_root / "src" / "analyze_model_fairness.py").read_text(
        encoding="utf-8"
    )

    assert "def mark_top_fraction(" not in source
    assert "mark_probability_top_fraction_diagnostic" in source
    assert "selected_probability_top_fraction_diagnostic" in source
