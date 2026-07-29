"""Tests for the post-policy salary-allocation equity audit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit_retention_policy_equity import (
    FORBIDDEN_OUTCOME_COLUMNS,
    POLICY_ORDER,
    add_salary_groups,
    apply_policies,
    expected_net_value,
    select_ranked_rows,
    stable_descending_order,
)


def load_equity_config(project_root: Path) -> dict:
    """Load the committed policy-equity configuration."""

    path = project_root / "config" / "retention_policy_equity.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def synthetic_policy_frame() -> pd.DataFrame:
    """Create a small deterministic allocation population."""

    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "snapshot_date": ["2026-06-30"] * 5,
            "department_name": ["Manufacturing"] * 5,
            "organizational_level": ["Individual Contributor"] * 5,
            "employment_type": ["Salaried"] * 5,
            "job_level": [2, 2, 2, 2, 2],
            "base_salary": [50000, 75000, 100000, 125000, 150000],
            "attrition_probability": [0.30, 0.25, 0.20, 0.15, 0.10],
            "period": ["2026 current"] * 5,
        }
    )


def test_equity_configuration_defines_salary_specific_groups(
    project_root: Path,
) -> None:
    """The audit must go beyond job level and include direct pay measures."""

    config = load_equity_config(project_root)
    groups = config["salary_groups"]
    edges = groups["absolute_bands"]["edges"]
    labels = groups["absolute_bands"]["labels"]

    assert len(edges) == len(labels) + 1
    assert edges == sorted(edges)
    assert groups["population_quantiles"] == 5
    assert groups["within_job_level_quantiles"] == 5


def test_expected_value_contains_the_salary_mechanism(
    project_root: Path,
) -> None:
    """Equal probabilities should give higher value to higher replacement cost."""

    config = load_equity_config(project_root)
    probability = np.array([0.20, 0.20])
    replacement_cost = np.array([50000.0, 150000.0])

    result = expected_net_value(probability, replacement_cost, config)

    assert np.allclose(result, [0.0, 5000.0])
    assert result[1] > result[0]


def test_stable_order_breaks_score_ties_by_employee_id() -> None:
    """Allocation results should be reproducible when scores tie."""

    values = np.array([1.0, 2.0, 2.0, 0.5])
    employee_ids = np.array([9, 7, 3, 1])

    order = stable_descending_order(values, employee_ids)

    assert order.tolist() == [2, 1, 0, 3]


def test_ranked_selection_respects_capacity_and_positive_value(
    project_root: Path,
) -> None:
    """The shared selector must enforce the committed resource constraints."""

    config = deepcopy(load_equity_config(project_root))
    config["capacity"]["maximum_employees"] = 2
    config["capacity"]["maximum_budget_usd"] = 5000.0
    scores = np.array([500.0, 200.0, -1.0, 100.0])
    employee_ids = np.array([4, 3, 2, 1])

    selected = select_ranked_rows(
        scores,
        employee_ids,
        config,
        require_positive=True,
    )

    assert selected.tolist() == [True, True, False, False]


def test_policy_sensitivities_are_distinct_and_capacity_matched(
    project_root: Path,
) -> None:
    """Salary-dependent and salary-neutral rankings should be comparable."""

    config = deepcopy(load_equity_config(project_root))
    config["capacity"]["maximum_employees"] = 2
    config["capacity"]["maximum_budget_usd"] = 5000.0
    thresholds = {
        "validation_salary_cap_usd": 100000.0,
        "validation_constant_replacement_cost_usd": 80000.0,
    }

    result = apply_policies(synthetic_policy_frame(), config, thresholds)

    for policy_key in POLICY_ORDER:
        assert result[f"selected_{policy_key}"].sum() == 2

    frozen_ids = set(
        result.loc[
            result["selected_frozen_expected_value"],
            "employee_id",
        ]
    )
    probability_ids = set(
        result.loc[result["selected_probability_only"], "employee_id"]
    )
    assert frozen_ids != probability_ids


def test_salary_groups_cover_every_employee(project_root: Path) -> None:
    """Absolute and within-level pay bands should have complete coverage."""

    config = load_equity_config(project_root)
    grouped = add_salary_groups(synthetic_policy_frame(), config)

    columns = [
        "absolute_salary_band",
        "salary_quintile",
        "within_job_level_salary_quintile",
        "risk_quintile",
    ]
    assert not grouped[columns].isna().any().any()
    assert grouped["salary_quintile"].nunique() == 5


def test_governance_prohibits_retuning_and_automatic_action(
    project_root: Path,
) -> None:
    """The audit must not silently replace the once-tested policy."""

    config = load_equity_config(project_root)
    governance = config["governance"]

    assert governance["do_not_access_outcome_columns"]
    assert governance["do_not_modify_frozen_policy"]
    assert governance["sensitivity_policies_are_not_selected"]
    assert governance["new_policy_requires_new_holdout_or_prospective_evaluation"]
    assert governance["human_review_required"]
    assert not governance["automatic_employment_action_permitted"]


def test_source_contract_excludes_outcome_fields() -> None:
    """The module-level forbidden-field contract must remain explicit."""

    assert {
        "actual_attrition",
        "attrition_next_12m",
        "termination_date",
        "termination_type",
    } == FORBIDDEN_OUTCOME_COLUMNS


def test_policy_equity_notebook_is_executed(project_root: Path) -> None:
    """The reviewer-facing audit notebook must contain saved outputs."""

    path = project_root / "notebooks" / "34_retention_policy_equity.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert outputs
    assert not [output for output in outputs if output.get("output_type") == "error"]


def test_policy_equity_document_reports_headline_findings(
    project_root: Path,
) -> None:
    """The audit document should answer the salary-allocation question."""

    path = project_root / "docs" / "retention_policy_equity.md"
    text = path.read_text(encoding="utf-8")

    for value in [
        "$125,670",
        "21.22%",
        "0.82%",
        "25.83",
        "$108,935",
        "379",
    ]:
        assert value in text

    for phrase in [
        "financially optimized, not equity-neutral",
        "does not access outcome columns",
        "does not change the frozen policy",
        "new holdout",
    ]:
        assert " ".join(phrase.lower().split()) in " ".join(text.lower().split())
