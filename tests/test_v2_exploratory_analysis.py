"""Tests for authoritative Version 2 EDA and history attribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analyze_v2_exploratory_data import (
    build_period_structure,
    build_training_review_missingness,
    build_v1_v2_missingness_comparison,
)
from audit_v2_department_history import (
    PANEL_COLUMNS,
    build_department_continuity,
    classify_transfer_semantics,
    parse_hire_assignments,
)


def load_yaml(path: Path) -> dict:
    """Load one committed contract."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def eda_contract(project_root: Path) -> dict:
    """Load the Version 2 EDA contract."""

    return load_yaml(project_root / "config" / "v2_exploratory_analysis.yaml")


def history_contract(project_root: Path) -> dict:
    """Load the Version 2 history contract."""

    return load_yaml(project_root / "config" / "v2_department_history.yaml")


def small_panel() -> pd.DataFrame:
    """Return a compact three-period panel for leakage-boundary tests."""

    return pd.DataFrame(
        {
            "employee_id": [1, 2, 1, 2, 1, 2],
            "snapshot_sequence": [1, 1, 2, 2, 3, 3],
            "attrition_next_12m": [0, 1, 1, 1, 0, 0],
            "no_prior_review": [0, 1, 0, 1, 0, 1],
        }
    )


def test_period_structure_uses_training_outcomes_only(
    project_root: Path,
) -> None:
    """Validation and final outcomes must remain outside pre-model EDA."""

    contract = eda_contract(project_root)
    summary = build_period_structure(small_panel(), contract)

    assert summary.loc[0, "attrition_rate_used_for_eda"] == 0.5
    assert summary.loc[1:, "attrition_rate_used_for_eda"].isna().all()
    assert summary["eda_role"].tolist() == [
        "training_eda",
        "structure_only",
        "structure_only",
    ]


def test_v1_missingness_signal_is_not_reused_as_v2(
    project_root: Path,
) -> None:
    """The historical V1 result must be presented as non-authoritative."""

    contract = eda_contract(project_root)
    review = build_training_review_missingness(small_panel(), contract)
    comparison = build_v1_v2_missingness_comparison(review, contract)

    assert comparison["authoritative_for_v2"].tolist() == [False, True]
    assert comparison.iloc[0]["rows"] == 7386
    assert comparison.iloc[1]["rows"] == 2
    assert not np.isclose(
        comparison.iloc[0]["missing_minus_present_gap"],
        comparison.iloc[1]["missing_minus_present_gap"],
    )


def test_eda_contract_protects_reserved_outcomes(project_root: Path) -> None:
    """The documented exploratory scope must preserve temporal evaluation."""

    governance = eda_contract(project_root)["governance"]

    assert governance["version_2_is_authoritative"] is True
    assert governance["training_outcomes_only_for_eda"] is True
    assert governance["validation_outcomes_used_for_eda"] is False
    assert governance["final_test_outcomes_used_for_eda"] is False
    assert governance["current_outcomes_known"] is False
    assert governance["causal_claims_permitted"] is False
    assert governance["changes_selected_model"] is False


def test_hire_notes_parse_department_and_location() -> None:
    """Version 2 hire events must disclose both initial assignments."""

    events = pd.DataFrame(
        {
            "employee_id": [1, 2, 1],
            "event_date": ["2020-01-01", "2020-02-01", "2021-01-01"],
            "event_type": ["Hire", "Hire", "Promotion"],
            "notes": [
                "Hired into department_id=3, location_id=2, job_role_id=8.",
                "Hired into department_id=1, location_id=4, job_role_id=2.",
                "Promoted.",
            ],
        }
    )

    hires = parse_hire_assignments(events)

    assert hires["hire_department_id"].tolist() == [3, 1]
    assert hires["hire_location_id"].tolist() == [2, 4]


def test_transfer_semantics_do_not_invent_department_changes() -> None:
    """Location transfer notes must not be read as department transfers."""

    events = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "employee_id": [10, 11, 12],
            "event_type": ["Transfer", "Transfer", "Hire"],
            "notes": [
                "old_value and new_value represent location_id.",
                "old_value and new_value represent department_id.",
                "Hired.",
            ],
        }
    )

    transfers = classify_transfer_semantics(events)

    assert transfers["transfer_semantic"].tolist() == [
        "Location",
        "Department",
    ]


def test_department_continuity_detects_real_panel_changes() -> None:
    """The audit must detect a changing department if one appears."""

    employees = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "department_id": [10, 20],
            "location_id": [1, 1],
        }
    )
    departments = pd.DataFrame(
        {
            "department_id": [10, 20],
            "department_name": ["Engineering", "Finance"],
        }
    )
    panel = pd.DataFrame(
        {
            "employee_id": [1, 1, 2],
            "department_name": ["Engineering", "Finance", "Finance"],
        }
    )

    continuity = build_department_continuity(
        employees,
        departments,
        panel,
    )

    assert continuity["employees_with_multiple_panel_departments"].sum() == 1


def test_department_history_audit_is_target_free(project_root: Path) -> None:
    """History attribution must not inspect attrition outcomes."""

    governance = history_contract(project_root)["governance"]
    source = (project_root / "src" / "audit_v2_department_history.py").read_text(
        encoding="utf-8"
    )

    assert "attrition_next_12m" not in PANEL_COLUMNS
    assert governance["target_column_accessed"] is False
    assert governance["changes_model_inputs"] is False
    assert governance["changes_selected_model"] is False
    assert "retention_final_test_predictions.csv" not in source


def test_v2_successor_documents_and_notebooks_exist(
    project_root: Path,
) -> None:
    """Both historical V1 notebooks must have authoritative V2 successors."""

    paths = [
        project_root / "docs" / "v2_exploratory_analysis.md",
        project_root / "docs" / "v2_department_history.md",
        project_root / "notebooks" / "37_v2_exploratory_analysis.ipynb",
        project_root / "notebooks" / "38_v2_department_history.ipynb",
    ]

    assert all(path.is_file() for path in paths)
