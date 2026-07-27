"""Tests for chronological and employee-grouped model assignments."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.create_model_splits import (
    assign_grouped_folds,
    configured_split_dates,
    create_assignment_output,
    create_primary_assignments,
    create_primary_summary,
    temporal_window_lookup,
)


def split_dataset() -> pd.DataFrame:
    """Create a repeated-employee temporal panel with three snapshots."""

    rows: list[dict[str, Any]] = []
    windows = [
        ("2023-06-30", "2024-06-30"),
        ("2024-06-30", "2025-06-30"),
        ("2025-06-30", "2026-06-30"),
    ]

    for employee_id in range(1, 61):
        for sequence, (snapshot, end) in enumerate(windows, start=1):
            rows.append(
                {
                    "employee_id": employee_id,
                    "snapshot_date": snapshot,
                    "prediction_end_date": end,
                    "organizational_level": (
                        "Individual Contributor"
                        if employee_id <= 55
                        else "Team Manager"
                    ),
                    "attrition_next_12m": int(
                        (employee_id + sequence) % 5 == 0
                    ),
                }
            )

    rows.append(
        {
            "employee_id": 999,
            "snapshot_date": "2024-06-30",
            "prediction_end_date": "2025-06-30",
            "organizational_level": "Senior Manager",
            "attrition_next_12m": 0,
        }
    )
    return pd.DataFrame(rows)


def test_configured_split_dates_preserve_order(
    split_strategy: dict[str, Any],
) -> None:
    """Train, validation, and test dates must remain chronological."""

    assert configured_split_dates(split_strategy) == {
        "train": "2023-06-30",
        "validation": "2024-06-30",
        "test": "2025-06-30",
    }


def test_temporal_window_lookup_maps_every_snapshot(
    temporal_config: dict[str, Any],
) -> None:
    """Each historical snapshot must map to its outcome-window end."""

    assert temporal_window_lookup(temporal_config) == {
        "2023-06-30": "2024-06-30",
        "2024-06-30": "2025-06-30",
        "2025-06-30": "2026-06-30",
    }


def test_primary_assignments_exclude_protected_levels_and_mask_test(
    split_strategy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> None:
    """Protected rows are excluded and reserved-test targets are hidden."""

    population = create_primary_assignments(
        split_dataset(),
        split_strategy,
        feature_policy,
    )

    assert 999 not in set(population["employee_id"])
    assert population.loc[
        population["primary_split"].eq("test"),
        "attrition_next_12m",
    ].isna().all()
    assert population.loc[
        population["primary_split"].isin(["train", "validation"]),
        "attrition_next_12m",
    ].notna().all()


def test_grouped_folds_keep_employee_histories_together(
    split_strategy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> None:
    """Repeated snapshots for one employee must use the same fold."""

    population = create_primary_assignments(
        split_dataset(),
        split_strategy,
        feature_policy,
    )
    assigned, summary = assign_grouped_folds(
        population,
        split_strategy,
    )
    development = assigned.loc[
        assigned["primary_split"].isin(["train", "validation"])
    ]

    assert development["group_cv_fold"].notna().all()
    assert (
        development.groupby("employee_id")["group_cv_fold"].nunique().max()
        == 1
    )
    assert summary["employee_overlap"].eq(0).all()
    assert set(summary["fold"]) == {1, 2, 3, 4, 5}


def test_reserved_test_has_no_group_fold(
    split_strategy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> None:
    """The final test is outside all development robustness folds."""

    population = create_primary_assignments(
        split_dataset(),
        split_strategy,
        feature_policy,
    )
    assigned, _ = assign_grouped_folds(population, split_strategy)

    assert assigned.loc[
        assigned["primary_split"].eq("test"),
        "group_cv_fold",
    ].isna().all()


def test_primary_summary_redacts_test_outcomes(
    split_strategy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> None:
    """The summary must not reveal test positives or rates."""

    population = create_primary_assignments(
        split_dataset(),
        split_strategy,
        feature_policy,
    )
    summary = create_primary_summary(population, split_strategy)
    test_row = summary.loc[
        summary["primary_split"].eq("test")
    ].iloc[0]

    assert test_row["target_status"] == "RESERVED_NOT_ACCESSED"
    assert test_row["positive_cases"] == "RESERVED"
    assert test_row["positive_rate"] == "RESERVED"


def test_assignment_artifact_contains_no_target(
    split_strategy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> None:
    """Saved split assignments must remain target-free."""

    population = create_primary_assignments(
        split_dataset(),
        split_strategy,
        feature_policy,
    )
    assigned, _ = assign_grouped_folds(population, split_strategy)
    output = create_assignment_output(assigned, split_strategy)

    assert "attrition_next_12m" not in output.columns
    assert output.loc[
        output["primary_split"].eq("test"),
        "reserved_test",
    ].all()
