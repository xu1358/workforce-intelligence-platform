"""Unit tests for leakage-safe temporal feature helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.build_multi_snapshot_retention_dataset import (
    add_salary_position,
    first_before,
    latest_before,
    maximum_timestamp,
    months_between,
    reconstruct_location_ids,
)


def dated_records() -> pd.DataFrame:
    """Return records with multiple employee histories."""

    return pd.DataFrame(
        {
            "employee_id": [1, 1, 1, 2, 2],
            "effective_date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2024-01-01",
                    "2025-01-01",
                    "2023-06-01",
                    "2024-06-01",
                ]
            ),
            "value": [10, 20, 30, 40, 50],
        }
    )


def test_latest_before_is_inclusive() -> None:
    """A record on the snapshot date is available to that snapshot."""

    result = latest_before(
        dated_records(),
        "effective_date",
        pd.Timestamp("2024-01-01"),
    ).set_index("employee_id")

    assert result.loc[1, "value"] == 20
    assert result.loc[2, "value"] == 40


def test_first_before_returns_earliest_available_record() -> None:
    """Initial-value features should use the first record before cutoff."""

    result = first_before(
        dated_records(),
        "effective_date",
        pd.Timestamp("2024-12-31"),
    ).set_index("employee_id")

    assert result.loc[1, "value"] == 10
    assert result.loc[2, "value"] == 40


def test_months_between_uses_calendar_months() -> None:
    """Calendar-month recency should not depend on month length."""

    earlier = pd.Series(pd.to_datetime(["2024-01-31", "2023-12-01"]))

    result = months_between(pd.Timestamp("2025-01-01"), earlier)

    assert result.tolist() == [12.0, 13.0]


def test_maximum_timestamp_handles_values_and_missing() -> None:
    """Cutoff audits should distinguish a real maximum from no date."""

    values = pd.Series(pd.to_datetime(["2024-01-01", "2025-03-01", None]))
    missing = pd.Series(pd.to_datetime([None, None]))

    assert maximum_timestamp(values) == pd.Timestamp("2025-03-01")
    assert maximum_timestamp(missing) is None


def test_location_reconstruction_ignores_future_transfer() -> None:
    """Historical location must not use a transfer after the snapshot."""

    eligible = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "location_id": [3, 4, 5],
        }
    )
    events = pd.DataFrame(
        {
            "employee_id": [1, 1, 1, 2],
            "event_date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2024-01-01",
                    "2026-01-01",
                    "2023-06-01",
                ]
            ),
            "event_type": ["Hire", "Transfer", "Transfer", "Hire"],
            "old_value": [None, "1", "2", None],
            "new_value": [None, "2", "3", None],
            "notes": [
                "Hired into department_id=1, location_id=1, job_role_id=4.",
                "Internal location transfer.",
                "Internal location transfer.",
                "Hired into department_id=2, location_id=4, job_role_id=7.",
            ],
        }
    )

    result = reconstruct_location_ids(
        eligible,
        events,
        pd.Timestamp("2025-06-30"),
    )

    assert result.tolist() == [2, 4, 5]


def test_salary_position_uses_job_peer_median() -> None:
    """Relative salary should compare employees within family and level."""

    frame = pd.DataFrame(
        {
            "job_family": [
                "Engineering",
                "Engineering",
                "Engineering",
                "Sales",
            ],
            "job_level": [2, 2, 2, 2],
            "base_salary": [80_000, 100_000, 120_000, 70_000],
        }
    )

    result = add_salary_position(frame.copy())

    assert result.loc[0, "salary_position_percent"] == pytest.approx(-20.0)
    assert result.loc[1, "salary_position_percent"] == pytest.approx(0.0)
    assert result.loc[2, "salary_position_percent"] == pytest.approx(20.0)
    assert result.loc[3, "salary_position_percent"] == pytest.approx(0.0)
