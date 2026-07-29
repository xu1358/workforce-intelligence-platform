"""Unit tests for monthly attrition-hazard helper functions."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random
from typing import Any

import pandas as pd
import pytest
import yaml

from src.attrition_hazard import (
    _cause_probabilities,
    _clamp,
    _compensation_features,
    _current_location,
    _employee_is_at_risk,
    _feature_cutoff,
    _manager_change_count,
    _month_difference,
    _month_starts,
    _performance_features,
    _random_event_date,
    _records_before,
    _records_in_window,
    _robust_salary_positions,
    _training_features,
    _transform_feature,
    hazard_simulation_settings,
    load_hazard_config,
)


def test_committed_hazard_config_loads(
    project_root: Path,
) -> None:
    """The production hazard configuration should pass basic validation."""

    config = load_hazard_config(
        project_root / "config" / "attrition_hazard_config.yaml"
    )

    assert config["simulation"]["time_step"] == "month"
    assert hazard_simulation_settings(config).cause_model == "multinomial_logit"


def test_committed_simulation_keys_are_materialized(
    hazard_config: dict[str, Any],
) -> None:
    """Every specification key must become a typed runtime setting."""

    settings = hazard_simulation_settings(hazard_config)

    assert settings.time_step == "month"
    assert settings.as_of_date == pd.Timestamp("2026-06-30")
    assert settings.cause_model == "multinomial_logit"
    assert settings.feature_lag_months == 1
    assert settings.allow_first_month_exit is True
    assert settings.censor_at_as_of_date is True


@pytest.mark.parametrize(
    "missing_key",
    [
        "time_step",
        "as_of_date",
        "cause_model",
        "feature_lag_months",
        "allow_first_month_exit",
        "censor_at_as_of_date",
    ],
)
def test_hazard_config_rejects_missing_simulation_key(
    hazard_config: dict[str, Any],
    missing_key: str,
) -> None:
    """A simulation control cannot exist only as optional decoration."""

    config = deepcopy(hazard_config)
    del config["simulation"][missing_key]

    with pytest.raises(ValueError, match="missing required keys"):
        hazard_simulation_settings(config)


def test_hazard_config_rejects_unsupported_cause_model(
    hazard_config: dict[str, Any],
) -> None:
    """A named cause model must map to an implemented probability rule."""

    config = deepcopy(hazard_config)
    config["simulation"]["cause_model"] = "independent_binary_logits"

    with pytest.raises(ValueError, match="Unsupported simulation.cause_model"):
        hazard_simulation_settings(config)


@pytest.mark.parametrize("invalid_lag", [0, -1, 1.5, True, "1"])
def test_hazard_config_rejects_invalid_feature_lag(
    hazard_config: dict[str, Any],
    invalid_lag: object,
) -> None:
    """Feature lag must be an explicit positive whole number of months."""

    config = deepcopy(hazard_config)
    config["simulation"]["feature_lag_months"] = invalid_lag

    with pytest.raises(ValueError, match="integer of at least 1"):
        hazard_simulation_settings(config)


@pytest.mark.parametrize(
    ("key", "invalid_value", "message"),
    [
        ("allow_first_month_exit", 1, "true or false"),
        ("censor_at_as_of_date", "true", "true or false"),
    ],
)
def test_hazard_config_rejects_nonboolean_flags(
    hazard_config: dict[str, Any],
    key: str,
    invalid_value: object,
    message: str,
) -> None:
    """Boolean switches cannot be silently coerced from other types."""

    config = deepcopy(hazard_config)
    config["simulation"][key] = invalid_value

    with pytest.raises(ValueError, match=message):
        hazard_simulation_settings(config)


def test_hazard_config_rejects_uncensored_open_horizon(
    hazard_config: dict[str, Any],
) -> None:
    """The generator cannot pretend to observe beyond its as-of boundary."""

    config = deepcopy(hazard_config)
    config["simulation"]["censor_at_as_of_date"] = False

    with pytest.raises(ValueError, match="no observation horizon"):
        hazard_simulation_settings(config)


def test_hazard_config_rejects_invalid_baseline(
    tmp_path: Path,
    hazard_config: dict[str, Any],
) -> None:
    """Invalid baseline probabilities must fail before simulation."""

    config = dict(hazard_config)
    config["baseline_probabilities"] = {
        "stay_active": 0.90,
        "voluntary_exit": 0.05,
        "involuntary_exit": 0.01,
    }
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="add to 1"):
        load_hazard_config(path)


def test_hazard_config_rejects_nonmonthly_step(
    tmp_path: Path,
    hazard_config: dict[str, Any],
) -> None:
    """The implementation must not silently accept another time unit."""

    config = dict(hazard_config)
    config["simulation"] = dict(hazard_config["simulation"])
    config["simulation"]["time_step"] = "week"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="monthly"):
        load_hazard_config(path)


def test_month_starts_are_inclusive() -> None:
    """Every calendar month in the range should appear exactly once."""

    result = _month_starts(
        pd.Timestamp("2025-01-15"),
        pd.Timestamp("2025-03-02"),
    )

    assert result == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-02-01"),
        pd.Timestamp("2025-03-01"),
    ]


@pytest.mark.parametrize(
    ("later", "earlier", "expected"),
    [
        ("2025-03-01", "2024-12-15", 3),
        ("2025-01-01", "2025-01-31", 0),
        ("2024-01-01", "2025-01-01", 0),
    ],
)
def test_month_difference_is_nonnegative(
    later: str,
    earlier: str,
    expected: int,
) -> None:
    """Calendar-month recency cannot become negative."""

    assert (
        _month_difference(
            pd.Timestamp(later),
            pd.Timestamp(earlier),
        )
        == expected
    )


@pytest.mark.parametrize(
    ("lag_months", "expected"),
    [
        (1, "2025-03-01"),
        (2, "2025-02-01"),
        (4, "2024-12-01"),
    ],
)
def test_feature_cutoff_applies_configured_lag(
    lag_months: int,
    expected: str,
) -> None:
    """A one-month lag preserves the original exclusive month-start cutoff."""

    assert _feature_cutoff(
        pd.Timestamp("2025-03-01"),
        lag_months,
    ) == pd.Timestamp(expected)


def test_first_month_exit_switch_controls_hire_month_risk() -> None:
    """The hire month enters the risk set only when the switch allows it."""

    hire_date = pd.Timestamp("2025-03-15")
    month_start = pd.Timestamp("2025-03-01")
    month_end = pd.Timestamp("2025-03-31")

    assert _employee_is_at_risk(
        hire_date,
        month_start,
        month_end,
        allow_first_month_exit=True,
    )
    assert not _employee_is_at_risk(
        hire_date,
        month_start,
        month_end,
        allow_first_month_exit=False,
    )
    assert _employee_is_at_risk(
        hire_date,
        pd.Timestamp("2025-04-01"),
        pd.Timestamp("2025-04-30"),
        allow_first_month_exit=False,
    )


def test_record_cutoffs_exclude_future_and_cutoff_date() -> None:
    """Monthly features may only use records strictly before the cutoff."""

    records = [
        {"event_date": "2024-12-31", "value": 1},
        {"event_date": "2025-01-01", "value": 2},
        {"event_date": "2025-01-02", "value": 3},
    ]

    result = _records_before(
        records,
        "event_date",
        pd.Timestamp("2025-01-01"),
    )

    assert [record["value"] for record in result] == [1]


def test_record_window_is_left_inclusive_right_exclusive() -> None:
    """Trailing windows should include their start but exclude the cutoff."""

    records = [
        {"event_date": "2024-01-01", "value": 1},
        {"event_date": "2024-06-01", "value": 2},
        {"event_date": "2025-01-01", "value": 3},
    ]

    result = _records_in_window(
        records,
        "event_date",
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2025-01-01"),
    )

    assert [record["value"] for record in result] == [1, 2]


@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "expected"),
    [
        (-2.0, -1.0, 1.0, -1.0),
        (0.25, -1.0, 1.0, 0.25),
        (3.0, -1.0, 1.0, 1.0),
    ],
)
def test_clamp_respects_closed_interval(
    value: float,
    minimum: float,
    maximum: float,
    expected: float,
) -> None:
    """Probability and feature values must remain inside their limits."""

    assert _clamp(value, minimum, maximum) == expected


@pytest.mark.parametrize(
    ("raw_value", "feature_config", "expected"),
    [
        (2.0, {"transform": {"type": "binary"}}, 1.0),
        (
            5.0,
            {"transform": {"type": "capped_count", "maximum": 2}},
            2.0,
        ),
        (
            4.0,
            {
                "transform": {
                    "type": "z_score",
                    "center": 3.5,
                    "scale": 0.5,
                    "minimum": -2.5,
                    "maximum": 2.5,
                }
            },
            1.0,
        ),
        (
            4.0,
            {
                "transform": {
                    "type": "robust_z_score",
                    "minimum": -2.5,
                    "maximum": 2.5,
                }
            },
            2.5,
        ),
    ],
)
def test_supported_feature_transforms(
    raw_value: float,
    feature_config: dict[str, Any],
    expected: float,
) -> None:
    """Every configured transform should produce its documented result."""

    assert _transform_feature(raw_value, feature_config) == expected


def test_unsupported_feature_transform_fails() -> None:
    """Unknown transform names must not be silently ignored."""

    with pytest.raises(ValueError, match="Unsupported"):
        _transform_feature(
            1.0,
            {"transform": {"type": "mystery"}},
        )


def test_current_location_uses_only_prior_transfers() -> None:
    """A future transfer cannot change a historical monthly location."""

    transfers = [
        {
            "event_date": "2024-01-15",
            "old_value": 1,
            "new_value": 2,
        },
        {
            "event_date": "2025-08-01",
            "old_value": 2,
            "new_value": 3,
        },
    ]

    assert (
        _current_location(
            3,
            transfers,
            pd.Timestamp("2025-01-01"),
        )
        == 2
    )


def test_performance_features_have_safe_defaults_and_trend() -> None:
    """No-review defaults and two-review trends should be deterministic."""

    assert _performance_features([], pd.Timestamp("2025-01-01")) == (
        3.5,
        0.0,
        1.0,
    )

    records = [
        {"review_date": "2023-12-01", "performance_rating": 3.2},
        {"review_date": "2024-12-01", "performance_rating": 3.7},
    ]
    latest, trend, missing = _performance_features(
        records,
        pd.Timestamp("2025-01-01"),
    )

    assert latest == 3.7
    assert trend == pytest.approx(0.5)
    assert missing == 0.0


def test_compensation_features_calculate_growth_and_promotion_recency() -> None:
    """Salary growth and promotion recency should use prior records only."""

    records = [
        {
            "effective_date": "2024-01-01",
            "base_salary": 100_000,
            "change_reason": "Hire",
        },
        {
            "effective_date": "2024-12-01",
            "base_salary": 120_000,
            "change_reason": "Promotion",
        },
    ]

    salary, growth, months, no_promotion = _compensation_features(
        records,
        pd.Timestamp("2025-01-01"),
    )

    assert salary == 120_000
    assert growth == pytest.approx(0.20)
    assert months == 1.0
    assert no_promotion == 0.0


def test_training_and_manager_features_use_trailing_year() -> None:
    """Old and cutoff-date events must not enter trailing-year features."""

    training = [
        {
            "completion_date": "2023-12-31",
            "completion_status": "Completed",
            "training_hours": 100,
        },
        {
            "completion_date": "2024-02-01",
            "completion_status": "Completed",
            "training_hours": 8,
        },
        {
            "completion_date": "2024-03-01",
            "completion_status": "Failed",
            "training_hours": 4,
        },
        {
            "completion_date": "2025-01-01",
            "completion_status": "Completed",
            "training_hours": 50,
        },
    ]
    manager_changes = [
        {"event_date": "2024-01-01"},
        {"event_date": "2024-06-01"},
        {"event_date": "2025-01-01"},
    ]

    assert _training_features(
        training,
        pd.Timestamp("2025-01-01"),
    ) == (8.0, 1.0)
    assert (
        _manager_change_count(
            manager_changes,
            pd.Timestamp("2025-01-01"),
        )
        == 2.0
    )


def test_robust_salary_positions_center_peer_group() -> None:
    """Peer-relative salary positions should be finite and median-centered."""

    records = [
        {
            "employee_id": 1,
            "job_family": "Engineering",
            "job_level": 2,
            "current_salary": 80_000,
        },
        {
            "employee_id": 2,
            "job_family": "Engineering",
            "job_level": 2,
            "current_salary": 100_000,
        },
        {
            "employee_id": 3,
            "job_family": "Engineering",
            "job_level": 2,
            "current_salary": 120_000,
        },
    ]

    result = _robust_salary_positions(records)

    assert result[2] == pytest.approx(0.0)
    assert result[1] == pytest.approx(-result[3])
    assert all(math.isfinite(value) for value in result.values())


def test_cause_probabilities_are_bounded_and_sum_to_one(
    hazard_config: dict[str, Any],
) -> None:
    """Competing-risk probabilities must form a valid distribution."""

    stay, voluntary, involuntary = _cause_probabilities(
        20.0,
        20.0,
        hazard_config,
        "multinomial_logit",
    )

    limits = hazard_config["probability_limits"]
    assert stay + voluntary + involuntary == pytest.approx(1.0)
    assert voluntary + involuntary <= limits["maximum_combined_probability"] + 1e-12
    assert stay >= 0.0


def test_cause_probability_dispatch_rejects_unimplemented_model(
    hazard_config: dict[str, Any],
) -> None:
    """The cause-model name must select a real implementation branch."""

    with pytest.raises(ValueError, match="Unsupported cause model"):
        _cause_probabilities(
            0.0,
            0.0,
            hazard_config,
            "decorative_model_name",
        )


def test_random_event_date_stays_inside_at_risk_month() -> None:
    """Sampled termination dates must respect hire and as-of boundaries."""

    result = _random_event_date(
        random.Random(42),
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-15"),
        pd.Timestamp("2025-01-20"),
    )

    assert pd.Timestamp("2025-01-15") <= result <= pd.Timestamp("2025-01-20")
