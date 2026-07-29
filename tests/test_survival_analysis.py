"""Tests for the censoring-aware employee survival extension."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analyze_employee_survival import (
    build_cox_design,
    build_horizon_summary,
    build_overall_curve,
    build_survival_cohort,
    fit_kaplan_meier,
    parse_hire_event_notes,
)


def miniature_tables() -> dict[str, pd.DataFrame]:
    """Return three complete employee histories for unit tests."""

    employees = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "first_name": ["A", "B", "C"],
            "last_name": ["One", "Two", "Three"],
            "hire_date": pd.to_datetime(["2025-01-01", "2025-06-01", "2024-01-01"]),
            "termination_date": pd.to_datetime(["2025-12-31", None, None]),
            "employment_status": ["Terminated", "Active", "Active"],
            "termination_type": ["Voluntary", None, None],
            "department_id": [1, 1, 1],
            "location_id": [1, 1, 1],
            "job_role_id": [1, 1, 1],
            "manager_id": [None, 1, None],
            "employment_type": ["Hourly", "Salaried", "Salaried"],
            "birth_year": [1990, 1985, 1970],
            "education_level": ["Associate", "Bachelor's", "Master's"],
            "organizational_level": [
                "Individual Contributor",
                "Team Manager",
                "Senior Manager",
            ],
        }
    )
    events = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "event_type": ["Hire", "Hire", "Hire"],
            "notes": [
                "Hired into department_id=1, location_id=1, job_role_id=1.",
                "Hired into department_id=1, location_id=1, job_role_id=1.",
                "Hired into department_id=1, location_id=1, job_role_id=1.",
            ],
        }
    )
    compensation = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "effective_date": pd.to_datetime(
                ["2025-01-01", "2025-06-01", "2024-01-01"]
            ),
            "base_salary": [70000, 75000, 80000],
            "change_reason": ["Hire", "Hire", "Hire"],
        }
    )
    departments = pd.DataFrame(
        {
            "department_id": [1],
            "department_name": ["Engineering"],
            "department_group": ["Product"],
            "cost_center": ["ENG-1"],
        }
    )
    locations = pd.DataFrame(
        {
            "location_id": [1],
            "city": ["Austin"],
            "state": ["Texas"],
            "region": ["South"],
            "location_type": ["Office"],
        }
    )
    roles = pd.DataFrame(
        {
            "job_role_id": [1],
            "job_title": ["Engineer"],
            "job_family": ["Engineering"],
            "job_level": [1],
            "salary_band_min": [60000],
            "salary_band_max": [100000],
        }
    )
    return {
        "employees": employees,
        "events": events,
        "compensation": compensation,
        "departments": departments,
        "locations": locations,
        "roles": roles,
    }


def miniature_config() -> dict[str, Any]:
    """Return the relevant subset of the committed analysis contract."""

    return {
        "cohort": {
            "as_of_date": "2026-06-30",
            "days_per_month": 30.4375,
            "inclusive_tenure_days": True,
            "outcome_eligible_levels": [
                "Individual Contributor",
                "Team Manager",
            ],
            "protected_levels": ["Department Head", "Senior Manager"],
        },
        "cox_model": {
            "numeric_features": [
                "age_at_hire",
                "salary_position_at_hire",
            ],
            "categorical_features": [
                "employment_type",
                "education_level",
                "hire_department_name",
                "hire_job_level",
            ],
            "reference_categories": {
                "employment_type": "Hourly",
                "education_level": "Associate",
                "hire_department_name": "Engineering",
                "hire_job_level": "1",
            },
        },
    }


def test_committed_survival_contract_is_isolated(
    project_root: Any,
) -> None:
    """The optional extension must not alter operational decisions."""

    import yaml

    path = project_root / "config" / "survival_analysis.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    governance = config["governance"]
    assert governance["baseline_at_hire_features_only"]
    assert governance["associations_are_not_causes"]
    assert not governance["individual_survival_export_permitted"]
    assert not governance["modifies_primary_classifier"]
    assert not governance["accesses_reserved_test_target"]
    assert not governance["modifies_frozen_policy"]
    assert not governance["modifies_dashboard"]
    assert not governance["automatic_employment_action_permitted"]


def test_hire_event_parser_reconstructs_original_dimensions() -> None:
    """Hire notes must yield the three original organizational identifiers."""

    parsed = parse_hire_event_notes(miniature_tables()["events"])

    assert parsed.columns.tolist() == [
        "employee_id",
        "hire_department_id",
        "hire_location_id",
        "hire_job_role_id",
    ]
    assert parsed.iloc[0].to_dict() == {
        "employee_id": 1,
        "hire_department_id": 1,
        "hire_location_id": 1,
        "hire_job_role_id": 1,
    }


def test_hire_event_parser_rejects_unparseable_note() -> None:
    """Malformed notes must fail rather than use current organization fields."""

    events = miniature_tables()["events"].copy()
    events.loc[0, "notes"] = "Hired somewhere."

    try:
        parse_hire_event_notes(events)
    except ValueError as error:
        assert "cannot be parsed" in str(error)
    else:
        raise AssertionError("Malformed Hire notes should raise ValueError.")


def test_survival_cohort_encodes_events_and_censoring() -> None:
    """Known exits are events and current employees are right-censored."""

    all_cohort, model_cohort = build_survival_cohort(
        miniature_tables(),
        miniature_config(),
    )

    assert len(all_cohort) == 3
    assert len(model_cohort) == 2
    assert model_cohort["event_observed"].tolist() == [1, 0]
    assert model_cohort.loc[
        model_cohort["employee_id"].eq(2),
        "observation_end_date",
    ].iloc[0] == pd.Timestamp("2026-06-30")


def test_survival_duration_is_positive_and_inclusive() -> None:
    """The calculation includes both the hire and observation-end dates."""

    _, cohort = build_survival_cohort(
        miniature_tables(),
        miniature_config(),
    )
    employee = cohort.loc[cohort["employee_id"].eq(1)].iloc[0]

    expected_days = (pd.Timestamp("2025-12-31") - pd.Timestamp("2025-01-01")).days + 1
    assert employee["duration_days"] == expected_days
    assert np.isclose(
        employee["duration_months"],
        expected_days / 30.4375,
    )


def test_hire_salary_position_uses_original_role_band() -> None:
    """Baseline salary position must use the role midpoint at hire."""

    _, cohort = build_survival_cohort(
        miniature_tables(),
        miniature_config(),
    )
    first = cohort.loc[cohort["employee_id"].eq(1)].iloc[0]

    assert np.isclose(first["salary_position_at_hire"], 70000 / 80000)
    assert first["hire_department_name"] == "Engineering"
    assert first["hire_region"] == "South"


def test_kaplan_meier_curve_is_bounded_and_nonincreasing() -> None:
    """Retention probability must stay in range and cannot increase."""

    frame = pd.DataFrame(
        {
            "duration_months": [2.0, 4.0, 6.0, 8.0],
            "event_observed": [1, 0, 1, 0],
        }
    )
    curve = build_overall_curve(fit_kaplan_meier(frame))

    assert curve["survival_probability"].between(0, 1).all()
    assert curve["survival_probability"].diff().fillna(0).le(0).all()


def test_horizon_summary_preserves_requested_order() -> None:
    """Committed horizons should be returned in their configured order."""

    frame = pd.DataFrame(
        {
            "duration_months": [2.0, 4.0, 6.0, 8.0],
            "event_observed": [1, 0, 1, 0],
        }
    )
    summary = build_horizon_summary(
        fit_kaplan_meier(frame),
        [1, 3, 6],
    )

    assert summary["horizon_months"].tolist() == [1, 3, 6]
    assert summary["retention_probability"].between(0, 1).all()
    assert summary["cumulative_attrition_probability"].between(0, 1).all()


def test_cox_design_uses_references_and_standardized_numeric_features() -> None:
    """The design must omit configured references and center numeric fields."""

    cohort = pd.DataFrame(
        {
            "duration_months": [12.0, 18.0, 24.0, 30.0],
            "event_observed": [1, 0, 1, 0],
            "age_at_hire": [25.0, 35.0, 45.0, 55.0],
            "salary_position_at_hire": [0.9, 1.0, 1.1, 1.2],
            "employment_type": ["Hourly", "Salaried", "Hourly", "Salaried"],
            "education_level": [
                "Associate",
                "Bachelor's",
                "Associate",
                "Bachelor's",
            ],
            "hire_department_name": [
                "Engineering",
                "Finance",
                "Engineering",
                "Finance",
            ],
            "hire_job_level": ["1", "2", "1", "2"],
        }
    )
    design, manifest = build_cox_design(cohort, miniature_config())

    encoded = manifest["encoded_feature"].tolist()
    assert "employment_type_Hourly" not in encoded
    assert "employment_type_Salaried" in encoded
    assert "education_level_Bachelor_s" in encoded
    assert "hire_department_name_Finance" in encoded
    assert "hire_job_level_2" in encoded
    assert np.isclose(design["age_at_hire_z"].mean(), 0.0)
    assert np.isclose(design["salary_position_at_hire_z"].std(ddof=0), 1.0)
