"""Monthly competing-risks attrition simulation.

The simulator consumes potential employee histories that were generated
without knowledge of future attrition.  For each calendar month it uses
only records dated before that month, then samples one of three outcomes:
stay active, voluntary exit, or involuntary exit.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import random
from typing import Any

import pandas as pd
import yaml


DATE_COLUMNS = {
    "compensation": "effective_date",
    "performance": "review_date",
    "training_start": "start_date",
    "training_completion": "completion_date",
    "events": "event_date",
}


def load_hazard_config(path: Path) -> dict[str, Any]:
    """Load and perform basic checks on the hazard configuration."""

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    baseline = config["baseline_probabilities"]
    baseline_sum = sum(float(value) for value in baseline.values())

    if not math.isclose(baseline_sum, 1.0, abs_tol=0.000001):
        raise ValueError(
            "Baseline stay and exit probabilities must add to 1."
        )

    if config["simulation"]["time_step"] != "month":
        raise ValueError("Only monthly hazard simulation is supported.")

    return config


def _month_starts(
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Return every month start in an inclusive date range."""

    first_month = first_date.to_period("M").to_timestamp()
    last_month = last_date.to_period("M").to_timestamp()

    return list(pd.date_range(first_month, last_month, freq="MS"))


def _month_difference(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    """Return the nonnegative number of calendar months between dates."""

    difference = (
        (later.year - earlier.year) * 12
        + later.month
        - earlier.month
    )
    return max(0, int(difference))


def _group_records(
    frame: pd.DataFrame,
    date_column: str,
) -> dict[int, list[dict[str, Any]]]:
    """Convert a dated table into sorted employee history lists."""

    histories: dict[int, list[dict[str, Any]]] = defaultdict(list)

    if frame.empty:
        return histories

    ordered = frame.sort_values(["employee_id", date_column])

    for record in ordered.to_dict(orient="records"):
        histories[int(record["employee_id"])].append(record)

    return histories


def _records_before(
    records: list[dict[str, Any]],
    date_column: str,
    cutoff: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Return records available strictly before the feature cutoff."""

    return [
        record
        for record in records
        if (
            not pd.isna(record[date_column])
            and pd.Timestamp(record[date_column]) < cutoff
        )
    ]


def _records_in_window(
    records: list[dict[str, Any]],
    date_column: str,
    window_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Return records in [window_start, cutoff)."""

    return [
        record
        for record in records
        if (
            not pd.isna(record[date_column])
            and window_start
            <= pd.Timestamp(record[date_column])
            < cutoff
        )
    ]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Limit a number to a closed interval."""

    return max(minimum, min(float(value), maximum))


def _transform_feature(
    raw_value: float,
    feature_config: dict[str, Any],
) -> float:
    """Apply the transformation declared in the YAML configuration."""

    transform = feature_config["transform"]
    transform_type = transform["type"]

    if transform_type == "binary":
        return 1.0 if raw_value else 0.0

    if transform_type == "capped_count":
        return min(float(raw_value), float(transform["maximum"]))

    if transform_type == "robust_z_score":
        return _clamp(
            raw_value,
            float(transform["minimum"]),
            float(transform["maximum"]),
        )

    if transform_type == "z_score":
        transformed = (
            (float(raw_value) - float(transform["center"]))
            / float(transform["scale"])
        )
        return _clamp(
            transformed,
            float(transform["minimum"]),
            float(transform["maximum"]),
        )

    raise ValueError(f"Unsupported feature transform: {transform_type}")


def _current_location(
    employee_location: int,
    transfer_records: list[dict[str, Any]],
    cutoff: pd.Timestamp,
) -> int:
    """Reconstruct the location known before the monthly cutoff."""

    if not transfer_records:
        return int(employee_location)

    ordered = sorted(
        transfer_records,
        key=lambda record: pd.Timestamp(record["event_date"]),
    )

    location_id = int(float(ordered[0]["old_value"]))

    for record in ordered:
        if pd.Timestamp(record["event_date"]) >= cutoff:
            break
        location_id = int(float(record["new_value"]))

    return location_id


def _performance_features(
    records: list[dict[str, Any]],
    cutoff: pd.Timestamp,
) -> tuple[float, float, float]:
    """Return level, trend, and no-review indicator."""

    available = _records_before(records, "review_date", cutoff)

    if not available:
        return 3.5, 0.0, 1.0

    ratings = [float(record["performance_rating"]) for record in available]
    trend = ratings[-1] - ratings[-2] if len(ratings) >= 2 else 0.0

    return ratings[-1], trend, 0.0


def _compensation_features(
    records: list[dict[str, Any]],
    cutoff: pd.Timestamp,
) -> tuple[float | None, float, float, float]:
    """Return salary, growth, months since promotion, and no-promotion."""

    available = _records_before(records, "effective_date", cutoff)

    if not available:
        return None, 0.03, 0.0, 1.0

    current_salary = float(available[-1]["base_salary"])
    trailing_start = cutoff - pd.DateOffset(years=1)
    prior_candidates = [
        record
        for record in available
        if pd.Timestamp(record["effective_date"]) <= trailing_start
    ]

    if prior_candidates:
        prior_salary = float(prior_candidates[-1]["base_salary"])
        growth = current_salary / prior_salary - 1.0
    else:
        growth = 0.03

    promotions = [
        record
        for record in available
        if str(record["change_reason"]) == "Promotion"
    ]

    if promotions:
        last_promotion = pd.Timestamp(promotions[-1]["effective_date"])
        months_since_promotion = _month_difference(cutoff, last_promotion)
        no_prior_promotion = 0.0
    else:
        months_since_promotion = 0.0
        no_prior_promotion = 1.0

    return (
        current_salary,
        growth,
        float(months_since_promotion),
        no_prior_promotion,
    )


def _training_features(
    records: list[dict[str, Any]],
    cutoff: pd.Timestamp,
) -> tuple[float, float]:
    """Return completed hours and failure count from the prior year."""

    window_start = cutoff - pd.DateOffset(years=1)
    available = _records_in_window(
        records,
        "completion_date",
        window_start,
        cutoff,
    )

    completed_hours = sum(
        float(record["training_hours"])
        for record in available
        if str(record["completion_status"]) == "Completed"
    )

    failed_count = sum(
        1
        for record in available
        if str(record["completion_status"]) == "Failed"
    )

    return completed_hours, float(failed_count)


def _manager_change_count(
    records: list[dict[str, Any]],
    cutoff: pd.Timestamp,
) -> float:
    """Count manager changes in the prior year."""

    window_start = cutoff - pd.DateOffset(years=1)
    return float(
        len(
            _records_in_window(
                records,
                "event_date",
                window_start,
                cutoff,
            )
        )
    )


def _robust_salary_positions(
    monthly_records: list[dict[str, Any]],
) -> dict[int, float]:
    """Calculate salary position within monthly peer groups."""

    salary_frame = pd.DataFrame(
        [
            {
                "employee_id": record["employee_id"],
                "job_family": record["job_family"],
                "job_level": record["job_level"],
                "salary": record["current_salary"],
            }
            for record in monthly_records
            if record["current_salary"] is not None
        ]
    )

    if salary_frame.empty:
        return {}

    group_columns = ["job_family", "job_level"]
    salary_frame["peer_median"] = salary_frame.groupby(
        group_columns
    )["salary"].transform("median")

    salary_frame["absolute_deviation"] = (
        salary_frame["salary"] - salary_frame["peer_median"]
    ).abs()
    salary_frame["peer_mad"] = salary_frame.groupby(
        group_columns
    )["absolute_deviation"].transform("median")

    denominator = 1.4826 * salary_frame["peer_mad"]
    fallback = (
        salary_frame.groupby(group_columns)["salary"].transform("std")
    )
    denominator = denominator.where(denominator > 0, fallback)
    denominator = denominator.fillna(1.0).where(denominator > 0, 1.0)

    salary_frame["salary_position"] = (
        (salary_frame["salary"] - salary_frame["peer_median"])
        / denominator
    )

    return dict(
        zip(
            salary_frame["employee_id"].astype(int),
            salary_frame["salary_position"].astype(float),
        )
    )


def _cause_probabilities(
    voluntary_logit: float,
    involuntary_logit: float,
    config: dict[str, Any],
) -> tuple[float, float, float]:
    """Convert cause logits to bounded multinomial probabilities."""

    voluntary_exp = math.exp(_clamp(voluntary_logit, -20.0, 20.0))
    involuntary_exp = math.exp(_clamp(involuntary_logit, -20.0, 20.0))
    denominator = 1.0 + voluntary_exp + involuntary_exp

    voluntary_probability = voluntary_exp / denominator
    involuntary_probability = involuntary_exp / denominator

    limits = config["probability_limits"]
    minimum = float(limits["minimum_cause_probability"])

    voluntary_probability = _clamp(
        voluntary_probability,
        minimum,
        float(limits["maximum_voluntary_probability"]),
    )
    involuntary_probability = _clamp(
        involuntary_probability,
        minimum,
        float(limits["maximum_involuntary_probability"]),
    )

    combined = voluntary_probability + involuntary_probability
    maximum_combined = float(limits["maximum_combined_probability"])

    if combined > maximum_combined:
        scale = maximum_combined / combined
        voluntary_probability *= scale
        involuntary_probability *= scale

    stay_probability = (
        1.0 - voluntary_probability - involuntary_probability
    )

    return (
        stay_probability,
        voluntary_probability,
        involuntary_probability,
    )


def _random_event_date(
    rng: random.Random,
    month_start: pd.Timestamp,
    hire_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> pd.Timestamp:
    """Choose an exit date within the employee's at-risk month."""

    event_start = max(month_start, hire_date.normalize())
    event_end = min(
        month_start + pd.offsets.MonthEnd(0),
        as_of_date,
    )

    available_days = int((event_end - event_start).days)
    return event_start + pd.to_timedelta(
        int(rng.randint(0, available_days)),
        unit="D",
    )


def simulate_attrition(
    employees: pd.DataFrame,
    job_roles: pd.DataFrame,
    compensation_history: pd.DataFrame,
    performance_reviews: pd.DataFrame,
    training_records: pd.DataFrame,
    employee_events: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate monthly attrition and return outcomes plus diagnostics."""

    employees = employees.copy()
    employees["hire_date"] = pd.to_datetime(employees["hire_date"])

    for frame, date_columns in [
        (compensation_history, ["effective_date"]),
        (performance_reviews, ["review_date"]),
        (training_records, ["start_date", "completion_date"]),
        (employee_events, ["event_date"]),
    ]:
        for date_column in date_columns:
            frame[date_column] = pd.to_datetime(frame[date_column])

    role_lookup = job_roles.set_index("job_role_id").to_dict(orient="index")

    compensation_by_employee = _group_records(
        compensation_history,
        "effective_date",
    )
    performance_by_employee = _group_records(
        performance_reviews,
        "review_date",
    )
    training_by_employee = _group_records(
        training_records,
        "start_date",
    )

    transfer_by_employee = _group_records(
        employee_events.loc[
            employee_events["event_type"].eq("Transfer")
        ],
        "event_date",
    )
    manager_changes_by_employee = _group_records(
        employee_events.loc[
            employee_events["event_type"].eq("Manager Change")
        ],
        "event_date",
    )

    rng = random.Random(int(config["random_seed"]))
    latent = config["latent_effects"]

    employee_ids = employees["employee_id"].astype(int).tolist()
    shared_frailty = {
        employee_id: rng.gauss(
            0.0,
            float(latent["shared_employee_frailty_sd"]),
        )
        for employee_id in employee_ids
    }
    voluntary_frailty = {
        employee_id: rng.gauss(
            0.0,
            float(latent["voluntary_employee_frailty_sd"]),
        )
        for employee_id in employee_ids
    }
    involuntary_frailty = {
        employee_id: rng.gauss(
            0.0,
            float(latent["involuntary_employee_frailty_sd"]),
        )
        for employee_id in employee_ids
    }

    as_of_date = pd.Timestamp(config["simulation"]["as_of_date"])
    month_starts = _month_starts(employees["hire_date"].min(), as_of_date)

    baseline = config["baseline_probabilities"]
    baseline_stay = float(baseline["stay_active"])
    voluntary_intercept = math.log(
        float(baseline["voluntary_exit"]) / baseline_stay
    )
    involuntary_intercept = math.log(
        float(baseline["involuntary_exit"]) / baseline_stay
    )

    protected_levels = set(
        config.get("hierarchy", {}).get("protected_levels", [])
    )
    active_employee_ids = set(employee_ids)
    employee_rows = {
        int(row.employee_id): row
        for row in employees.itertuples(index=False)
    }

    outcomes: dict[int, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    last_probabilities: dict[int, tuple[float, float, float]] = {}

    department_effects: dict[tuple[int, str], float] = {}
    location_effects: dict[tuple[int, str], float] = {}
    organization_effects: dict[str, float] = {}

    for month_start in month_starts:
        month_end = min(month_start + pd.offsets.MonthEnd(0), as_of_date)
        risk_ids = sorted(
            employee_id
            for employee_id in active_employee_ids
            if pd.Timestamp(employee_rows[employee_id].hire_date) <= month_end
        )

        monthly_records: list[dict[str, Any]] = []

        for employee_id in risk_ids:
            employee = employee_rows[employee_id]
            role = role_lookup[int(employee.job_role_id)]
            cutoff = month_start

            (
                performance_level,
                performance_trend,
                no_prior_review,
            ) = _performance_features(
                performance_by_employee.get(employee_id, []),
                cutoff,
            )

            (
                current_salary,
                trailing_growth,
                months_since_promotion,
                no_prior_promotion,
            ) = _compensation_features(
                compensation_by_employee.get(employee_id, []),
                cutoff,
            )

            tenure_months = _month_difference(
                cutoff,
                pd.Timestamp(employee.hire_date),
            )

            if no_prior_promotion:
                months_since_promotion = float(tenure_months)

            training_hours, failed_training_count = _training_features(
                training_by_employee.get(employee_id, []),
                cutoff,
            )

            manager_changes = _manager_change_count(
                manager_changes_by_employee.get(employee_id, []),
                cutoff,
            )

            location_id = _current_location(
                int(employee.location_id),
                transfer_by_employee.get(employee_id, []),
                cutoff,
            )

            monthly_records.append(
                {
                    "employee_id": employee_id,
                    "employee": employee,
                    "job_family": str(role["job_family"]),
                    "job_level": int(role["job_level"]),
                    "location_id": location_id,
                    "tenure_months": tenure_months,
                    "current_salary": current_salary,
                    "performance_level": performance_level,
                    "performance_trend": performance_trend,
                    "no_prior_review": no_prior_review,
                    "trailing_compensation_growth": trailing_growth,
                    "months_since_promotion": months_since_promotion,
                    "no_prior_promotion": no_prior_promotion,
                    "completed_training_hours": training_hours,
                    "failed_training_count": failed_training_count,
                    "manager_changes": manager_changes,
                }
            )

        salary_positions = _robust_salary_positions(monthly_records)
        voluntary_exits = 0
        involuntary_exits = 0
        protected_count = 0
        voluntary_probability_sum = 0.0
        involuntary_probability_sum = 0.0

        period_key = f"{month_start.year}-Q{month_start.quarter}"
        month_key = month_start.strftime("%Y-%m")

        if month_key not in organization_effects:
            organization_effects[month_key] = rng.gauss(
                0.0,
                float(latent["organization_period_shock_sd"]),
            )

        for record in monthly_records:
            employee_id = int(record["employee_id"])
            employee = record["employee"]
            department_key = (int(employee.department_id), period_key)
            location_key = (int(record["location_id"]), period_key)

            if department_key not in department_effects:
                department_effects[department_key] = rng.gauss(
                    0.0,
                    float(latent["department_period_effect_sd"]),
                )
            if location_key not in location_effects:
                location_effects[location_key] = rng.gauss(
                    0.0,
                    float(latent["location_period_effect_sd"]),
                )

            tenure_months = float(record["tenure_months"])
            onboarding = math.exp(-tenure_months / 3.0)
            early_tenure = math.exp(
                -((tenure_months - 18.0) / 12.0) ** 2
            )
            long_tenure = math.log1p(tenure_months) / math.log(73.0)

            voluntary_logit = voluntary_intercept
            involuntary_logit = involuntary_intercept

            for effect_name, effect_value in [
                ("onboarding_protection", onboarding),
                ("early_tenure_peak", early_tenure),
                ("long_tenure_stability", long_tenure),
            ]:
                effect_config = config["tenure_effects"][effect_name]
                voluntary_logit += (
                    effect_value
                    * float(effect_config["voluntary_coefficient"])
                )
                involuntary_logit += (
                    effect_value
                    * float(effect_config["involuntary_coefficient"])
                )

            raw_features = {
                "performance_level": record["performance_level"],
                "performance_trend": record["performance_trend"],
                "no_prior_review": record["no_prior_review"],
                "salary_position": salary_positions.get(employee_id, 0.0),
                "trailing_compensation_growth": record[
                    "trailing_compensation_growth"
                ],
                "months_since_promotion": record["months_since_promotion"],
                "no_prior_promotion": record["no_prior_promotion"],
                "completed_training_hours": record[
                    "completed_training_hours"
                ],
                "failed_training_count": record["failed_training_count"],
                "manager_changes": record["manager_changes"],
                "hourly_employee": (
                    str(employee.employment_type) == "Hourly"
                ),
                "job_level": record["job_level"],
            }

            for feature_name, raw_value in raw_features.items():
                feature_config = config["features"][feature_name]
                transformed = _transform_feature(
                    float(raw_value),
                    feature_config,
                )
                voluntary_logit += (
                    transformed
                    * float(feature_config["coefficients"]["voluntary"])
                )
                involuntary_logit += (
                    transformed
                    * float(feature_config["coefficients"]["involuntary"])
                )

            level_effect = config["organizational_level_effects"][
                str(employee.organizational_level)
            ]
            voluntary_logit += float(level_effect["voluntary"])
            involuntary_logit += float(level_effect["involuntary"])

            if config["seasonality"]["enabled"]:
                seasonal_value = math.sin(
                    2.0 * math.pi * month_start.month / 12.0
                )
                voluntary_logit += (
                    seasonal_value
                    * float(config["seasonality"]["voluntary_amplitude"])
                )
                involuntary_logit += (
                    seasonal_value
                    * float(config["seasonality"]["involuntary_amplitude"])
                )

            shared_context = (
                shared_frailty[employee_id]
                + department_effects[department_key]
                + location_effects[location_key]
                + organization_effects[month_key]
                + rng.gauss(
                    0.0,
                    float(latent["shared_monthly_noise_sd"]),
                )
            )
            voluntary_logit += (
                shared_context + voluntary_frailty[employee_id]
            )
            involuntary_logit += (
                shared_context + involuntary_frailty[employee_id]
            )

            probabilities = _cause_probabilities(
                voluntary_logit,
                involuntary_logit,
                config,
            )
            last_probabilities[employee_id] = probabilities
            _, voluntary_probability, involuntary_probability = probabilities

            voluntary_probability_sum += voluntary_probability
            involuntary_probability_sum += involuntary_probability

            if str(employee.organizational_level) in protected_levels:
                protected_count += 1
                continue

            draw = rng.random()
            termination_type: str | None = None

            if draw < voluntary_probability:
                termination_type = "Voluntary"
                voluntary_exits += 1
            elif draw < voluntary_probability + involuntary_probability:
                termination_type = "Involuntary"
                involuntary_exits += 1

            if termination_type is None:
                continue

            termination_date = _random_event_date(
                rng,
                month_start,
                pd.Timestamp(employee.hire_date),
                as_of_date,
            )

            outcomes[employee_id] = {
                "employee_id": employee_id,
                "employment_status": "Terminated",
                "termination_date": termination_date,
                "termination_type": termination_type,
                "stay_probability_at_exit": probabilities[0],
                "voluntary_probability_at_exit": probabilities[1],
                "involuntary_probability_at_exit": probabilities[2],
            }
            active_employee_ids.remove(employee_id)

        risk_count = len(monthly_records)
        diagnostics.append(
            {
                "simulation_month": month_start,
                "risk_set_size": risk_count,
                "average_voluntary_probability": (
                    voluntary_probability_sum / risk_count
                    if risk_count
                    else 0.0
                ),
                "average_involuntary_probability": (
                    involuntary_probability_sum / risk_count
                    if risk_count
                    else 0.0
                ),
                "voluntary_exits": voluntary_exits,
                "involuntary_exits": involuntary_exits,
                "protected_level_records": protected_count,
            }
        )

    outcome_records = []

    for employee_id in employee_ids:
        if employee_id in outcomes:
            record = outcomes[employee_id]
        else:
            probabilities = last_probabilities.get(
                employee_id,
                (1.0, 0.0, 0.0),
            )
            record = {
                "employee_id": employee_id,
                "employment_status": "Active",
                "termination_date": pd.NaT,
                "termination_type": None,
                "stay_probability_at_exit": probabilities[0],
                "voluntary_probability_at_exit": probabilities[1],
                "involuntary_probability_at_exit": probabilities[2],
            }

        record["shared_employee_frailty"] = shared_frailty[employee_id]
        record["voluntary_employee_frailty"] = voluntary_frailty[employee_id]
        record["involuntary_employee_frailty"] = involuntary_frailty[
            employee_id
        ]
        outcome_records.append(record)

    outcomes_frame = pd.DataFrame(outcome_records).sort_values("employee_id")
    outcomes_frame["termination_date"] = pd.to_datetime(
        outcomes_frame["termination_date"]
    )

    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame["simulation_month"] = pd.to_datetime(
        diagnostics_frame["simulation_month"]
    )

    return outcomes_frame, diagnostics_frame
