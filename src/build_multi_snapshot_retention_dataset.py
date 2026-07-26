"""Build leakage-safe historical and current retention datasets.

Historical snapshots contain features available by each snapshot date
and a target covering the following twelve months. The current scoring
population contains active employees as of the simulation date and
deliberately contains no known future outcome.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VALIDATION_DIR = PROCESSED_DIR / "temporal_dataset_validation"
CONFIG_PATH = PROJECT_ROOT / "config" / "temporal_snapshots.yaml"
HAZARD_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "attrition_hazard_config.yaml"
)

HISTORICAL_OUTPUT = PROCESSED_DIR / "retention_multi_snapshot.csv"
CURRENT_OUTPUT = (
    PROCESSED_DIR / "current_active_scoring_population.csv"
)

DATE_COLUMNS = {
    "employees.csv": ["hire_date", "termination_date"],
    "compensation_history.csv": ["effective_date"],
    "performance_reviews.csv": ["review_date"],
    "training_records.csv": ["start_date", "completion_date"],
    "employee_events.csv": ["event_date"],
}

MODEL_COLUMNS = [
    "employee_id",
    "dataset_type",
    "snapshot_sequence",
    "snapshot_date",
    "prediction_end_date",
    "approx_age",
    "tenure_years",
    "employment_type",
    "education_level",
    "organizational_level",
    "department_name",
    "department_group",
    "city",
    "region",
    "location_type",
    "job_title",
    "job_family",
    "job_level",
    "initial_base_salary",
    "base_salary",
    "bonus_target",
    "equity_value",
    "salary_growth_percent",
    "salary_growth_12m_percent",
    "salary_position_percent",
    "days_since_compensation_change",
    "compensation_record_count",
    "promotion_compensation_count",
    "performance_rating",
    "goal_completion",
    "promotion_recommended",
    "days_since_review",
    "review_count",
    "average_performance_rating",
    "performance_trend",
    "no_prior_review",
    "completed_training_programs_12m",
    "failed_training_programs_12m",
    "completed_training_hours_12m",
    "average_training_score_12m",
    "prior_promotion_events",
    "prior_transfer_events",
    "prior_manager_change_events",
    "prior_leave_events",
    "prior_change_events",
    "promotions_12m",
    "manager_changes_12m",
    "leaves_12m",
    "days_since_last_change_event",
    "months_since_promotion",
    "no_prior_promotion",
    "attrition_next_12m",
]

DIRECT_LEAKAGE_COLUMNS = {
    "employment_status",
    "termination_date",
    "termination_type",
    "future_termination_date",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_table(filename: str) -> pd.DataFrame:
    """Load one raw table and parse its known date columns."""

    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {filename}. Run Checkpoint 37 first."
        )

    frame = pd.read_csv(path)
    for column in DATE_COLUMNS.get(filename, []):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    return frame


def latest_before(
    frame: pd.DataFrame,
    date_column: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Return the latest row per employee on or before a cutoff."""

    return (
        frame.loc[frame[date_column].le(cutoff)]
        .sort_values(["employee_id", date_column])
        .groupby("employee_id", as_index=False)
        .tail(1)
    )


def first_before(
    frame: pd.DataFrame,
    date_column: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Return the first row per employee on or before a cutoff."""

    return (
        frame.loc[frame[date_column].le(cutoff)]
        .sort_values(["employee_id", date_column])
        .groupby("employee_id", as_index=False)
        .head(1)
    )


def months_between(
    later: pd.Timestamp,
    earlier: pd.Series,
) -> pd.Series:
    """Calculate approximate whole months between dates."""

    return (
        (later.year - earlier.dt.year) * 12
        + later.month
        - earlier.dt.month
    ).astype("float64")


def maximum_timestamp(values: pd.Series) -> pd.Timestamp | None:
    """Return the maximum timestamp, or None when no date exists."""

    maximum = values.max()
    if pd.isna(maximum):
        return None

    return pd.Timestamp(maximum)


def reconstruct_location_ids(
    eligible: pd.DataFrame,
    events: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.Series:
    """Reconstruct location as it existed at a historical snapshot.

    Start with the hire-event location and apply the latest transfer
    recorded on or before the snapshot. No future transfer is used.
    """

    hire_events = events.loc[
        events["event_type"].eq("Hire")
    ].copy()
    hire_events["hire_location_id"] = pd.to_numeric(
        hire_events["notes"].str.extract(r"location_id=(\d+)")[0],
        errors="coerce",
    )
    hire_location_lookup = dict(
        zip(
            hire_events["employee_id"].astype(int),
            hire_events["hire_location_id"],
        )
    )

    prior_transfers = (
        events.loc[
            events["event_type"].eq("Transfer")
            & events["event_date"].le(snapshot_date)
        ]
        .sort_values(["employee_id", "event_date"])
        .groupby("employee_id", as_index=False)
        .last()[["employee_id", "new_value"]]
    )
    transferred_location = pd.to_numeric(
        prior_transfers["new_value"],
        errors="coerce",
    )
    transfer_lookup = dict(
        zip(
            prior_transfers["employee_id"].astype(int),
            transferred_location,
        )
    )

    hire_location = eligible["employee_id"].map(hire_location_lookup)
    reconstructed = eligible["employee_id"].map(transfer_lookup)
    reconstructed = reconstructed.fillna(hire_location)
    return reconstructed.fillna(eligible["location_id"]).astype(int)


def build_compensation_features(
    compensation: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Build compensation features using records available at snapshot."""

    available = compensation.loc[
        compensation["effective_date"].le(snapshot_date)
    ].copy()
    available = available.sort_values(
        ["employee_id", "effective_date"]
    )

    initial = first_before(
        compensation,
        "effective_date",
        snapshot_date,
    )[["employee_id", "base_salary"]].rename(
        columns={"base_salary": "initial_base_salary"}
    )

    latest = latest_before(
        compensation,
        "effective_date",
        snapshot_date,
    )[
        [
            "employee_id",
            "effective_date",
            "base_salary",
            "bonus_target",
            "equity_value",
        ]
    ].rename(columns={"effective_date": "latest_compensation_date"})

    counts = available.groupby("employee_id", as_index=False).agg(
        compensation_record_count=("compensation_id", "size"),
        promotion_compensation_count=(
            "change_reason",
            lambda values: int(values.eq("Promotion").sum()),
        ),
    )

    one_year_before = snapshot_date - pd.DateOffset(years=1)
    prior_year = latest_before(
        compensation,
        "effective_date",
        one_year_before,
    )[["employee_id", "base_salary"]].rename(
        columns={"base_salary": "base_salary_12m_reference"}
    )

    features = (
        latest.merge(initial, on="employee_id", how="left")
        .merge(counts, on="employee_id", how="left")
        .merge(prior_year, on="employee_id", how="left")
    )

    features["salary_growth_percent"] = (
        (
            features["base_salary"]
            / features["initial_base_salary"]
            - 1.0
        )
        * 100.0
    )

    reference_salary = features[
        "base_salary_12m_reference"
    ].fillna(features["initial_base_salary"])
    features["salary_growth_12m_percent"] = (
        (features["base_salary"] / reference_salary - 1.0) * 100.0
    )
    features["days_since_compensation_change"] = (
        snapshot_date - features["latest_compensation_date"]
    ).dt.days

    return features, maximum_timestamp(available["effective_date"])


def build_performance_features(
    performance: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Build performance features using reviews available at snapshot."""

    available = performance.loc[
        performance["review_date"].le(snapshot_date)
    ].sort_values(["employee_id", "review_date"])

    latest = (
        available.groupby("employee_id", as_index=False)
        .last()[
            [
                "employee_id",
                "review_date",
                "performance_rating",
                "goal_completion",
                "promotion_recommended",
            ]
        ]
        .rename(columns={"review_date": "latest_review_date"})
    )
    summary = available.groupby("employee_id", as_index=False).agg(
        review_count=("review_id", "size"),
        average_performance_rating=("performance_rating", "mean"),
        first_performance_rating=("performance_rating", "first"),
    )

    features = latest.merge(summary, on="employee_id", how="outer")
    features["performance_trend"] = (
        features["performance_rating"]
        - features["first_performance_rating"]
    )
    features["days_since_review"] = (
        snapshot_date - features["latest_review_date"]
    ).dt.days
    features["no_prior_review"] = 0

    return features, maximum_timestamp(available["review_date"])


def build_training_features(
    training: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Build trailing-twelve-month completed training features."""

    trailing_start = snapshot_date - pd.DateOffset(years=1)
    available = training.loc[
        training["completion_date"].notna()
        & training["completion_date"].gt(trailing_start)
        & training["completion_date"].le(snapshot_date)
    ].copy()

    if available.empty:
        return (
            pd.DataFrame(
                columns=[
                    "employee_id",
                    "completed_training_programs_12m",
                    "failed_training_programs_12m",
                    "completed_training_hours_12m",
                    "average_training_score_12m",
                ]
            ),
            None,
        )

    available["completed_hours"] = np.where(
        available["completion_status"].eq("Completed"),
        available["training_hours"],
        0.0,
    )
    available["completed_program"] = (
        available["completion_status"].eq("Completed").astype(int)
    )
    available["failed_program"] = (
        available["completion_status"].eq("Failed").astype(int)
    )
    completed_scores = available["score"].where(
        available["completion_status"].eq("Completed")
    )
    available["completed_score"] = completed_scores

    features = available.groupby("employee_id", as_index=False).agg(
        completed_training_programs_12m=("completed_program", "sum"),
        failed_training_programs_12m=("failed_program", "sum"),
        completed_training_hours_12m=("completed_hours", "sum"),
        average_training_score_12m=("completed_score", "mean"),
    )

    return features, maximum_timestamp(available["completion_date"])


def build_event_features(
    events: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Build all-time and trailing-twelve-month event features."""

    available = events.loc[
        events["event_date"].le(snapshot_date)
    ].copy()
    trailing_start = snapshot_date - pd.DateOffset(years=1)
    trailing = available.loc[
        available["event_date"].gt(trailing_start)
    ].copy()

    event_types = {
        "Promotion": "prior_promotion_events",
        "Transfer": "prior_transfer_events",
        "Manager Change": "prior_manager_change_events",
        "Leave": "prior_leave_events",
    }
    counts = pd.DataFrame(
        {"employee_id": available["employee_id"].unique()}
    )
    for event_type, column in event_types.items():
        event_count = (
            available.loc[available["event_type"].eq(event_type)]
            .groupby("employee_id")
            .size()
            .rename(column)
            .reset_index()
        )
        counts = counts.merge(event_count, on="employee_id", how="left")

    counts["prior_change_events"] = counts[
        list(event_types.values())
    ].sum(axis=1, min_count=1)

    trailing_types = {
        "Promotion": "promotions_12m",
        "Manager Change": "manager_changes_12m",
        "Leave": "leaves_12m",
    }
    for event_type, column in trailing_types.items():
        event_count = (
            trailing.loc[trailing["event_type"].eq(event_type)]
            .groupby("employee_id")
            .size()
            .rename(column)
            .reset_index()
        )
        counts = counts.merge(event_count, on="employee_id", how="left")

    change_events = available.loc[
        available["event_type"].isin(event_types)
    ]
    latest_change = (
        change_events.groupby("employee_id", as_index=False)[
            "event_date"
        ]
        .max()
        .rename(columns={"event_date": "latest_change_date"})
    )
    latest_promotion = (
        available.loc[available["event_type"].eq("Promotion")]
        .groupby("employee_id", as_index=False)["event_date"]
        .max()
        .rename(columns={"event_date": "latest_promotion_date"})
    )

    features = (
        counts.merge(latest_change, on="employee_id", how="left")
        .merge(latest_promotion, on="employee_id", how="left")
    )
    features["days_since_last_change_event"] = (
        snapshot_date - features["latest_change_date"]
    ).dt.days
    features["months_since_promotion"] = months_between(
        snapshot_date,
        features["latest_promotion_date"],
    )
    features["no_prior_promotion"] = (
        features["latest_promotion_date"].isna().astype(int)
    )

    return features, maximum_timestamp(available["event_date"])


def add_salary_position(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Add salary position relative to snapshot peer medians."""

    peer_median = snapshot.groupby(
        ["job_family", "job_level"],
        dropna=False,
    )["base_salary"].transform("median")
    snapshot["salary_position_percent"] = (
        (snapshot["base_salary"] / peer_median - 1.0) * 100.0
    )
    return snapshot


def build_one_snapshot(
    *,
    employees: pd.DataFrame,
    departments: pd.DataFrame,
    locations: pd.DataFrame,
    job_roles: pd.DataFrame,
    compensation: pd.DataFrame,
    performance: pd.DataFrame,
    training: pd.DataFrame,
    events: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    prediction_end_date: pd.Timestamp | None,
    sequence: int,
    dataset_type: str,
    minimum_tenure_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one feature snapshot and its temporal audit tables."""

    employed = (
        employees["hire_date"].le(snapshot_date)
        & (
            employees["termination_date"].isna()
            | employees["termination_date"].gt(snapshot_date)
        )
    )
    tenure_days = (snapshot_date - employees["hire_date"]).dt.days
    eligible = employees.loc[
        employed & tenure_days.ge(minimum_tenure_days)
    ].copy()

    eligible["snapshot_location_id"] = reconstruct_location_ids(
        eligible,
        events,
        snapshot_date,
    )
    eligible["snapshot_sequence"] = sequence
    eligible["dataset_type"] = dataset_type
    eligible["snapshot_date"] = snapshot_date
    eligible["prediction_end_date"] = prediction_end_date
    eligible["approx_age"] = snapshot_date.year - eligible["birth_year"]
    eligible["tenure_years"] = (
        snapshot_date - eligible["hire_date"]
    ).dt.days / 365.25

    if prediction_end_date is None:
        eligible["attrition_next_12m"] = pd.Series(
            pd.NA,
            index=eligible.index,
            dtype="Int64",
        )
    else:
        eligible["attrition_next_12m"] = (
            eligible["termination_date"].notna()
            & eligible["termination_date"].gt(snapshot_date)
            & eligible["termination_date"].le(prediction_end_date)
        ).astype("Int64")

    compensation_features, compensation_cutoff = (
        build_compensation_features(compensation, snapshot_date)
    )
    performance_features, performance_cutoff = (
        build_performance_features(performance, snapshot_date)
    )
    training_features, training_cutoff = build_training_features(
        training,
        snapshot_date,
    )
    event_features, event_cutoff = build_event_features(
        events,
        snapshot_date,
    )

    snapshot = (
        eligible.merge(
            departments,
            on="department_id",
            how="left",
        )
        .merge(
            locations,
            left_on="snapshot_location_id",
            right_on="location_id",
            how="left",
            suffixes=("", "_snapshot"),
        )
        .merge(job_roles, on="job_role_id", how="left")
        .merge(
            compensation_features,
            on="employee_id",
            how="left",
        )
        .merge(
            performance_features,
            on="employee_id",
            how="left",
        )
        .merge(
            training_features,
            on="employee_id",
            how="left",
        )
        .merge(event_features, on="employee_id", how="left")
    )

    snapshot = add_salary_position(snapshot)
    snapshot["no_prior_review"] = snapshot[
        "no_prior_review"
    ].fillna(1).astype(int)
    snapshot["performance_trend"] = snapshot[
        "performance_trend"
    ].fillna(0.0)

    integer_count_columns = [
        "compensation_record_count",
        "promotion_compensation_count",
        "review_count",
        "completed_training_programs_12m",
        "failed_training_programs_12m",
        "prior_promotion_events",
        "prior_transfer_events",
        "prior_manager_change_events",
        "prior_leave_events",
        "prior_change_events",
        "promotions_12m",
        "manager_changes_12m",
        "leaves_12m",
        "no_prior_promotion",
    ]
    snapshot[integer_count_columns] = (
        snapshot[integer_count_columns].fillna(0).astype(int)
    )
    snapshot["completed_training_hours_12m"] = snapshot[
        "completed_training_hours_12m"
    ].fillna(0.0)

    snapshot["months_since_promotion"] = snapshot[
        "months_since_promotion"
    ].fillna(snapshot["tenure_years"] * 12.0)

    for column in MODEL_COLUMNS:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA

    source_cutoffs = pd.DataFrame(
        {
            "dataset_type": dataset_type,
            "snapshot_sequence": sequence,
            "snapshot_date": snapshot_date,
            "source": [
                "compensation_history",
                "performance_reviews",
                "training_records",
                "employee_events",
            ],
            "latest_date_used": [
                compensation_cutoff,
                performance_cutoff,
                training_cutoff,
                event_cutoff,
            ],
            "required_cutoff": snapshot_date,
        }
    )

    row_audit = pd.DataFrame(
        {
            "employee_id": snapshot["employee_id"],
            "dataset_type": dataset_type,
            "snapshot_sequence": sequence,
            "snapshot_date": snapshot_date,
            "employee_hire_date": snapshot["hire_date"],
            "employee_termination_date": snapshot["termination_date"],
            "final_location_id": snapshot["location_id"],
            "snapshot_location_id": snapshot["snapshot_location_id"],
            "latest_compensation_date": snapshot[
                "latest_compensation_date"
            ],
            "latest_review_date": snapshot["latest_review_date"],
            "latest_change_date": snapshot["latest_change_date"],
            "latest_promotion_date": snapshot[
                "latest_promotion_date"
            ],
        }
    )

    return snapshot[MODEL_COLUMNS].copy(), source_cutoffs, row_audit


def add_validation_check(
    records: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    """Append one validation check."""

    records.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 fingerprint."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_temporal_outputs(
    *,
    historical: pd.DataFrame,
    current: pd.DataFrame,
    cutoffs: pd.DataFrame,
    row_audit: pd.DataFrame,
    config: dict[str, Any],
    employees: pd.DataFrame,
    simulation_as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate temporal construction, labels, and current scoring."""

    checks: list[dict[str, Any]] = []
    definitions = config["historical_snapshots"]
    validation = config["validation"]
    expected_dates = {
        pd.Timestamp(item["snapshot_date"]) for item in definitions
    }
    actual_dates = set(pd.to_datetime(historical["snapshot_date"]))
    configured_current_date = pd.Timestamp(
        config["current_scoring"]["as_of_date"]
    )

    add_validation_check(
        checks,
        "Current date matches simulation as-of",
        configured_current_date == simulation_as_of_date,
        configured_current_date.date(),
        str(simulation_as_of_date.date()),
        "Scoring must use the final date supported by generated history.",
    )

    latest_prediction_end = max(
        pd.Timestamp(item["prediction_end_date"])
        for item in definitions
    )
    add_validation_check(
        checks,
        "Historical outcomes fully observable",
        latest_prediction_end <= simulation_as_of_date,
        latest_prediction_end.date(),
        f"<= {simulation_as_of_date.date()}",
        "No historical label may extend beyond generated outcomes.",
    )

    add_validation_check(
        checks,
        "Expected historical snapshots",
        actual_dates == expected_dates,
        len(actual_dates),
        f"{len(expected_dates)} exact configured dates",
        "All configured historical snapshots must be present.",
    )

    duplicate_rows = int(
        historical.duplicated(["employee_id", "snapshot_date"]).sum()
    )
    current_duplicates = int(current["employee_id"].duplicated().sum())
    add_validation_check(
        checks,
        "Unique employee-snapshot keys",
        duplicate_rows == 0,
        duplicate_rows,
        "0 duplicates",
        "An employee may repeat across dates but only once per snapshot.",
    )
    add_validation_check(
        checks,
        "Unique current scoring employees",
        current_duplicates == 0,
        current_duplicates,
        "0 duplicates",
        "The current population must contain one row per employee.",
    )

    leakage_columns = sorted(
        DIRECT_LEAKAGE_COLUMNS.intersection(historical.columns)
    )
    add_validation_check(
        checks,
        "No direct leakage columns",
        not leakage_columns,
        ", ".join(leakage_columns) if leakage_columns else "None",
        "None",
        "Status and termination fields are excluded from model features.",
    )

    historical_target_missing = int(
        historical["attrition_next_12m"].isna().sum()
    )
    current_target_present = int(
        current["attrition_next_12m"].notna().sum()
    )
    add_validation_check(
        checks,
        "Historical targets complete",
        historical_target_missing == 0,
        historical_target_missing,
        "0 missing",
        "Every historical row must have a known future outcome.",
    )
    add_validation_check(
        checks,
        "Current targets intentionally unknown",
        current_target_present == 0,
        current_target_present,
        "0 known targets",
        "Current employees cannot have a known future outcome.",
    )

    historical_target_values = set(
        historical["attrition_next_12m"].astype(int)
    )
    add_validation_check(
        checks,
        "Historical targets are binary",
        historical_target_values.issubset({0, 1}),
        sorted(historical_target_values),
        "[0, 1]",
        "Historical outcomes must use only zero and one.",
    )

    core_columns = [
        "employee_id",
        "snapshot_date",
        "employment_type",
        "organizational_level",
        "department_name",
        "city",
        "job_family",
        "job_level",
        "initial_base_salary",
        "base_salary",
    ]
    core_missing = int(
        historical[core_columns].isna().sum().sum()
        + current[core_columns].isna().sum().sum()
    )
    add_validation_check(
        checks,
        "Core features complete",
        core_missing == 0,
        core_missing,
        "0 missing core values",
        "Identifiers, organization, role, and salary must be present.",
    )

    cutoff_violations = int(
        cutoffs["latest_date_used"].gt(
            cutoffs["required_cutoff"]
        ).sum()
    )
    add_validation_check(
        checks,
        "Source dates respect snapshots",
        cutoff_violations == 0,
        cutoff_violations,
        "0 future-source cutoffs",
        "No source table may contribute records after a snapshot.",
    )

    audit_date_columns = [
        "latest_compensation_date",
        "latest_review_date",
        "latest_change_date",
        "latest_promotion_date",
    ]
    row_future_counts = {
        column: int(
            row_audit[column].gt(row_audit["snapshot_date"]).sum()
        )
        for column in audit_date_columns
    }
    row_future_total = sum(row_future_counts.values())
    add_validation_check(
        checks,
        "Row-level feature dates respect snapshots",
        row_future_total == 0,
        row_future_total,
        "0 future records",
        str(row_future_counts),
    )

    employed_violations = int(
        (
            row_audit["employee_hire_date"].gt(
                row_audit["snapshot_date"]
            )
            | (
                row_audit["employee_termination_date"].notna()
                & row_audit["employee_termination_date"].le(
                    row_audit["snapshot_date"]
                )
            )
        ).sum()
    )
    add_validation_check(
        checks,
        "Population employed at snapshot",
        employed_violations == 0,
        employed_violations,
        "0 eligibility violations",
        "Every row represents an employee active on its snapshot date.",
    )

    reconstructed_location_rows = int(
        row_audit["snapshot_location_id"]
        .ne(row_audit["final_location_id"])
        .sum()
    )
    add_validation_check(
        checks,
        "Historical locations reconstructed",
        reconstructed_location_rows > 0,
        reconstructed_location_rows,
        "> 0 rows",
        (
            "Hire and prior-transfer events reconstruct location without "
            "using later transfers as snapshot features."
        ),
    )

    snapshot_summary = (
        historical.groupby(
            [
                "snapshot_sequence",
                "snapshot_date",
                "prediction_end_date",
            ],
            as_index=False,
        )
        .agg(
            eligible_employees=("employee_id", "size"),
            positive_cases=("attrition_next_12m", "sum"),
            positive_rate=("attrition_next_12m", "mean"),
        )
        .sort_values("snapshot_sequence")
    )
    snapshot_summary["negative_cases"] = (
        snapshot_summary["eligible_employees"]
        - snapshot_summary["positive_cases"]
    )

    minimum_rows = int(
        snapshot_summary["eligible_employees"].min()
    )
    minimum_positives = int(snapshot_summary["positive_cases"].min())
    minimum_rate = float(snapshot_summary["positive_rate"].min())
    maximum_rate = float(snapshot_summary["positive_rate"].max())

    add_validation_check(
        checks,
        "Minimum rows per historical snapshot",
        minimum_rows
        >= validation["minimum_historical_rows_per_snapshot"],
        minimum_rows,
        f">= {validation['minimum_historical_rows_per_snapshot']}",
        "Every backtest period needs a usable evaluation population.",
    )
    add_validation_check(
        checks,
        "Minimum positive cases per snapshot",
        minimum_positives
        >= validation["minimum_positive_cases_per_snapshot"],
        minimum_positives,
        f">= {validation['minimum_positive_cases_per_snapshot']}",
        "Every period needs enough attrition cases for evaluation.",
    )
    add_validation_check(
        checks,
        "Historical positive-rate range",
        minimum_rate >= validation["minimum_positive_rate"]
        and maximum_rate <= validation["maximum_positive_rate"],
        f"{minimum_rate:.2%}–{maximum_rate:.2%}",
        (
            f"{validation['minimum_positive_rate']:.0%}–"
            f"{validation['maximum_positive_rate']:.0%}"
        ),
        "Period rates should remain plausible and nondegenerate.",
    )

    window_definitions = pd.DataFrame(definitions)
    window_definitions["snapshot_date"] = pd.to_datetime(
        window_definitions["snapshot_date"]
    )
    window_definitions["prediction_end_date"] = pd.to_datetime(
        window_definitions["prediction_end_date"]
    )
    window_definitions = window_definitions.sort_values("sequence")
    gaps = (
        window_definitions["snapshot_date"].iloc[1:].reset_index(drop=True)
        - window_definitions["prediction_end_date"]
        .iloc[:-1]
        .reset_index(drop=True)
    ).dt.days
    nonoverlapping = bool((gaps >= 0).all())
    add_validation_check(
        checks,
        "Prediction windows do not overlap",
        nonoverlapping,
        gaps.tolist(),
        "All gaps >= 0 days",
        "Consecutive target periods must not count the same day twice.",
    )

    employee_lookup = employees.set_index("employee_id")
    historical_employee = historical["employee_id"].map(
        employee_lookup["termination_date"]
    )
    expected_target = (
        historical_employee.notna()
        & historical_employee.gt(
            pd.to_datetime(historical["snapshot_date"])
        )
        & historical_employee.le(
            pd.to_datetime(historical["prediction_end_date"])
        )
    ).astype(int)
    target_mismatches = int(
        expected_target.ne(
            historical["attrition_next_12m"].astype(int)
        ).sum()
    )
    add_validation_check(
        checks,
        "Targets match termination windows",
        target_mismatches == 0,
        target_mismatches,
        "0 mismatches",
        "Labels are derived only from exits after each snapshot.",
    )

    as_of_date = pd.Timestamp(config["current_scoring"]["as_of_date"])
    expected_current_ids = set(
        employees.loc[
            employees["hire_date"].le(as_of_date)
            & (
                employees["termination_date"].isna()
                | employees["termination_date"].gt(as_of_date)
            ),
            "employee_id",
        ].astype(int)
    )
    actual_current_ids = set(current["employee_id"].astype(int))
    add_validation_check(
        checks,
        "Current active population reconciliation",
        actual_current_ids == expected_current_ids,
        len(actual_current_ids),
        f"{len(expected_current_ids)} exact active IDs",
        "Current scoring includes all and only active employees.",
    )

    repeated_employees = int(
        historical.groupby("employee_id")["snapshot_date"]
        .nunique()
        .gt(1)
        .sum()
    )
    add_validation_check(
        checks,
        "Repeated employees are identified",
        repeated_employees > 0,
        repeated_employees,
        "> 0",
        (
            "Repeated employees are expected in temporal panels and will "
            "require grouped robustness checks later."
        ),
    )

    validation_table = pd.DataFrame(checks)
    failures = validation_table.loc[
        validation_table["status"].eq("FAIL")
    ]
    if not failures.empty:
        raise ValueError(
            "Temporal dataset validation failed:\n"
            + failures.to_string(index=False)
        )

    panel_summary = pd.DataFrame(
        {
            "metric": [
                "Historical rows",
                "Historical unique employees",
                "Repeated historical employees",
                "Current active scoring rows",
                "Columns excluding target",
            ],
            "value": [
                len(historical),
                historical["employee_id"].nunique(),
                repeated_employees,
                len(current),
                len(MODEL_COLUMNS) - 1,
            ],
        }
    )

    return validation_table, snapshot_summary, panel_summary


def save_outputs(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    cutoffs: pd.DataFrame,
    validation_table: pd.DataFrame,
    snapshot_summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
) -> None:
    """Save modeling datasets and validation artifacts."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    historical.to_csv(HISTORICAL_OUTPUT, index=False)
    current.to_csv(CURRENT_OUTPUT, index=False)

    outputs = {
        "validation_checks.csv": validation_table,
        "snapshot_summary.csv": snapshot_summary,
        "panel_summary.csv": panel_summary,
        "feature_source_cutoffs.csv": cutoffs,
    }
    for filename, frame in outputs.items():
        frame.to_csv(VALIDATION_DIR / filename, index=False)

    hashes = pd.DataFrame(
        {
            "file": [
                str(HISTORICAL_OUTPUT.relative_to(PROJECT_ROOT)),
                str(CURRENT_OUTPUT.relative_to(PROJECT_ROOT)),
            ],
            "sha256": [
                sha256_file(HISTORICAL_OUTPUT),
                sha256_file(CURRENT_OUTPUT),
            ],
            "rows": [len(historical), len(current)],
            "columns": [len(historical.columns), len(current.columns)],
        }
    )
    hashes.to_csv(
        VALIDATION_DIR / "dataset_fingerprints.csv",
        index=False,
    )


def main() -> None:
    """Build, validate, and save all temporal datasets."""

    config = load_yaml(CONFIG_PATH)
    hazard_config = load_yaml(HAZARD_CONFIG_PATH)
    simulation_as_of_date = pd.Timestamp(
        hazard_config["simulation"]["as_of_date"]
    )
    employees = load_table("employees.csv")
    departments = load_table("departments.csv")
    locations = load_table("locations.csv")
    job_roles = load_table("job_roles.csv")
    compensation = load_table("compensation_history.csv")
    performance = load_table("performance_reviews.csv")
    training = load_table("training_records.csv")
    events = load_table("employee_events.csv")

    minimum_tenure_days = int(
        config["eligibility"]["minimum_tenure_days"]
    )
    historical_frames: list[pd.DataFrame] = []
    cutoff_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []

    for definition in config["historical_snapshots"]:
        snapshot, cutoffs, audit = build_one_snapshot(
            employees=employees,
            departments=departments,
            locations=locations,
            job_roles=job_roles,
            compensation=compensation,
            performance=performance,
            training=training,
            events=events,
            snapshot_date=pd.Timestamp(definition["snapshot_date"]),
            prediction_end_date=pd.Timestamp(
                definition["prediction_end_date"]
            ),
            sequence=int(definition["sequence"]),
            dataset_type="historical",
            minimum_tenure_days=minimum_tenure_days,
        )
        historical_frames.append(snapshot)
        cutoff_frames.append(cutoffs)
        audit_frames.append(audit)

    current_config = config["current_scoring"]
    current, current_cutoffs, current_audit = build_one_snapshot(
        employees=employees,
        departments=departments,
        locations=locations,
        job_roles=job_roles,
        compensation=compensation,
        performance=performance,
        training=training,
        events=events,
        snapshot_date=pd.Timestamp(current_config["as_of_date"]),
        prediction_end_date=None,
        sequence=int(current_config["sequence"]),
        dataset_type="current_scoring",
        minimum_tenure_days=minimum_tenure_days,
    )
    cutoff_frames.append(current_cutoffs)
    audit_frames.append(current_audit)

    historical = pd.concat(historical_frames, ignore_index=True)
    historical["attrition_next_12m"] = historical[
        "attrition_next_12m"
    ].astype(int)
    current["attrition_next_12m"] = current[
        "attrition_next_12m"
    ].astype("Int64")

    cutoffs = pd.concat(cutoff_frames, ignore_index=True)
    row_audit = pd.concat(audit_frames, ignore_index=True)

    validation_table, snapshot_summary, panel_summary = (
        validate_temporal_outputs(
            historical=historical,
            current=current,
            cutoffs=cutoffs,
            row_audit=row_audit,
            config=config,
            employees=employees,
            simulation_as_of_date=simulation_as_of_date,
        )
    )

    save_outputs(
        historical,
        current,
        cutoffs,
        validation_table,
        snapshot_summary,
        panel_summary,
    )

    print("\nTEMPORAL DATASET VALIDATION")
    print(validation_table.to_string(index=False))
    print("\nHISTORICAL SNAPSHOT SUMMARY")
    print(snapshot_summary.to_string(index=False))
    print("\nPANEL SUMMARY")
    print(panel_summary.to_string(index=False))
    print(f"\nSaved historical dataset: {HISTORICAL_OUTPUT}")
    print(f"Saved current scoring data: {CURRENT_OUTPUT}")
    print(
        "\nMULTI-SNAPSHOT TEMPORAL DATASETS COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
