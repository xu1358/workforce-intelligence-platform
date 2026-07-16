from datetime import date
from pathlib import Path
import math
import random

import pandas as pd


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AS_OF_DATE = date(2026, 6, 30)
RANDOM_SEED = 42

FIRST_COMPENSATION_ID = 500_001

ALLOWED_CHANGE_REASONS = {
    "Hire",
    "Annual Review",
    "Promotion",
    "Market Adjustment",
}


# A small synthetic location effect on starting salary.
LOCATION_PAY_ADJUSTMENTS = {
    1: 0.03,   # Austin
    2: 0.10,   # Fremont
    3: -0.02,  # Reno
    4: -0.05,  # Buffalo
    5: -0.01,  # Phoenix
}


# Starting position inside the job-role salary band.
ORGANIZATIONAL_LEVEL_PERCENTILES = {
    "Department Head": 0.86,
    "Senior Manager": 0.76,
    "Team Manager": 0.64,
    "Individual Contributor": 0.36,
}


# ---------------------------------------------------------
# File-loading helper
# ---------------------------------------------------------

def load_table(
    filename: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load one CSV file from the raw-data folder."""

    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {filename}."
        )

    return pd.read_csv(
        path,
        parse_dates=parse_dates,
    )


# ---------------------------------------------------------
# Date and rounding helpers
# ---------------------------------------------------------

def add_years(
    original_date: date,
    number_of_years: int,
) -> date:
    """
    Add complete years to a date.

    February 29 becomes February 28 in a non-leap year.
    """

    try:
        return original_date.replace(
            year=(
                original_date.year
                + number_of_years
            )
        )
    except ValueError:
        return original_date.replace(
            year=(
                original_date.year
                + number_of_years
            ),
            day=28,
        )


def round_to_nearest_100(
    value: float,
) -> int:
    """Round a salary to the nearest 100 dollars."""

    return int(
        round(value / 100) * 100
    )


def round_to_nearest_500(
    value: float,
) -> int:
    """Round equity to the nearest 500 dollars."""

    return int(
        round(value / 500) * 500
    )


# ---------------------------------------------------------
# Initial compensation
# ---------------------------------------------------------

def choose_initial_salary(
    salary_band_min: float,
    salary_band_max: float,
    job_level: int,
    organizational_level: str,
    location_id: int,
) -> int:
    """Choose a starting salary inside the job-role band."""

    percentile = (
        ORGANIZATIONAL_LEVEL_PERCENTILES[
            organizational_level
        ]
    )

    # Higher job levels receive a modest adjustment.
    percentile += (
        job_level - 1
    ) * 0.025

    percentile += (
        LOCATION_PAY_ADJUSTMENTS[
            location_id
        ]
    )

    percentile += random.uniform(
        -0.07,
        0.07,
    )

    percentile = max(
        0.08,
        min(percentile, 0.95),
    )

    salary = (
        salary_band_min
        + percentile
        * (
            salary_band_max
            - salary_band_min
        )
    )

    rounded_salary = (
        round_to_nearest_100(salary)
    )

    minimum_salary = int(
        math.ceil(
            salary_band_min / 100
        )
        * 100
    )

    maximum_salary = int(
        math.floor(
            salary_band_max / 100
        )
        * 100
    )

    return max(
        minimum_salary,
        min(
            rounded_salary,
            maximum_salary,
        ),
    )


def choose_bonus_target(
    job_level: int,
    organizational_level: str,
    employment_type: str,
) -> float:
    """
    Choose a target bonus percentage.

    A value of 10.0 means a 10% target bonus.
    """

    bonus_by_job_level = {
        1: 5.0,
        2: 8.0,
        3: 12.0,
        4: 15.0,
    }

    organizational_addition = {
        "Individual Contributor": 0.0,
        "Team Manager": 3.0,
        "Senior Manager": 7.0,
        "Department Head": 12.0,
    }

    bonus_target = (
        bonus_by_job_level[job_level]
        + organizational_addition[
            organizational_level
        ]
    )

    if (
        employment_type == "Hourly"
        and organizational_level
        == "Individual Contributor"
    ):
        bonus_target -= 3.0

    bonus_target += random.choice(
        [-1.0, 0.0, 1.0]
    )

    bonus_target = max(
        0.0,
        min(bonus_target, 35.0),
    )

    return round(
        bonus_target,
        1,
    )


def choose_equity_value(
    job_level: int,
    organizational_level: str,
    employment_type: str,
) -> int:
    """Choose estimated annual equity value."""

    equity_by_job_level = {
        1: 1_500,
        2: 4_000,
        3: 10_000,
        4: 20_000,
    }

    organizational_multiplier = {
        "Individual Contributor": 1.0,
        "Team Manager": 1.6,
        "Senior Manager": 2.6,
        "Department Head": 4.2,
    }

    equity_value = (
        equity_by_job_level[job_level]
        * organizational_multiplier[
            organizational_level
        ]
    )

    if (
        employment_type == "Hourly"
        and organizational_level
        == "Individual Contributor"
    ):
        equity_value *= 0.30

    equity_value *= random.uniform(
        0.80,
        1.20,
    )

    return max(
        0,
        round_to_nearest_500(
            equity_value
        ),
    )


# ---------------------------------------------------------
# Annual compensation changes
# ---------------------------------------------------------

def choose_change_reason(
    organizational_level: str,
) -> str:
    """Choose a reason for an annual compensation change."""

    if organizational_level == "Department Head":
        reasons = [
            "Annual Review",
            "Market Adjustment",
        ]

        weights = [
            0.90,
            0.10,
        ]
    elif organizational_level == "Senior Manager":
        reasons = [
            "Annual Review",
            "Promotion",
            "Market Adjustment",
        ]

        weights = [
            0.86,
            0.04,
            0.10,
        ]
    else:
        reasons = [
            "Annual Review",
            "Promotion",
            "Market Adjustment",
        ]

        weights = [
            0.80,
            0.10,
            0.10,
        ]

    return random.choices(
        reasons,
        weights=weights,
        k=1,
    )[0]


def choose_salary_increase(
    change_reason: str,
) -> float:
    """Choose a raise percentage from the change reason."""

    if change_reason == "Annual Review":
        return random.uniform(
            0.025,
            0.060,
        )

    if change_reason == "Promotion":
        return random.uniform(
            0.080,
            0.140,
        )

    if change_reason == "Market Adjustment":
        return random.uniform(
            0.040,
            0.090,
        )

    raise ValueError(
        f"Unknown change reason: "
        f"{change_reason}"
    )


def calculate_new_salary(
    previous_salary: int,
    salary_band_max: float,
    change_reason: str,
) -> int:
    """Apply an increase without exceeding the allowed cap."""

    increase_rate = (
        choose_salary_increase(
            change_reason
        )
    )

    proposed_salary = (
        previous_salary
        * (1 + increase_rate)
    )

    # Salaries may move modestly above the current band
    # after several historical increases.
    salary_cap = int(
        math.floor(
            (
                salary_band_max
                * 1.08
            )
            / 100
        )
        * 100
    )

    new_salary = min(
        round_to_nearest_100(
            proposed_salary
        ),
        salary_cap,
    )

    return max(
        previous_salary,
        new_salary,
    )


def calculate_new_bonus_target(
    previous_bonus_target: float,
    change_reason: str,
) -> float:
    """Update the target bonus percentage."""

    if change_reason == "Promotion":
        increase = random.choice(
            [1.5, 2.0, 2.5, 3.0]
        )
    elif change_reason == "Market Adjustment":
        increase = random.choice(
            [0.0, 0.5, 1.0]
        )
    else:
        increase = random.choice(
            [0.0, 0.0, 0.5]
        )

    return round(
        min(
            previous_bonus_target
            + increase,
            40.0,
        ),
        1,
    )


def calculate_new_equity_value(
    previous_equity_value: int,
    change_reason: str,
) -> int:
    """Update estimated annual equity."""

    if change_reason == "Promotion":
        multiplier = random.uniform(
            1.15,
            1.30,
        )
    elif change_reason == "Market Adjustment":
        multiplier = random.uniform(
            1.05,
            1.15,
        )
    else:
        multiplier = random.uniform(
            1.02,
            1.08,
        )

    return max(
        previous_equity_value,
        round_to_nearest_500(
            previous_equity_value
            * multiplier
        ),
    )


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------

def create_compensation_history(
    employees: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> pd.DataFrame:
    """Create compensation records for every employee."""

    random.seed(RANDOM_SEED)

    role_lookup = (
        job_roles
        .set_index("job_role_id")
        .to_dict(orient="index")
    )

    compensation_records = []

    next_compensation_id = (
        FIRST_COMPENSATION_ID
    )

    employees = employees.sort_values(
        "employee_id"
    )

    for employee in employees.itertuples(
        index=False
    ):
        employee_id = int(
            employee.employee_id
        )

        job_role_id = int(
            employee.job_role_id
        )

        role_information = (
            role_lookup[job_role_id]
        )

        job_level = int(
            role_information[
                "job_level"
            ]
        )

        salary_band_min = float(
            role_information[
                "salary_band_min"
            ]
        )

        salary_band_max = float(
            role_information[
                "salary_band_max"
            ]
        )

        hire_date = pd.Timestamp(
            employee.hire_date
        ).date()

        if pd.isna(
            employee.termination_date
        ):
            employee_end_date = (
                AS_OF_DATE
            )
        else:
            employee_end_date = (
                pd.Timestamp(
                    employee.termination_date
                ).date()
            )

        base_salary = (
            choose_initial_salary(
                salary_band_min=(
                    salary_band_min
                ),
                salary_band_max=(
                    salary_band_max
                ),
                job_level=job_level,
                organizational_level=(
                    employee.organizational_level
                ),
                location_id=int(
                    employee.location_id
                ),
            )
        )

        bonus_target = (
            choose_bonus_target(
                job_level=job_level,
                organizational_level=(
                    employee.organizational_level
                ),
                employment_type=(
                    employee.employment_type
                ),
            )
        )

        equity_value = (
            choose_equity_value(
                job_level=job_level,
                organizational_level=(
                    employee.organizational_level
                ),
                employment_type=(
                    employee.employment_type
                ),
            )
        )

        # Every employee receives one record
        # on their hire date.
        compensation_records.append(
            {
                "compensation_id": (
                    next_compensation_id
                ),
                "employee_id": employee_id,
                "effective_date": hire_date,
                "base_salary": base_salary,
                "bonus_target": bonus_target,
                "equity_value": equity_value,
                "change_reason": "Hire",
            }
        )

        next_compensation_id += 1

        completed_years = 1

        while True:
            effective_date = add_years(
                hire_date,
                completed_years,
            )

            if (
                effective_date
                > employee_end_date
            ):
                break

            change_reason = (
                choose_change_reason(
                    employee.organizational_level
                )
            )

            base_salary = (
                calculate_new_salary(
                    previous_salary=(
                        base_salary
                    ),
                    salary_band_max=(
                        salary_band_max
                    ),
                    change_reason=(
                        change_reason
                    ),
                )
            )

            bonus_target = (
                calculate_new_bonus_target(
                    previous_bonus_target=(
                        bonus_target
                    ),
                    change_reason=(
                        change_reason
                    ),
                )
            )

            equity_value = (
                calculate_new_equity_value(
                    previous_equity_value=(
                        equity_value
                    ),
                    change_reason=(
                        change_reason
                    ),
                )
            )

            compensation_records.append(
                {
                    "compensation_id": (
                        next_compensation_id
                    ),
                    "employee_id": (
                        employee_id
                    ),
                    "effective_date": (
                        effective_date
                    ),
                    "base_salary": (
                        base_salary
                    ),
                    "bonus_target": (
                        bonus_target
                    ),
                    "equity_value": (
                        equity_value
                    ),
                    "change_reason": (
                        change_reason
                    ),
                }
            )

            next_compensation_id += 1
            completed_years += 1

    compensation_history = (
        pd.DataFrame(
            compensation_records
        )
    )

    compensation_history[
        "effective_date"
    ] = pd.to_datetime(
        compensation_history[
            "effective_date"
        ]
    )

    return compensation_history


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_compensation_history(
    compensation_history: pd.DataFrame,
    employees: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> None:
    """Validate compensation-history rules."""

    expected_columns = {
        "compensation_id",
        "employee_id",
        "effective_date",
        "base_salary",
        "bonus_target",
        "equity_value",
        "change_reason",
    }

    if (
        set(
            compensation_history.columns
        )
        != expected_columns
    ):
        raise ValueError(
            "compensation_history: "
            "unexpected columns."
        )

    if compensation_history[
        "compensation_id"
    ].isna().any():
        raise ValueError(
            "compensation_history: "
            "missing compensation IDs."
        )

    if not compensation_history[
        "compensation_id"
    ].is_unique:
        raise ValueError(
            "compensation_history: "
            "compensation IDs are not unique."
        )

    valid_employee_ids = set(
        employees["employee_id"]
        .astype(int)
    )

    used_employee_ids = set(
        compensation_history[
            "employee_id"
        ].astype(int)
    )

    if used_employee_ids != valid_employee_ids:
        raise ValueError(
            "compensation_history: "
            "not every employee has records."
        )

    if not set(
        compensation_history[
            "change_reason"
        ]
    ).issubset(
        ALLOWED_CHANGE_REASONS
    ):
        raise ValueError(
            "compensation_history: "
            "invalid change reason."
        )

    if (
        compensation_history[
            "base_salary"
        ]
        <= 0
    ).any():
        raise ValueError(
            "compensation_history: "
            "base salary must be positive."
        )

    if not compensation_history[
        "bonus_target"
    ].between(
        0,
        40,
    ).all():
        raise ValueError(
            "compensation_history: "
            "bonus target is outside "
            "the allowed range."
        )

    if (
        compensation_history[
            "equity_value"
        ]
        < 0
    ).any():
        raise ValueError(
            "compensation_history: "
            "equity value cannot be negative."
        )

    if compensation_history.duplicated(
        subset=[
            "employee_id",
            "effective_date",
        ]
    ).any():
        raise ValueError(
            "compensation_history: "
            "duplicate employee-date records."
        )

    sorted_history = (
        compensation_history
        .sort_values(
            [
                "employee_id",
                "effective_date",
            ]
        )
        .copy()
    )

    sorted_history[
        "previous_salary"
    ] = (
        sorted_history
        .groupby("employee_id")[
            "base_salary"
        ]
        .shift(1)
    )

    salary_decreases = (
        sorted_history[
            "previous_salary"
        ].notna()
        & (
            sorted_history[
                "base_salary"
            ]
            < sorted_history[
                "previous_salary"
            ]
        )
    )

    if salary_decreases.any():
        raise ValueError(
            "compensation_history: "
            "salary decreases were found."
        )

    employee_dates = employees[
        [
            "employee_id",
            "hire_date",
            "termination_date",
            "job_role_id",
        ]
    ].copy()

    employee_dates[
        "employment_end_date"
    ] = employee_dates[
        "termination_date"
    ].fillna(
        pd.Timestamp(
            AS_OF_DATE
        )
    )

    detailed_history = (
        compensation_history
        .merge(
            employee_dates,
            on="employee_id",
            how="left",
        )
        .merge(
            job_roles[
                [
                    "job_role_id",
                    "salary_band_min",
                    "salary_band_max",
                ]
            ],
            on="job_role_id",
            how="left",
        )
    )

    if (
        detailed_history[
            "effective_date"
        ]
        < detailed_history[
            "hire_date"
        ]
    ).any():
        raise ValueError(
            "compensation_history: "
            "a record occurs before hire."
        )

    if (
        detailed_history[
            "effective_date"
        ]
        > detailed_history[
            "employment_end_date"
        ]
    ).any():
        raise ValueError(
            "compensation_history: "
            "a record occurs after employment ends."
        )

    if (
        detailed_history[
            "base_salary"
        ]
        < detailed_history[
            "salary_band_min"
        ]
    ).any():
        raise ValueError(
            "compensation_history: "
            "salary is below the role band."
        )

    if (
        detailed_history[
            "base_salary"
        ]
        > (
            detailed_history[
                "salary_band_max"
            ]
            * 1.08
        )
    ).any():
        raise ValueError(
            "compensation_history: "
            "salary exceeds the allowed cap."
        )

    first_records = (
        sorted_history
        .groupby(
            "employee_id",
            as_index=False,
        )
        .first()
        .merge(
            employees[
                [
                    "employee_id",
                    "hire_date",
                ]
            ],
            on="employee_id",
            how="left",
        )
    )

    if not first_records[
        "change_reason"
    ].eq("Hire").all():
        raise ValueError(
            "compensation_history: "
            "an employee's first record "
            "is not a hire record."
        )

    if not first_records[
        "effective_date"
    ].eq(
        first_records[
            "hire_date"
        ]
    ).all():
        raise ValueError(
            "compensation_history: "
            "a hire record date does not "
            "match the hire date."
        )

    hire_record_counts = (
        compensation_history[
            compensation_history[
                "change_reason"
            ]
            == "Hire"
        ]
        .groupby("employee_id")
        .size()
    )

    if not hire_record_counts.eq(1).all():
        raise ValueError(
            "compensation_history: "
            "each employee must have "
            "exactly one hire record."
        )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_compensation_history(
    compensation_history: pd.DataFrame,
) -> None:
    """Save compensation history as a CSV."""

    output_path = (
        RAW_DATA_DIR
        / "compensation_history.csv"
    )

    compensation_history.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved compensation_history.csv: "
        f"{len(compensation_history)} rows and "
        f"{len(compensation_history.columns)} columns"
    )


def print_summary(
    compensation_history: pd.DataFrame,
) -> None:
    """Print a basic compensation summary."""

    print("\nRecords by change reason:")
    print(
        compensation_history[
            "change_reason"
        ].value_counts()
    )

    records_per_employee = (
        compensation_history
        .groupby("employee_id")
        .size()
    )

    print("\nEmployees with compensation records:")
    print(
        records_per_employee.size
    )

    print("\nAverage records per employee:")
    print(
        round(
            records_per_employee.mean(),
            2,
        )
    )

    latest_records = (
        compensation_history
        .sort_values(
            [
                "employee_id",
                "effective_date",
            ]
        )
        .groupby("employee_id")
        .tail(1)
    )

    print("\nAverage latest base salary:")
    print(
        round(
            latest_records[
                "base_salary"
            ].mean(),
            2,
        )
    )


def main() -> None:
    """Generate and validate compensation history."""

    employees = load_table(
        "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    job_roles = load_table(
        "job_roles.csv"
    )

    compensation_history = (
        create_compensation_history(
            employees,
            job_roles,
        )
    )

    validate_compensation_history(
        compensation_history,
        employees,
        job_roles,
    )

    save_compensation_history(
        compensation_history
    )

    print_summary(
        compensation_history
    )

    print(
        "\nCompensation history generated "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()