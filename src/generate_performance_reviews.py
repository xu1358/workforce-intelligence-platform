from datetime import date
from pathlib import Path
import random

import pandas as pd


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AS_OF_DATE = date(2026, 6, 30)
RANDOM_SEED = 42

FIRST_REVIEW_ID = 700_001

MANAGER_LEVELS = {
    "Department Head",
    "Senior Manager",
    "Team Manager",
}


# Small synthetic department effects.
DEPARTMENT_RATING_ADJUSTMENTS = {
    1: 0.08,
    2: -0.05,
    3: 0.02,
    4: 0.04,
    5: 0.06,
    6: 0.03,
    7: 0.05,
    8: -0.08,
}


# Small synthetic organizational-level effects.
ORGANIZATIONAL_LEVEL_ADJUSTMENTS = {
    "Department Head": 0.20,
    "Senior Manager": 0.16,
    "Team Manager": 0.10,
    "Individual Contributor": 0.00,
}


# ---------------------------------------------------------
# File-loading helper
# ---------------------------------------------------------

def load_table(
    filename: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load one CSV table from the raw-data folder."""

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
# General helpers
# ---------------------------------------------------------

def add_years(
    original_date: date,
    number_of_years: int,
) -> date:
    """
    Add complete years to a date.

    February 29 becomes February 28 when the
    destination year is not a leap year.
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


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Keep a numeric value inside a defined range."""

    return max(
        minimum,
        min(value, maximum),
    )


# ---------------------------------------------------------
# Performance-score generation
# ---------------------------------------------------------

def choose_employee_baseline(
    department_id: int,
    organizational_level: str,
) -> float:
    """Create an outcome-independent performance tendency."""

    baseline = random.gauss(
        3.45,
        0.45,
    )

    baseline += (
        DEPARTMENT_RATING_ADJUSTMENTS[
            department_id
        ]
    )

    baseline += (
        ORGANIZATIONAL_LEVEL_ADJUSTMENTS[
            organizational_level
        ]
    )

    return clamp(
        baseline,
        1.5,
        4.8,
    )


def choose_performance_rating(
    employee_baseline: float,
    review_number: int,
    reviewer_id: int,
) -> float:
    """Generate a rating between 1.0 and 5.0."""

    # A small positive development trend over time.
    experience_effect = min(
        0.03 * (review_number - 1),
        0.12,
    )

    # A small stable reviewer effect.
    reviewer_bias = (
        (
            reviewer_id % 7
        )
        - 3
    ) * 0.025

    annual_noise = random.gauss(
        0,
        0.24,
    )

    rating = (
        employee_baseline
        + experience_effect
        + reviewer_bias
        + annual_noise
    )

    return round(
        clamp(
            rating,
            1.0,
            5.0,
        ),
        1,
    )


def choose_goal_completion(
    performance_rating: float,
) -> float:
    """Generate goal completion correlated with rating."""

    goal_completion = (
        43
        + performance_rating * 14
        + random.gauss(
            0,
            7,
        )
    )

    return round(
        clamp(
            goal_completion,
            0,
            120,
        ),
        1,
    )


def choose_promotion_recommendation(
    performance_rating: float,
    goal_completion: float,
    organizational_level: str,
) -> bool:
    """Determine whether promotion is recommended."""

    if organizational_level == "Department Head":
        return False

    if (
        performance_rating >= 4.3
        and goal_completion >= 95
    ):
        probability = 0.55

    elif (
        performance_rating >= 3.8
        and goal_completion >= 85
    ):
        probability = 0.20

    elif (
        performance_rating >= 3.4
        and goal_completion >= 75
    ):
        probability = 0.05

    else:
        probability = 0.01

    return random.random() < probability


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------

def create_performance_reviews(
    employees: pd.DataFrame,
) -> pd.DataFrame:
    """Create annual performance reviews."""

    random.seed(RANDOM_SEED)

    employees = employees.sort_values(
        "employee_id"
    ).copy()

    employees["manager_id"] = (
        employees["manager_id"]
        .astype("Int64")
    )

    employee_lookup = (
        employees
        .set_index("employee_id")
    )

    review_records = []

    next_review_id = FIRST_REVIEW_ID

    for employee in employees.itertuples(
        index=False
    ):
        # Department heads currently have no manager.
        # Therefore, they do not receive reviews yet.
        if pd.isna(employee.manager_id):
            continue

        employee_id = int(
            employee.employee_id
        )

        reviewer_id = int(
            employee.manager_id
        )

        reviewer_information = (
            employee_lookup.loc[
                reviewer_id
            ]
        )

        hire_date = pd.Timestamp(
            employee.hire_date
        ).date()

        reviewer_hire_date = pd.Timestamp(
            reviewer_information[
                "hire_date"
            ]
        ).date()

        if pd.isna(
            employee.termination_date
        ):
            employment_end_date = (
                AS_OF_DATE
            )
        else:
            employment_end_date = (
                pd.Timestamp(
                    employee.termination_date
                ).date()
            )

        employee_baseline = (
            choose_employee_baseline(
                department_id=int(
                    employee.department_id
                ),
                organizational_level=(
                    employee.organizational_level
                ),
            )
        )

        completed_years = 1
        review_number = 1

        while True:
            review_date = add_years(
                hire_date,
                completed_years,
            )

            if (
                review_date
                > employment_end_date
            ):
                break

            # We use the employee's current manager as
            # reviewer. Reviews before that manager's
            # hire date are skipped.
            if (
                review_date
                < reviewer_hire_date
            ):
                completed_years += 1
                continue

            performance_rating = (
                choose_performance_rating(
                    employee_baseline=(
                        employee_baseline
                    ),
                    review_number=(
                        review_number
                    ),
                    reviewer_id=(
                        reviewer_id
                    ),
                )
            )

            goal_completion = (
                choose_goal_completion(
                    performance_rating
                )
            )

            promotion_recommended = (
                choose_promotion_recommendation(
                    performance_rating=(
                        performance_rating
                    ),
                    goal_completion=(
                        goal_completion
                    ),
                    organizational_level=(
                        employee.organizational_level
                    ),
                )
            )

            review_records.append(
                {
                    "review_id": (
                        next_review_id
                    ),
                    "employee_id": (
                        employee_id
                    ),
                    "review_date": (
                        review_date
                    ),
                    "review_period": (
                        f"{review_date.year} Annual"
                    ),
                    "performance_rating": (
                        performance_rating
                    ),
                    "goal_completion": (
                        goal_completion
                    ),
                    "promotion_recommended": (
                        promotion_recommended
                    ),
                    "reviewer_id": (
                        reviewer_id
                    ),
                }
            )

            next_review_id += 1
            completed_years += 1
            review_number += 1

    performance_reviews = pd.DataFrame(
        review_records
    )

    performance_reviews[
        "review_date"
    ] = pd.to_datetime(
        performance_reviews[
            "review_date"
        ]
    )

    return performance_reviews


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_performance_reviews(
    performance_reviews: pd.DataFrame,
    employees: pd.DataFrame,
) -> None:
    """Validate performance-review rules."""

    expected_columns = {
        "review_id",
        "employee_id",
        "review_date",
        "review_period",
        "performance_rating",
        "goal_completion",
        "promotion_recommended",
        "reviewer_id",
    }

    if (
        set(
            performance_reviews.columns
        )
        != expected_columns
    ):
        raise ValueError(
            "performance_reviews: "
            "unexpected columns."
        )

    if performance_reviews.empty:
        raise ValueError(
            "performance_reviews: "
            "no records were generated."
        )

    if performance_reviews[
        "review_id"
    ].isna().any():
        raise ValueError(
            "performance_reviews: "
            "missing review IDs."
        )

    if not performance_reviews[
        "review_id"
    ].is_unique:
        raise ValueError(
            "performance_reviews: "
            "review IDs are not unique."
        )

    if performance_reviews.duplicated(
        subset=[
            "employee_id",
            "review_period",
        ]
    ).any():
        raise ValueError(
            "performance_reviews: "
            "duplicate employee-period records."
        )

    valid_employee_ids = set(
        employees["employee_id"]
        .astype(int)
    )

    reviewed_employee_ids = set(
        performance_reviews[
            "employee_id"
        ].astype(int)
    )

    reviewer_ids = set(
        performance_reviews[
            "reviewer_id"
        ].astype(int)
    )

    if not reviewed_employee_ids.issubset(
        valid_employee_ids
    ):
        raise ValueError(
            "performance_reviews: "
            "invalid employee ID."
        )

    if not reviewer_ids.issubset(
        valid_employee_ids
    ):
        raise ValueError(
            "performance_reviews: "
            "invalid reviewer ID."
        )

    if (
        performance_reviews[
            "employee_id"
        ]
        == performance_reviews[
            "reviewer_id"
        ]
    ).any():
        raise ValueError(
            "performance_reviews: "
            "an employee reviewed themselves."
        )

    if not performance_reviews[
        "performance_rating"
    ].between(
        1.0,
        5.0,
    ).all():
        raise ValueError(
            "performance_reviews: "
            "rating is outside 1.0 to 5.0."
        )

    if not performance_reviews[
        "goal_completion"
    ].between(
        0,
        120,
    ).all():
        raise ValueError(
            "performance_reviews: "
            "goal completion is outside "
            "0 to 120."
        )

    if not performance_reviews[
        "promotion_recommended"
    ].isin(
        [
            True,
            False,
        ]
    ).all():
        raise ValueError(
            "performance_reviews: "
            "promotion recommendation "
            "must be boolean."
        )

    employee_details = (
        employees[
            [
                "employee_id",
                "hire_date",
                "termination_date",
                "department_id",
                "manager_id",
                "organizational_level",
            ]
        ]
        .rename(
            columns={
                "hire_date": (
                    "employee_hire_date"
                ),
                "termination_date": (
                    "employee_termination_date"
                ),
                "department_id": (
                    "employee_department_id"
                ),
                "manager_id": (
                    "expected_reviewer_id"
                ),
                "organizational_level": (
                    "employee_level"
                ),
            }
        )
    )

    reviewer_details = (
        employees[
            [
                "employee_id",
                "hire_date",
                "department_id",
                "employment_status",
                "organizational_level",
            ]
        ]
        .rename(
            columns={
                "employee_id": (
                    "reviewer_id"
                ),
                "hire_date": (
                    "reviewer_hire_date"
                ),
                "department_id": (
                    "reviewer_department_id"
                ),
                "employment_status": (
                    "reviewer_status"
                ),
                "organizational_level": (
                    "reviewer_level"
                ),
            }
        )
    )

    detailed_reviews = (
        performance_reviews
        .merge(
            employee_details,
            on="employee_id",
            how="left",
        )
        .merge(
            reviewer_details,
            on="reviewer_id",
            how="left",
        )
    )

    detailed_reviews[
        "employment_end_date"
    ] = detailed_reviews[
        "employee_termination_date"
    ].fillna(
        pd.Timestamp(
            AS_OF_DATE
        )
    )

    if (
        detailed_reviews[
            "review_date"
        ]
        < detailed_reviews[
            "employee_hire_date"
        ]
    ).any():
        raise ValueError(
            "performance_reviews: "
            "a review occurs before hire."
        )

    if (
        detailed_reviews[
            "review_date"
        ]
        > detailed_reviews[
            "employment_end_date"
        ]
    ).any():
        raise ValueError(
            "performance_reviews: "
            "a review occurs after employment ends."
        )

    first_anniversary = (
        detailed_reviews[
            "employee_hire_date"
        ]
        + pd.DateOffset(
            years=1
        )
    )

    if (
        detailed_reviews[
            "review_date"
        ]
        < first_anniversary
    ).any():
        raise ValueError(
            "performance_reviews: "
            "a review occurs before one year "
            "of employment."
        )

    if (
        detailed_reviews[
            "review_date"
        ]
        < detailed_reviews[
            "reviewer_hire_date"
        ]
    ).any():
        raise ValueError(
            "performance_reviews: "
            "a reviewer was not yet employed."
        )

    if not detailed_reviews[
        "reviewer_id"
    ].eq(
        detailed_reviews[
            "expected_reviewer_id"
        ]
    ).all():
        raise ValueError(
            "performance_reviews: "
            "reviewer does not match "
            "the current manager."
        )

    if not detailed_reviews[
        "employee_department_id"
    ].eq(
        detailed_reviews[
            "reviewer_department_id"
        ]
    ).all():
        raise ValueError(
            "performance_reviews: "
            "employee and reviewer are "
            "in different departments."
        )

    if not detailed_reviews[
        "reviewer_status"
    ].eq("Active").all():
        raise ValueError(
            "performance_reviews: "
            "a reviewer is inactive."
        )

    if not detailed_reviews[
        "reviewer_level"
    ].isin(
        MANAGER_LEVELS
    ).all():
        raise ValueError(
            "performance_reviews: "
            "a reviewer is not a manager."
        )

    expected_period = (
        detailed_reviews[
            "review_date"
        ]
        .dt.year
        .astype(str)
        + " Annual"
    )

    if not detailed_reviews[
        "review_period"
    ].eq(
        expected_period
    ).all():
        raise ValueError(
            "performance_reviews: "
            "review period does not match "
            "the review date."
        )

    department_head_ids = set(
        employees.loc[
            employees[
                "organizational_level"
            ]
            == "Department Head",
            "employee_id",
        ].astype(int)
    )

    if reviewed_employee_ids.intersection(
        department_head_ids
    ):
        raise ValueError(
            "performance_reviews: "
            "department heads should not "
            "have reviews yet."
        )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_performance_reviews(
    performance_reviews: pd.DataFrame,
) -> None:
    """Save performance reviews as a CSV."""

    output_path = (
        RAW_DATA_DIR
        / "performance_reviews.csv"
    )

    performance_reviews.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved performance_reviews.csv: "
        f"{len(performance_reviews)} rows and "
        f"{len(performance_reviews.columns)} columns"
    )


def print_summary(
    performance_reviews: pd.DataFrame,
    employees: pd.DataFrame,
) -> None:
    """Print a basic review summary."""

    employees_with_reviews = (
        performance_reviews[
            "employee_id"
        ].nunique()
    )

    print(
        "\nEmployees with reviews:"
    )
    print(
        employees_with_reviews
    )

    print(
        "\nEmployees without reviews:"
    )
    print(
        len(employees)
        - employees_with_reviews
    )

    print(
        "\nReviews by period:"
    )
    print(
        performance_reviews[
            "review_period"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nAverage performance rating:"
    )
    print(
        round(
            performance_reviews[
                "performance_rating"
            ].mean(),
            2,
        )
    )

    print(
        "\nPromotion recommendations:"
    )
    print(
        performance_reviews[
            "promotion_recommended"
        ].value_counts()
    )


def main() -> None:
    """Generate and validate performance reviews."""

    employees = load_table(
        "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    performance_reviews = (
        create_performance_reviews(
            employees
        )
    )

    validate_performance_reviews(
        performance_reviews,
        employees,
    )

    save_performance_reviews(
        performance_reviews
    )

    print_summary(
        performance_reviews,
        employees,
    )

    print(
        "\nPerformance reviews generated "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()
