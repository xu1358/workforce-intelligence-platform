from datetime import date, timedelta
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

FIRST_TRAINING_RECORD_ID = 900_001


MANAGER_LEVELS = {
    "Department Head",
    "Senior Manager",
    "Team Manager",
}


SAFETY_DEPARTMENTS = {
    2,  # Manufacturing
    3,  # Supply Chain
}


TECHNICAL_DEPARTMENTS = {
    1,  # Engineering
    2,  # Manufacturing
    3,  # Supply Chain
    7,  # Information Technology
}


REQUIRED_PROGRAM_CATEGORIES = {
    "Onboarding",
    "Safety",
    "Technical",
    "Leadership",
}


ALLOWED_COMPLETION_STATUSES = {
    "Completed",
    "Incomplete",
    "Failed",
}


# ---------------------------------------------------------
# File-loading helper
# ---------------------------------------------------------

def load_table(
    filename: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file from the raw-data folder."""

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
# General helper functions
# ---------------------------------------------------------

def random_date(
    start_date: date,
    end_date: date,
) -> date:
    """Generate one random date inside a date range."""

    if start_date > end_date:
        raise ValueError(
            "start_date cannot occur after end_date."
        )

    total_days = (
        end_date - start_date
    ).days

    random_days = random.randint(
        0,
        total_days,
    )

    return start_date + timedelta(
        days=random_days
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Keep a numeric value inside a range."""

    return max(
        minimum,
        min(value, maximum),
    )


def normalize_boolean(
    value,
) -> bool:
    """Convert CSV boolean values safely."""

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return bool(value)


# ---------------------------------------------------------
# Program lookup
# ---------------------------------------------------------

def build_program_category_lookup(
    training_programs: pd.DataFrame,
) -> dict[str, list[int]]:
    """Group program IDs by program category."""

    program_ids_by_category = (
        training_programs
        .groupby("program_category")[
            "program_id"
        ]
        .apply(
            lambda values: [
                int(value)
                for value in values
            ]
        )
        .to_dict()
    )

    missing_categories = (
        REQUIRED_PROGRAM_CATEGORIES
        - set(
            program_ids_by_category
        )
    )

    if missing_categories:
        raise ValueError(
            "training_programs: missing required "
            f"categories: {sorted(missing_categories)}"
        )

    return program_ids_by_category


def choose_unused_program(
    category: str,
    program_ids_by_category: dict[str, list[int]],
    selected_program_ids: set[int],
) -> int | None:
    """Choose one program not already assigned."""

    available_program_ids = [
        program_id
        for program_id
        in program_ids_by_category[
            category
        ]
        if program_id
        not in selected_program_ids
    ]

    if not available_program_ids:
        return None

    return random.choice(
        available_program_ids
    )


# ---------------------------------------------------------
# Employee-program assignment
# ---------------------------------------------------------

def select_employee_programs(
    employee,
    program_ids_by_category: dict[str, list[int]],
    employment_end_date: date,
) -> list[int]:
    """Choose appropriate training programs for one employee."""

    selected_program_ids = set()

    hire_date = pd.Timestamp(
        employee.hire_date
    ).date()

    employment_days = (
        employment_end_date
        - hire_date
    ).days

    # Every employee receives exactly one
    # onboarding program.
    onboarding_program_id = (
        choose_unused_program(
            category="Onboarding",
            program_ids_by_category=(
                program_ids_by_category
            ),
            selected_program_ids=(
                selected_program_ids
            ),
        )
    )

    if onboarding_program_id is None:
        raise ValueError(
            "No onboarding program is available."
        )

    selected_program_ids.add(
        onboarding_program_id
    )

    # Safety training applies to:
    # - Manufacturing
    # - Supply Chain
    # - All hourly employees
    safety_required = (
        int(employee.department_id)
        in SAFETY_DEPARTMENTS
        or employee.employment_type
        == "Hourly"
    )

    if safety_required:
        safety_program_id = (
            choose_unused_program(
                category="Safety",
                program_ids_by_category=(
                    program_ids_by_category
                ),
                selected_program_ids=(
                    selected_program_ids
                ),
            )
        )

        if safety_program_id is not None:
            selected_program_ids.add(
                safety_program_id
            )

    # All managers receive leadership training.
    if (
        employee.organizational_level
        in MANAGER_LEVELS
    ):
        leadership_program_id = (
            choose_unused_program(
                category="Leadership",
                program_ids_by_category=(
                    program_ids_by_category
                ),
                selected_program_ids=(
                    selected_program_ids
                ),
            )
        )

        if leadership_program_id is not None:
            selected_program_ids.add(
                leadership_program_id
            )

    # Employees in technical departments may
    # receive technical training after 60 days.
    technical_eligible = (
        int(employee.department_id)
        in TECHNICAL_DEPARTMENTS
        and employment_days >= 60
    )

    if technical_eligible:
        technical_program_id = (
            choose_unused_program(
                category="Technical",
                program_ids_by_category=(
                    program_ids_by_category
                ),
                selected_program_ids=(
                    selected_program_ids
                ),
            )
        )

        if technical_program_id is not None:
            selected_program_ids.add(
                technical_program_id
            )

    # Longer-tenured technical employees sometimes
    # receive a second technical program.
    if (
        technical_eligible
        and employment_days >= 730
        and random.random() < 0.35
    ):
        second_technical_program_id = (
            choose_unused_program(
                category="Technical",
                program_ids_by_category=(
                    program_ids_by_category
                ),
                selected_program_ids=(
                    selected_program_ids
                ),
            )
        )

        if (
            second_technical_program_id
            is not None
        ):
            selected_program_ids.add(
                second_technical_program_id
            )

    # Some employees receive an additional optional
    # safety or technical program.
    if (
        employment_days >= 180
        and random.random() < 0.40
    ):
        optional_candidates = []

        for category in [
            "Technical",
            "Safety",
        ]:
            optional_candidates.extend(
                [
                    program_id
                    for program_id
                    in program_ids_by_category[
                        category
                    ]
                    if program_id
                    not in selected_program_ids
                ]
            )

        if optional_candidates:
            selected_program_ids.add(
                random.choice(
                    optional_candidates
                )
            )

    return sorted(
        selected_program_ids
    )


# ---------------------------------------------------------
# Training-date generation
# ---------------------------------------------------------

def choose_training_start_date(
    hire_date: date,
    employment_end_date: date,
    program_category: str,
) -> date | None:
    """Choose a valid training start date."""

    date_windows = {
        "Onboarding": (
            0,
            14,
        ),
        "Safety": (
            0,
            180,
        ),
        "Technical": (
            60,
            1_000,
        ),
        "Leadership": (
            180,
            1_200,
        ),
    }

    minimum_days, maximum_days = (
        date_windows[
            program_category
        ]
    )

    earliest_start_date = (
        hire_date
        + timedelta(
            days=minimum_days
        )
    )

    latest_start_date = min(
        hire_date
        + timedelta(
            days=maximum_days
        ),
        employment_end_date,
    )

    if (
        earliest_start_date
        > latest_start_date
    ):
        return None

    return random_date(
        earliest_start_date,
        latest_start_date,
    )


# ---------------------------------------------------------
# Completion-result generation
# ---------------------------------------------------------

def choose_completion_status(
    program_category: str,
    mandatory: bool,
) -> str:
    """Choose Completed, Incomplete, or Failed."""

    if (
        program_category == "Onboarding"
        and mandatory
    ):
        statuses = [
            "Completed",
            "Incomplete",
            "Failed",
        ]

        weights = [
            0.96,
            0.03,
            0.01,
        ]

    elif mandatory:
        statuses = [
            "Completed",
            "Incomplete",
            "Failed",
        ]

        weights = [
            0.93,
            0.05,
            0.02,
        ]

    elif program_category == "Leadership":
        statuses = [
            "Completed",
            "Incomplete",
            "Failed",
        ]

        weights = [
            0.91,
            0.06,
            0.03,
        ]

    elif program_category == "Technical":
        statuses = [
            "Completed",
            "Incomplete",
            "Failed",
        ]

        weights = [
            0.88,
            0.08,
            0.04,
        ]

    else:
        statuses = [
            "Completed",
            "Incomplete",
            "Failed",
        ]

        weights = [
            0.86,
            0.10,
            0.04,
        ]

    return random.choices(
        statuses,
        weights=weights,
        k=1,
    )[0]


def choose_training_hours(
    required_hours: float,
    completion_status: str,
) -> float:
    """Generate completed training hours."""

    if completion_status == "Completed":
        hours = (
            required_hours
            * random.uniform(
                1.00,
                1.15,
            )
        )

    elif completion_status == "Failed":
        hours = (
            required_hours
            * random.uniform(
                0.40,
                0.95,
            )
        )

    else:
        hours = (
            required_hours
            * random.uniform(
                0.00,
                0.75,
            )
        )

    return round(
        max(0, hours),
        1,
    )


def choose_training_score(
    completion_status: str,
) -> float | None:
    """Generate an optional assessment score."""

    if completion_status == "Completed":
        score = random.gauss(
            86,
            8,
        )

        return round(
            clamp(
                score,
                70,
                100,
            ),
            1,
        )

    if completion_status == "Failed":
        return round(
            random.uniform(
                35,
                69.9,
            ),
            1,
        )

    return None


# ---------------------------------------------------------
# Create one training record
# ---------------------------------------------------------

def create_training_record(
    training_record_id: int,
    employee_id: int,
    program_information: dict,
    start_date: date,
    employment_end_date: date,
) -> dict:
    """Create one synthetic training record."""

    program_category = str(
        program_information[
            "program_category"
        ]
    )

    required_hours = float(
        program_information[
            "required_hours"
        ]
    )

    mandatory = normalize_boolean(
        program_information[
            "mandatory"
        ]
    )

    completion_status = (
        choose_completion_status(
            program_category=(
                program_category
            ),
            mandatory=mandatory,
        )
    )

    completion_date = None

    if (
        completion_status
        != "Incomplete"
    ):
        estimated_duration_days = max(
            1,
            math.ceil(
                required_hours / 4
            )
            + random.randint(
                0,
                10,
            ),
        )

        proposed_completion_date = (
            start_date
            + timedelta(
                days=estimated_duration_days
            )
        )

        if (
            proposed_completion_date
            <= employment_end_date
        ):
            completion_date = (
                proposed_completion_date
            )
        else:
            # The employee's employment period ended
            # before the program could finish.
            completion_status = "Incomplete"

    training_hours = (
        choose_training_hours(
            required_hours=(
                required_hours
            ),
            completion_status=(
                completion_status
            ),
        )
    )

    score = choose_training_score(
        completion_status
    )

    return {
        "training_record_id": (
            training_record_id
        ),
        "employee_id": employee_id,
        "program_id": int(
            program_information[
                "program_id"
            ]
        ),
        "start_date": start_date,
        "completion_date": (
            completion_date
        ),
        "completion_status": (
            completion_status
        ),
        "training_hours": (
            training_hours
        ),
        "score": score,
    }


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------

def create_training_records(
    employees: pd.DataFrame,
    training_programs: pd.DataFrame,
) -> pd.DataFrame:
    """Create training records for the workforce."""

    random.seed(RANDOM_SEED)

    program_ids_by_category = (
        build_program_category_lookup(
            training_programs
        )
    )

    program_lookup = (
        training_programs
        .set_index("program_id")
        .to_dict(
            orient="index"
        )
    )

    training_records = []

    next_training_record_id = (
        FIRST_TRAINING_RECORD_ID
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

        hire_date = pd.Timestamp(
            employee.hire_date
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

        selected_program_ids = (
            select_employee_programs(
                employee=employee,
                program_ids_by_category=(
                    program_ids_by_category
                ),
                employment_end_date=(
                    employment_end_date
                ),
            )
        )

        for program_id in (
            selected_program_ids
        ):
            program_information = dict(
                program_lookup[
                    program_id
                ]
            )

            # Add program_id because it became the
            # dictionary index in program_lookup.
            program_information[
                "program_id"
            ] = program_id

            program_category = str(
                program_information[
                    "program_category"
                ]
            )

            start_date = (
                choose_training_start_date(
                    hire_date=hire_date,
                    employment_end_date=(
                        employment_end_date
                    ),
                    program_category=(
                        program_category
                    ),
                )
            )

            if start_date is None:
                continue

            training_record = (
                create_training_record(
                    training_record_id=(
                        next_training_record_id
                    ),
                    employee_id=(
                        employee_id
                    ),
                    program_information=(
                        program_information
                    ),
                    start_date=start_date,
                    employment_end_date=(
                        employment_end_date
                    ),
                )
            )

            training_records.append(
                training_record
            )

            next_training_record_id += 1

    training_records_df = pd.DataFrame(
        training_records
    )

    training_records_df[
        "start_date"
    ] = pd.to_datetime(
        training_records_df[
            "start_date"
        ]
    )

    training_records_df[
        "completion_date"
    ] = pd.to_datetime(
        training_records_df[
            "completion_date"
        ]
    )

    return training_records_df


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_training_records(
    training_records: pd.DataFrame,
    employees: pd.DataFrame,
    training_programs: pd.DataFrame,
) -> None:
    """Validate training-record business rules."""

    expected_columns = {
        "training_record_id",
        "employee_id",
        "program_id",
        "start_date",
        "completion_date",
        "completion_status",
        "training_hours",
        "score",
    }

    if (
        set(training_records.columns)
        != expected_columns
    ):
        raise ValueError(
            "training_records: unexpected columns."
        )

    if training_records.empty:
        raise ValueError(
            "training_records: no records generated."
        )

    if training_records[
        "training_record_id"
    ].isna().any():
        raise ValueError(
            "training_records: missing record IDs."
        )

    if not training_records[
        "training_record_id"
    ].is_unique:
        raise ValueError(
            "training_records: record IDs "
            "are not unique."
        )

    valid_employee_ids = set(
        employees["employee_id"]
        .astype(int)
    )

    valid_program_ids = set(
        training_programs[
            "program_id"
        ].astype(int)
    )

    used_employee_ids = set(
        training_records[
            "employee_id"
        ].astype(int)
    )

    used_program_ids = set(
        training_records[
            "program_id"
        ].astype(int)
    )

    if used_employee_ids != valid_employee_ids:
        raise ValueError(
            "training_records: not every employee "
            "has a training record."
        )

    if not used_program_ids.issubset(
        valid_program_ids
    ):
        raise ValueError(
            "training_records: invalid program ID."
        )

    if training_records.duplicated(
        subset=[
            "employee_id",
            "program_id",
        ]
    ).any():
        raise ValueError(
            "training_records: an employee has "
            "duplicate program records."
        )

    if not set(
        training_records[
            "completion_status"
        ]
    ).issubset(
        ALLOWED_COMPLETION_STATUSES
    ):
        raise ValueError(
            "training_records: invalid "
            "completion status."
        )

    employee_dates = (
        employees[
            [
                "employee_id",
                "hire_date",
                "termination_date",
                "department_id",
                "employment_type",
                "organizational_level",
            ]
        ]
        .copy()
    )

    employee_dates[
        "employment_end_date"
    ] = (
        employee_dates[
            "termination_date"
        ]
        .fillna(
            pd.Timestamp(
                AS_OF_DATE
            )
        )
    )

    detailed_records = (
        training_records
        .merge(
            employee_dates,
            on="employee_id",
            how="left",
        )
        .merge(
            training_programs,
            on="program_id",
            how="left",
        )
    )

    if detailed_records[
        "program_category"
    ].isna().any():
        raise ValueError(
            "training_records: missing "
            "program details."
        )

    if (
        detailed_records[
            "start_date"
        ]
        < detailed_records[
            "hire_date"
        ]
    ).any():
        raise ValueError(
            "training_records: training starts "
            "before hire."
        )

    if (
        detailed_records[
            "start_date"
        ]
        > detailed_records[
            "employment_end_date"
        ]
    ).any():
        raise ValueError(
            "training_records: training starts "
            "after employment ends."
        )

    completed_or_failed = (
        detailed_records[
            "completion_status"
        ].isin(
            [
                "Completed",
                "Failed",
            ]
        )
    )

    incomplete = (
        detailed_records[
            "completion_status"
        ]
        == "Incomplete"
    )

    if detailed_records.loc[
        completed_or_failed,
        "completion_date",
    ].isna().any():
        raise ValueError(
            "training_records: completed or failed "
            "training is missing a completion date."
        )

    if detailed_records.loc[
        incomplete,
        "completion_date",
    ].notna().any():
        raise ValueError(
            "training_records: incomplete training "
            "has a completion date."
        )

    dated_records = detailed_records[
        detailed_records[
            "completion_date"
        ].notna()
    ]

    if (
        dated_records[
            "completion_date"
        ]
        < dated_records[
            "start_date"
        ]
    ).any():
        raise ValueError(
            "training_records: completion occurs "
            "before training begins."
        )

    if (
        dated_records[
            "completion_date"
        ]
        > dated_records[
            "employment_end_date"
        ]
    ).any():
        raise ValueError(
            "training_records: completion occurs "
            "after employment ends."
        )

    if (
        detailed_records[
            "training_hours"
        ]
        < 0
    ).any():
        raise ValueError(
            "training_records: training hours "
            "cannot be negative."
        )

    completed = (
        detailed_records[
            "completion_status"
        ]
        == "Completed"
    )

    failed = (
        detailed_records[
            "completion_status"
        ]
        == "Failed"
    )

    if not detailed_records.loc[
        completed,
        "training_hours",
    ].ge(
        detailed_records.loc[
            completed,
            "required_hours",
        ]
    ).all():
        raise ValueError(
            "training_records: completed training "
            "has insufficient hours."
        )

    if not detailed_records.loc[
        failed | incomplete,
        "training_hours",
    ].lt(
        detailed_records.loc[
            failed | incomplete,
            "required_hours",
        ]
    ).all():
        raise ValueError(
            "training_records: failed or incomplete "
            "training has too many hours."
        )

    if not detailed_records.loc[
        completed,
        "score",
    ].between(
        70,
        100,
    ).all():
        raise ValueError(
            "training_records: completed training "
            "has an invalid score."
        )

    if not detailed_records.loc[
        failed,
        "score",
    ].between(
        0,
        69.9,
    ).all():
        raise ValueError(
            "training_records: failed training "
            "has an invalid score."
        )

    if detailed_records.loc[
        incomplete,
        "score",
    ].notna().any():
        raise ValueError(
            "training_records: incomplete training "
            "should not have a score."
        )

    # Every employee must have exactly one
    # onboarding record.
    onboarding_records = (
        detailed_records[
            detailed_records[
                "program_category"
            ]
            == "Onboarding"
        ]
    )

    onboarding_counts = (
        onboarding_records
        .groupby("employee_id")
        .size()
        .reindex(
            employees[
                "employee_id"
            ],
            fill_value=0,
        )
    )

    if not onboarding_counts.eq(1).all():
        raise ValueError(
            "training_records: every employee must "
            "have exactly one onboarding record."
        )

    # Every manager must have leadership training.
    manager_ids = set(
        employees.loc[
            employees[
                "organizational_level"
            ].isin(
                MANAGER_LEVELS
            ),
            "employee_id",
        ].astype(int)
    )

    leadership_employee_ids = set(
        detailed_records.loc[
            detailed_records[
                "program_category"
            ]
            == "Leadership",
            "employee_id",
        ].astype(int)
    )

    if not manager_ids.issubset(
        leadership_employee_ids
    ):
        raise ValueError(
            "training_records: a manager is missing "
            "leadership training."
        )

    # Every employee in a safety-required group
    # must have at least one safety record.
    safety_required_ids = set(
        employees.loc[
            employees[
                "department_id"
            ].isin(
                SAFETY_DEPARTMENTS
            )
            | employees[
                "employment_type"
            ].eq("Hourly"),
            "employee_id",
        ].astype(int)
    )

    safety_employee_ids = set(
        detailed_records.loc[
            detailed_records[
                "program_category"
            ]
            == "Safety",
            "employee_id",
        ].astype(int)
    )

    if not safety_required_ids.issubset(
        safety_employee_ids
    ):
        raise ValueError(
            "training_records: a safety-required "
            "employee is missing safety training."
        )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_training_records(
    training_records: pd.DataFrame,
) -> None:
    """Save the training records as a CSV."""

    output_path = (
        RAW_DATA_DIR
        / "training_records.csv"
    )

    training_records.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved training_records.csv: "
        f"{len(training_records)} rows and "
        f"{len(training_records.columns)} columns"
    )


def print_summary(
    training_records: pd.DataFrame,
    training_programs: pd.DataFrame,
) -> None:
    """Print a basic training summary."""

    training_details = (
        training_records
        .merge(
            training_programs[
                [
                    "program_id",
                    "program_category",
                ]
            ],
            on="program_id",
            how="left",
        )
    )

    print(
        "\nEmployees with training records:"
    )

    print(
        training_records[
            "employee_id"
        ].nunique()
    )

    print(
        "\nRecords by program category:"
    )

    print(
        training_details[
            "program_category"
        ].value_counts()
    )

    print(
        "\nRecords by completion status:"
    )

    print(
        training_records[
            "completion_status"
        ].value_counts()
    )

    print(
        "\nAverage records per employee:"
    )

    records_per_employee = (
        training_records
        .groupby("employee_id")
        .size()
    )

    print(
        round(
            records_per_employee.mean(),
            2,
        )
    )

    print(
        "\nOnboarding records:"
    )

    print(
        (
            training_details[
                "program_category"
            ]
            == "Onboarding"
        ).sum()
    )


def main() -> None:
    """Generate and validate training records."""

    employees = load_table(
        "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    training_programs = load_table(
        "training_programs.csv"
    )

    training_records = (
        create_training_records(
            employees,
            training_programs,
        )
    )

    validate_training_records(
        training_records,
        employees,
        training_programs,
    )

    save_training_records(
        training_records
    )

    print_summary(
        training_records,
        training_programs,
    )

    print(
        "\nTraining records generated "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()