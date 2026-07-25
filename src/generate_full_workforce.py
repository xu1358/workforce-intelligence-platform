from datetime import date, timedelta
from pathlib import Path
import math
import random

import pandas as pd
from faker import Faker


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AS_OF_DATE = date(2026, 6, 30)
RANDOM_SEED = 42

NUM_EMPLOYEES = 10_000
FIRST_EMPLOYEE_ID = 100_001

MAX_DEPARTMENT_HEAD_REPORTS = 30
MAX_MANAGER_REPORTS = 12

# Build spare team-manager capacity so that reports can be
# reassigned when a team manager leaves.  The hard validation
# limit remains MAX_MANAGER_REPORTS.
TARGET_TEAM_MANAGER_REPORTS = 8


# ---------------------------------------------------------
# Workforce-distribution settings
# ---------------------------------------------------------

DEPARTMENT_WEIGHTS = {
    1: 0.20,
    2: 0.25,
    3: 0.12,
    4: 0.10,
    5: 0.08,
    6: 0.07,
    7: 0.10,
    8: 0.08,
}


INDIVIDUAL_ROLE_MAP = {
    1: [1, 2, 3, 4, 6],
    2: [6, 7],
    3: [9, 11],
    4: [12],
    5: [1, 14],
    6: [1, 16],
    7: [1, 2, 3, 4, 18],
    8: [20],
}


MANAGER_ROLE_BY_DEPARTMENT = {
    1: 5,
    2: 8,
    3: 10,
    4: 13,
    5: 15,
    6: 17,
    7: 19,
    8: 20,
}


LOCATION_WEIGHTS = {
    1: 0.28,
    2: 0.24,
    3: 0.18,
    4: 0.12,
    5: 0.18,
}


HOURLY_ROLE_IDS = {
    7,
    20,
}


# ---------------------------------------------------------
# General helper functions
# ---------------------------------------------------------

def load_reference_table(filename: str) -> pd.DataFrame:
    """Load one reference table."""

    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {filename}. "
            "Run generate_reference_data.py first."
        )

    return pd.read_csv(path)


def random_date(
    start_date: date,
    end_date: date,
) -> date:
    """Generate a random date inside a date range."""

    if start_date > end_date:
        raise ValueError(
            "start_date cannot be after end_date."
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


def choose_from_weight_map(
    weight_map: dict[int, float],
) -> int:
    """Choose one ID according to predefined weights."""

    values = list(weight_map.keys())
    weights = list(weight_map.values())

    return random.choices(
        values,
        weights=weights,
        k=1,
    )[0]


def choose_education(
    job_level: int,
) -> str:
    """Choose an education level partly based on job level."""

    education_levels = [
        "High School",
        "Associate",
        "Bachelor's",
        "Master's",
        "Doctorate",
    ]

    if job_level >= 3:
        weights = [
            0.04,
            0.06,
            0.46,
            0.38,
            0.06,
        ]
    else:
        weights = [
            0.18,
            0.20,
            0.47,
            0.14,
            0.01,
        ]

    return random.choices(
        education_levels,
        weights=weights,
        k=1,
    )[0]


# ---------------------------------------------------------
# Department-allocation logic
# ---------------------------------------------------------

def allocate_department_counts(
    total_employees: int,
    department_weights: dict[int, float],
) -> dict[int, int]:
    """Allocate an exact workforce total across departments."""

    if total_employees < len(department_weights):
        raise ValueError(
            "The employee total must be at least "
            "the number of departments."
        )

    if not math.isclose(
        sum(department_weights.values()),
        1.0,
        abs_tol=0.000001,
    ):
        raise ValueError(
            "Department weights must add up to 1."
        )

    # Every department receives one employee first.
    # That employee will become the department head.
    department_counts = {
        department_id: 1
        for department_id
        in department_weights
    }

    remaining_employees = (
        total_employees
        - len(department_weights)
    )

    exact_allocations = {
        department_id: (
            remaining_employees * weight
        )
        for department_id, weight
        in department_weights.items()
    }

    floor_allocations = {
        department_id: int(exact_value)
        for department_id, exact_value
        in exact_allocations.items()
    }

    for department_id, floor_value in (
        floor_allocations.items()
    ):
        department_counts[
            department_id
        ] += floor_value

    unassigned_employees = (
        remaining_employees
        - sum(floor_allocations.values())
    )

    departments_by_remainder = sorted(
        department_weights,
        key=lambda department_id: (
            exact_allocations[department_id]
            - floor_allocations[department_id]
        ),
        reverse=True,
    )

    for department_id in (
        departments_by_remainder[
            :unassigned_employees
        ]
    ):
        department_counts[
            department_id
        ] += 1

    return department_counts


# ---------------------------------------------------------
# Hierarchy calculation
# ---------------------------------------------------------

def calculate_hierarchy_counts(
    department_size: int,
) -> tuple[int, int, int]:
    """
    Return:

    senior_manager_count,
    team_manager_count,
    individual_contributor_count
    """

    positions_below_head = (
        department_size - 1
    )

    # A small department can report directly to the head.
    if (
        positions_below_head
        <= MAX_MANAGER_REPORTS
    ):
        return (
            0,
            0,
            positions_below_head,
        )

    # Search for the smallest number of team managers
    # that satisfies all direct-report limits.
    for team_manager_count in range(
        1,
        department_size,
    ):
        if (
            team_manager_count
            <= MAX_DEPARTMENT_HEAD_REPORTS
        ):
            senior_manager_count = 0
        else:
            senior_manager_count = math.ceil(
                team_manager_count
                / MAX_MANAGER_REPORTS
            )

        individual_contributor_count = (
            department_size
            - 1
            - senior_manager_count
            - team_manager_count
        )

        if individual_contributor_count < 0:
            continue

        team_manager_capacity = (
            team_manager_count
            * TARGET_TEAM_MANAGER_REPORTS
        )

        if (
            individual_contributor_count
            > team_manager_capacity
        ):
            continue

        if (
            senior_manager_count
            > MAX_DEPARTMENT_HEAD_REPORTS
        ):
            continue

        if senior_manager_count > 0:
            senior_manager_capacity = (
                senior_manager_count
                * MAX_MANAGER_REPORTS
            )

            if (
                team_manager_count
                > senior_manager_capacity
            ):
                continue

        return (
            senior_manager_count,
            team_manager_count,
            individual_contributor_count,
        )

    raise ValueError(
        f"Unable to create hierarchy for "
        f"department size {department_size}."
    )


# ---------------------------------------------------------
# Manager-record helper
# ---------------------------------------------------------

def create_manager_record(
    employee_id: int,
    fake: Faker,
    department_id: int,
    job_role_id: int,
    manager_id: int | None,
    organizational_level: str,
    job_level: int,
    hire_date_end: date,
    earliest_birth_year: int,
    latest_birth_year: int,
) -> dict:
    """Create one active management employee."""

    return {
        "employee_id": employee_id,
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "hire_date": random_date(
            date(2021, 1, 1),
            hire_date_end,
        ),
        "termination_date": None,
        "employment_status": "Active",
        "termination_type": None,
        "department_id": department_id,
        "location_id": choose_from_weight_map(
            LOCATION_WEIGHTS
        ),
        "job_role_id": job_role_id,
        "manager_id": manager_id,
        "employment_type": "Salaried",
        "birth_year": random.randint(
            earliest_birth_year,
            latest_birth_year,
        ),
        "education_level": choose_education(
            job_level
        ),
        "organizational_level": (
            organizational_level
        ),
    }


# ---------------------------------------------------------
# Main workforce-generation function
# ---------------------------------------------------------

def create_full_workforce(
    departments: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> pd.DataFrame:
    """Create the complete 10,000-employee workforce."""

    random.seed(RANDOM_SEED)
    Faker.seed(RANDOM_SEED)

    fake = Faker("en_US")

    department_ids = sorted(
        departments["department_id"]
        .astype(int)
        .tolist()
    )

    job_level_by_role = (
        job_roles
        .set_index("job_role_id")["job_level"]
        .astype(int)
        .to_dict()
    )

    department_counts = (
        allocate_department_counts(
            NUM_EMPLOYEES,
            DEPARTMENT_WEIGHTS,
        )
    )

    employees = []

    next_employee_id = (
        FIRST_EMPLOYEE_ID
    )

    for department_id in department_ids:
        department_size = (
            department_counts[
                department_id
            ]
        )

        manager_role_id = (
            MANAGER_ROLE_BY_DEPARTMENT[
                department_id
            ]
        )

        (
            senior_manager_count,
            team_manager_count,
            individual_contributor_count,
        ) = calculate_hierarchy_counts(
            department_size
        )

        # -------------------------------------------------
        # Department head
        # -------------------------------------------------

        department_head_id = (
            next_employee_id
        )

        department_head = create_manager_record(
            employee_id=next_employee_id,
            fake=fake,
            department_id=department_id,
            job_role_id=manager_role_id,
            manager_id=None,
            organizational_level=(
                "Department Head"
            ),
            job_level=job_level_by_role[
                manager_role_id
            ],
            hire_date_end=date(
                2022,
                12,
                31,
            ),
            earliest_birth_year=1962,
            latest_birth_year=1992,
        )

        employees.append(
            department_head
        )

        next_employee_id += 1

        # -------------------------------------------------
        # Senior managers
        # -------------------------------------------------

        senior_manager_ids = []

        for _ in range(
            senior_manager_count
        ):
            senior_manager_id = (
                next_employee_id
            )

            senior_manager = (
                create_manager_record(
                    employee_id=(
                        senior_manager_id
                    ),
                    fake=fake,
                    department_id=(
                        department_id
                    ),
                    job_role_id=(
                        manager_role_id
                    ),
                    manager_id=(
                        department_head_id
                    ),
                    organizational_level=(
                        "Senior Manager"
                    ),
                    job_level=(
                        job_level_by_role[
                            manager_role_id
                        ]
                    ),
                    hire_date_end=date(
                        2023,
                        12,
                        31,
                    ),
                    earliest_birth_year=1965,
                    latest_birth_year=1996,
                )
            )

            employees.append(
                senior_manager
            )

            senior_manager_ids.append(
                senior_manager_id
            )

            next_employee_id += 1

        # -------------------------------------------------
        # Team managers
        # -------------------------------------------------

        team_manager_ids = []

        for position_number in range(
            team_manager_count
        ):
            if senior_manager_ids:
                manager_id = (
                    senior_manager_ids[
                        position_number
                        % len(
                            senior_manager_ids
                        )
                    ]
                )
            else:
                manager_id = (
                    department_head_id
                )

            team_manager_id = (
                next_employee_id
            )

            team_manager = (
                create_manager_record(
                    employee_id=(
                        team_manager_id
                    ),
                    fake=fake,
                    department_id=(
                        department_id
                    ),
                    job_role_id=(
                        manager_role_id
                    ),
                    manager_id=manager_id,
                    organizational_level=(
                        "Team Manager"
                    ),
                    job_level=(
                        job_level_by_role[
                            manager_role_id
                        ]
                    ),
                    hire_date_end=date(
                        2024,
                        12,
                        31,
                    ),
                    earliest_birth_year=1968,
                    latest_birth_year=1999,
                )
            )

            employees.append(
                team_manager
            )

            team_manager_ids.append(
                team_manager_id
            )

            next_employee_id += 1

        # -------------------------------------------------
        # Individual contributors
        # -------------------------------------------------

        for position_number in range(
            individual_contributor_count
        ):
            job_role_id = random.choice(
                INDIVIDUAL_ROLE_MAP[
                    department_id
                ]
            )

            location_id = (
                choose_from_weight_map(
                    LOCATION_WEIGHTS
                )
            )

            hire_date = random_date(
                date(2021, 1, 1),
                AS_OF_DATE,
            )

            if team_manager_ids:
                manager_id = (
                    team_manager_ids[
                        position_number
                        % len(
                            team_manager_ids
                        )
                    ]
                )
            else:
                manager_id = (
                    department_head_id
                )

            employee = {
                "employee_id": next_employee_id,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "hire_date": hire_date,
                # Attrition is applied later, after the
                # potential histories have been generated.
                "termination_date": None,
                "employment_status": "Active",
                "termination_type": None,
                "department_id": department_id,
                "location_id": location_id,
                "job_role_id": job_role_id,
                "manager_id": manager_id,
                "employment_type": (
                    "Hourly"
                    if job_role_id
                    in HOURLY_ROLE_IDS
                    else "Salaried"
                ),
                "birth_year": random.randint(
                    1962,
                    2003,
                ),
                "education_level": (
                    choose_education(
                        job_level_by_role[
                            job_role_id
                        ]
                    )
                ),
                "organizational_level": (
                    "Individual Contributor"
                ),
            }

            employees.append(employee)

            next_employee_id += 1

    employee_df = pd.DataFrame(
        employees
    )

    employee_df["hire_date"] = (
        pd.to_datetime(
            employee_df["hire_date"]
        )
    )

    employee_df["termination_date"] = (
        pd.to_datetime(
            employee_df[
                "termination_date"
            ]
        )
    )

    employee_df["manager_id"] = (
        employee_df["manager_id"]
        .astype("Int64")
    )

    return employee_df


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_full_workforce(
    employees: pd.DataFrame,
    departments: pd.DataFrame,
    locations: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> None:
    """Validate the complete workforce."""

    if len(employees) != NUM_EMPLOYEES:
        raise ValueError(
            "employees: incorrect row count."
        )

    if employees["employee_id"].isna().any():
        raise ValueError(
            "employees: missing employee IDs."
        )

    if not employees[
        "employee_id"
    ].is_unique:
        raise ValueError(
            "employees: employee IDs "
            "are not unique."
        )

    valid_department_ids = set(
        departments[
            "department_id"
        ].astype(int)
    )

    valid_location_ids = set(
        locations[
            "location_id"
        ].astype(int)
    )

    valid_job_role_ids = set(
        job_roles[
            "job_role_id"
        ].astype(int)
    )

    valid_employee_ids = set(
        employees[
            "employee_id"
        ].astype(int)
    )

    if not set(
        employees["department_id"]
    ).issubset(valid_department_ids):
        raise ValueError(
            "employees: invalid department ID."
        )

    if not set(
        employees["location_id"]
    ).issubset(valid_location_ids):
        raise ValueError(
            "employees: invalid location ID."
        )

    if not set(
        employees["job_role_id"]
    ).issubset(valid_job_role_ids):
        raise ValueError(
            "employees: invalid job-role ID."
        )

    used_manager_ids = set(
        employees["manager_id"]
        .dropna()
        .astype(int)
    )

    if not used_manager_ids.issubset(
        valid_employee_ids
    ):
        raise ValueError(
            "employees: invalid manager ID."
        )

    self_managed = (
        employees["manager_id"].notna()
        & (
            employees["manager_id"]
            == employees["employee_id"]
        )
    )

    if self_managed.any():
        raise ValueError(
            "employees: an employee manages "
            "themselves."
        )

    allowed_levels = {
        "Department Head",
        "Senior Manager",
        "Team Manager",
        "Individual Contributor",
    }

    if not set(
        employees[
            "organizational_level"
        ]
    ).issubset(allowed_levels):
        raise ValueError(
            "employees: invalid "
            "organizational level."
        )

    # -----------------------------------------------------
    # Employment-status checks
    # -----------------------------------------------------

    active = (
        employees["employment_status"]
        == "Active"
    )

    terminated = (
        employees["employment_status"]
        == "Terminated"
    )

    if employees.loc[
        active,
        "termination_date",
    ].notna().any():
        raise ValueError(
            "employees: an active employee "
            "has a termination date."
        )

    if employees.loc[
        terminated,
        "termination_date",
    ].isna().any():
        raise ValueError(
            "employees: a terminated employee "
            "is missing a termination date."
        )

    if employees.loc[
        terminated,
        "termination_type",
    ].isna().any():
        raise ValueError(
            "employees: a terminated employee "
            "is missing a termination type."
        )

    invalid_termination_dates = (
        terminated
        & (
            employees[
                "termination_date"
            ]
            < employees["hire_date"]
        )
    )

    if invalid_termination_dates.any():
        raise ValueError(
            "employees: termination occurs "
            "before hire."
        )

    # -----------------------------------------------------
    # Hierarchy checks
    # -----------------------------------------------------

    employee_lookup = (
        employees
        .set_index("employee_id")
    )

    department_heads = employees[
        employees["organizational_level"]
        == "Department Head"
    ]

    if (
        len(department_heads)
        != len(departments)
    ):
        raise ValueError(
            "employees: every department must "
            "have one department head."
        )

    if (
        department_heads[
            "department_id"
        ].nunique()
        != len(departments)
    ):
        raise ValueError(
            "employees: department-head "
            "assignment is invalid."
        )

    if department_heads[
        "manager_id"
    ].notna().any():
        raise ValueError(
            "employees: department heads "
            "must not have managers."
        )

    managers = employees[
        employees[
            "organizational_level"
        ].isin(
            [
                "Department Head",
                "Senior Manager",
                "Team Manager",
            ]
        )
    ]

    if not managers[
        "employment_status"
    ].eq("Active").all():
        raise ValueError(
            "employees: all managers must "
            "be active."
        )

    managed_employees = employees[
        employees["manager_id"].notna()
    ].copy()

    managed_employees[
        "manager_department_id"
    ] = (
        managed_employees["manager_id"]
        .astype(int)
        .map(
            employee_lookup[
                "department_id"
            ]
        )
    )

    if not (
        managed_employees[
            "department_id"
        ]
        ==
        managed_employees[
            "manager_department_id"
        ]
    ).all():
        raise ValueError(
            "employees: employee and manager "
            "departments do not match."
        )

    managed_employees[
        "manager_status"
    ] = (
        managed_employees["manager_id"]
        .astype(int)
        .map(
            employee_lookup[
                "employment_status"
            ]
        )
    )

    if not managed_employees[
        "manager_status"
    ].eq("Active").all():
        raise ValueError(
            "employees: an employee reports "
            "to an inactive manager."
        )

    managed_employees[
        "manager_level"
    ] = (
        managed_employees["manager_id"]
        .astype(int)
        .map(
            employee_lookup[
                "organizational_level"
            ]
        )
    )

    senior_managers = managed_employees[
        managed_employees[
            "organizational_level"
        ]
        == "Senior Manager"
    ]

    if not senior_managers[
        "manager_level"
    ].eq("Department Head").all():
        raise ValueError(
            "employees: senior managers must "
            "report to department heads."
        )

    team_managers = managed_employees[
        managed_employees[
            "organizational_level"
        ]
        == "Team Manager"
    ]

    if not team_managers[
        "manager_level"
    ].isin(
        [
            "Department Head",
            "Senior Manager",
        ]
    ).all():
        raise ValueError(
            "employees: team-manager "
            "relationship is invalid."
        )

    individual_contributors = (
        managed_employees[
            managed_employees[
                "organizational_level"
            ]
            == "Individual Contributor"
        ]
    )

    if not individual_contributors[
        "manager_level"
    ].isin(
        [
            "Department Head",
            "Team Manager",
        ]
    ).all():
        raise ValueError(
            "employees: individual-contributor "
            "relationship is invalid."
        )

    # -----------------------------------------------------
    # Direct-report limits
    # -----------------------------------------------------

    direct_report_counts = (
        employees["manager_id"]
        .dropna()
        .astype(int)
        .value_counts()
        .rename("direct_reports")
        .to_frame()
    )

    direct_report_counts[
        "manager_level"
    ] = (
        direct_report_counts.index
        .map(
            employee_lookup[
                "organizational_level"
            ]
        )
    )

    excessive_department_heads = (
        direct_report_counts[
            (
                direct_report_counts[
                    "manager_level"
                ]
                == "Department Head"
            )
            & (
                direct_report_counts[
                    "direct_reports"
                ]
                > MAX_DEPARTMENT_HEAD_REPORTS
            )
        ]
    )

    excessive_other_managers = (
        direct_report_counts[
            (
                direct_report_counts[
                    "manager_level"
                ].isin(
                    [
                        "Senior Manager",
                        "Team Manager",
                    ]
                )
            )
            & (
                direct_report_counts[
                    "direct_reports"
                ]
                > MAX_MANAGER_REPORTS
            )
        ]
    )

    if not excessive_department_heads.empty:
        raise ValueError(
            "employees: a department head "
            "exceeds the direct-report limit."
        )

    if not excessive_other_managers.empty:
        raise ValueError(
            "employees: a manager exceeds "
            "the direct-report limit."
        )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_full_workforce(
    employees: pd.DataFrame,
) -> None:
    """Save the complete workforce."""

    output_path = (
        RAW_DATA_DIR / "employees.csv"
    )

    employees.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved employees.csv: "
        f"{len(employees)} rows and "
        f"{len(employees.columns)} columns"
    )


def print_summary(
    employees: pd.DataFrame,
) -> None:
    """Print workforce and hierarchy summaries."""

    print("\nOrganizational levels:")
    print(
        employees[
            "organizational_level"
        ].value_counts()
    )

    print("\nEmployment status:")
    print(
        employees[
            "employment_status"
        ].value_counts()
    )

    print("\nEmployees by department:")
    print(
        employees[
            "department_id"
        ]
        .value_counts()
        .sort_index()
    )

    direct_report_counts = (
        employees["manager_id"]
        .dropna()
        .astype(int)
        .value_counts()
    )

    print(
        "\nLargest direct-report count:"
    )

    print(
        int(
            direct_report_counts.max()
        )
    )


def main() -> None:
    """Generate and validate the complete workforce."""

    departments = load_reference_table(
        "departments.csv"
    )

    locations = load_reference_table(
        "locations.csv"
    )

    job_roles = load_reference_table(
        "job_roles.csv"
    )

    employees = create_full_workforce(
        departments,
        job_roles,
    )

    validate_full_workforce(
        employees,
        departments,
        locations,
        job_roles,
    )

    save_full_workforce(
        employees
    )

    print_summary(
        employees
    )

    print(
        "\nFull workforce generated "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()
