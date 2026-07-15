from datetime import date, timedelta
from pathlib import Path
import math
import random

import pandas as pd
from faker import Faker


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder containing generated CSV files.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Fixed snapshot date for reproducible analysis.
AS_OF_DATE = date(2026, 6, 30)

# Fixed random seed for reproducible data.
RANDOM_SEED = 42

# Continue using a small teaching sample.
NUM_EMPLOYEES = 50

# No manager should have more than this many direct reports.
MAX_DIRECT_REPORTS = 6


# Reasonable job roles for each department.
DEPARTMENT_ROLE_MAP = {
    1: [1, 2, 3, 4, 5, 6],
    2: [6, 7, 8],
    3: [9, 10, 11],
    4: [12, 13],
    5: [1, 14, 15],
    6: [1, 16, 17],
    7: [1, 2, 3, 4, 5, 18, 19],
    8: [1, 20],
}


# Approximate percentage of employees in each department.
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


# Job role assigned to managers in each department.
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


# Approximate employee distribution across locations.
LOCATION_WEIGHTS = {
    1: 0.28,
    2: 0.24,
    3: 0.18,
    4: 0.12,
    5: 0.18,
}


# These individual-contributor roles are hourly.
HOURLY_ROLE_IDS = {7, 20}


def load_reference_table(filename: str) -> pd.DataFrame:
    """Load one previously generated reference table."""

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
    """Generate a random date between two dates."""

    if start_date > end_date:
        raise ValueError(
            "start_date cannot be after end_date."
        )

    total_days = (
        end_date - start_date
    ).days

    random_number_of_days = random.randint(
        0,
        total_days,
    )

    return start_date + timedelta(
        days=random_number_of_days
    )


def choose_from_weight_map(
    weight_map: dict[int, float],
) -> int:
    """Select one ID using predefined probabilities."""

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
    """Choose education partly according to job level."""

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


def choose_termination(
    hire_date: date,
) -> tuple[str, date | None, str | None]:
    """Determine whether an employee is active or terminated."""

    days_since_hire = (
        AS_OF_DATE - hire_date
    ).days

    # Very recent hires remain active.
    if days_since_hire < 120:
        return "Active", None, None

    # Approximately 22% of eligible individual contributors
    # will be terminated.
    if random.random() > 0.22:
        return "Active", None, None

    earliest_termination = (
        hire_date + timedelta(days=60)
    )

    termination_date = random_date(
        earliest_termination,
        AS_OF_DATE,
    )

    termination_type = random.choices(
        ["Voluntary", "Involuntary"],
        weights=[0.75, 0.25],
        k=1,
    )[0]

    return (
        "Terminated",
        termination_date,
        termination_type,
    )


def allocate_department_counts(
    total_employees: int,
    department_weights: dict[int, float],
) -> dict[int, int]:
    """
    Allocate an exact number of employees across departments.

    Every department receives at least one employee so that
    it can have a department head.
    """

    number_of_departments = len(
        department_weights
    )

    if total_employees < number_of_departments:
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

    # Give every department one department head first.
    counts = {
        department_id: 1
        for department_id
        in department_weights
    }

    remaining_employees = (
        total_employees
        - number_of_departments
    )

    exact_allocations = {
        department_id: (
            remaining_employees * weight
        )
        for department_id, weight
        in department_weights.items()
    }

    floor_allocations = {
        department_id: int(exact_count)
        for department_id, exact_count
        in exact_allocations.items()
    }

    for department_id, floor_count in (
        floor_allocations.items()
    ):
        counts[department_id] += floor_count

    employees_still_unassigned = (
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
            :employees_still_unassigned
        ]
    ):
        counts[department_id] += 1

    return counts


def calculate_team_manager_count(
    department_size: int,
) -> int:
    """
    Determine how many team managers a department needs.

    Small departments can report directly to the department
    head. Larger departments need team managers.
    """

    positions_below_head = (
        department_size - 1
    )

    if positions_below_head <= MAX_DIRECT_REPORTS:
        return 0

    # Each team manager occupies one position and may supervise
    # up to MAX_DIRECT_REPORTS individual contributors.
    return math.ceil(
        positions_below_head
        / (MAX_DIRECT_REPORTS + 1)
    )


def create_employees(
    departments: pd.DataFrame,
    locations: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> pd.DataFrame:
    """Create employees with a two-level management hierarchy."""

    random.seed(RANDOM_SEED)
    Faker.seed(RANDOM_SEED)

    fake = Faker("en_US")

    valid_department_ids = sorted(
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

    next_employee_id = 10001

    for department_id in valid_department_ids:
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

        # -------------------------------------------------
        # Step 1: Create the department head.
        # -------------------------------------------------

        department_head_id = next_employee_id

        department_head = {
            "employee_id": department_head_id,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "hire_date": random_date(
                date(2021, 1, 1),
                date(2022, 12, 31),
            ),
            "termination_date": None,
            "employment_status": "Active",
            "termination_type": None,
            "department_id": department_id,
            "location_id": choose_from_weight_map(
                LOCATION_WEIGHTS
            ),
            "job_role_id": manager_role_id,
            "manager_id": None,
            "employment_type": "Salaried",
            "birth_year": random.randint(
                1965,
                1995,
            ),
            "education_level": choose_education(
                job_level_by_role[
                    manager_role_id
                ]
            ),
            "organizational_level": (
                "Department Head"
            ),
        }

        employees.append(department_head)

        next_employee_id += 1

        positions_below_head = (
            department_size - 1
        )

        number_of_team_managers = (
            calculate_team_manager_count(
                department_size
            )
        )

        team_manager_ids = []

        # -------------------------------------------------
        # Step 2: Create team managers when necessary.
        # -------------------------------------------------

        for _ in range(
            number_of_team_managers
        ):
            team_manager_id = next_employee_id

            team_manager = {
                "employee_id": team_manager_id,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "hire_date": random_date(
                    date(2021, 1, 1),
                    date(2023, 12, 31),
                ),
                "termination_date": None,
                "employment_status": "Active",
                "termination_type": None,
                "department_id": department_id,
                "location_id": (
                    choose_from_weight_map(
                        LOCATION_WEIGHTS
                    )
                ),
                "job_role_id": manager_role_id,
                "manager_id": (
                    department_head_id
                ),
                "employment_type": "Salaried",
                "birth_year": random.randint(
                    1968,
                    1998,
                ),
                "education_level": (
                    choose_education(
                        job_level_by_role[
                            manager_role_id
                        ]
                    )
                ),
                "organizational_level": (
                    "Team Manager"
                ),
            }

            employees.append(team_manager)
            team_manager_ids.append(
                team_manager_id
            )

            next_employee_id += 1

        number_of_individual_contributors = (
            positions_below_head
            - number_of_team_managers
        )

        possible_individual_roles = [
            role_id
            for role_id
            in DEPARTMENT_ROLE_MAP[
                department_id
            ]
            if role_id != manager_role_id
        ]

        # This fallback protects against a department that
        # has only one available role.
        if not possible_individual_roles:
            possible_individual_roles = [
                manager_role_id
            ]

        # -------------------------------------------------
        # Step 3: Create individual contributors.
        # -------------------------------------------------

        for position_number in range(
            number_of_individual_contributors
        ):
            job_role_id = random.choice(
                possible_individual_roles
            )

            hire_date = random_date(
                date(2021, 1, 1),
                AS_OF_DATE,
            )

            (
                employment_status,
                termination_date,
                termination_type,
            ) = choose_termination(hire_date)

            if team_manager_ids:
                # Assign employees across team managers
                # in a repeating pattern.
                manager_id = team_manager_ids[
                    position_number
                    % len(team_manager_ids)
                ]
            else:
                # In a small department, employees report
                # directly to the department head.
                manager_id = department_head_id

            employee = {
                "employee_id": next_employee_id,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "hire_date": hire_date,
                "termination_date": (
                    termination_date
                ),
                "employment_status": (
                    employment_status
                ),
                "termination_type": (
                    termination_type
                ),
                "department_id": department_id,
                "location_id": (
                    choose_from_weight_map(
                        LOCATION_WEIGHTS
                    )
                ),
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


def validate_employees(
    employees: pd.DataFrame,
    departments: pd.DataFrame,
    locations: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> None:
    """Validate employee and hierarchy rules."""

    # ---------------------------------------------
    # Primary-key checks
    # ---------------------------------------------

    if employees["employee_id"].isna().any():
        raise ValueError(
            "employees: employee_id contains "
            "missing values."
        )

    if employees[
        "employee_id"
    ].duplicated().any():
        raise ValueError(
            "employees: employee_id contains "
            "duplicates."
        )

    # ---------------------------------------------
    # Foreign-key checks
    # ---------------------------------------------

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
            "employees: one or more "
            "department IDs are invalid."
        )

    if not set(
        employees["location_id"]
    ).issubset(valid_location_ids):
        raise ValueError(
            "employees: one or more "
            "location IDs are invalid."
        )

    if not set(
        employees["job_role_id"]
    ).issubset(valid_job_role_ids):
        raise ValueError(
            "employees: one or more "
            "job-role IDs are invalid."
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
            "employees: one or more manager IDs "
            "are not valid employees."
        )

    # ---------------------------------------------
    # Basic employment checks
    # ---------------------------------------------

    self_managed = (
        employees["manager_id"].notna()
        & (
            employees["manager_id"]
            == employees["employee_id"]
        )
    )

    if self_managed.any():
        raise ValueError(
            "employees: an employee cannot "
            "manage themselves."
        )

    invalid_role_assignment = (
        employees.apply(
            lambda row: (
                int(row["job_role_id"])
                not in DEPARTMENT_ROLE_MAP[
                    int(row["department_id"])
                ]
            ),
            axis=1,
        )
    )

    if invalid_role_assignment.any():
        raise ValueError(
            "employees: one or more job roles "
            "do not match the department."
        )

    as_of_timestamp = pd.Timestamp(
        AS_OF_DATE
    )

    if (
        employees["hire_date"]
        > as_of_timestamp
    ).any():
        raise ValueError(
            "employees: a hire date occurs "
            "after the analysis date."
        )

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
            "employees: a termination date "
            "occurs before a hire date."
        )

    # ---------------------------------------------
    # Organizational-level checks
    # ---------------------------------------------

    allowed_levels = {
        "Department Head",
        "Team Manager",
        "Individual Contributor",
    }

    if not set(
        employees[
            "organizational_level"
        ]
    ).issubset(allowed_levels):
        raise ValueError(
            "employees: an organizational level "
            "is invalid."
        )

    department_heads = employees[
        employees[
            "organizational_level"
        ]
        == "Department Head"
    ]

    if department_heads[
        "manager_id"
    ].notna().any():
        raise ValueError(
            "employees: department heads "
            "should not have managers."
        )

    if (
        len(department_heads)
        != len(departments)
    ):
        raise ValueError(
            "employees: every department must "
            "have exactly one department head."
        )

    individual_contributors = employees[
        employees[
            "organizational_level"
        ]
        == "Individual Contributor"
    ]

    if individual_contributors[
        "manager_id"
    ].isna().any():
        raise ValueError(
            "employees: every individual "
            "contributor must have a manager."
        )

    team_managers = employees[
        employees[
            "organizational_level"
        ]
        == "Team Manager"
    ]

    if team_managers[
        "manager_id"
    ].isna().any():
        raise ValueError(
            "employees: every team manager "
            "must report to a department head."
        )

    # ---------------------------------------------
    # Manager relationship checks
    # ---------------------------------------------

    manager_lookup = (
        employees
        .set_index("employee_id")
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
            manager_lookup[
                "department_id"
            ]
        )
    )

    if (
        managed_employees[
            "department_id"
        ]
        != managed_employees[
            "manager_department_id"
        ]
    ).any():
        raise ValueError(
            "employees: an employee and manager "
            "belong to different departments."
        )

    managed_employees[
        "manager_status"
    ] = (
        managed_employees["manager_id"]
        .astype(int)
        .map(
            manager_lookup[
                "employment_status"
            ]
        )
    )

    if (
        managed_employees[
            "manager_status"
        ]
        != "Active"
    ).any():
        raise ValueError(
            "employees: an employee reports "
            "to an inactive manager."
        )

    team_manager_details = (
        team_managers.copy()
    )

    team_manager_details[
        "manager_level"
    ] = (
        team_manager_details[
            "manager_id"
        ]
        .astype(int)
        .map(
            manager_lookup[
                "organizational_level"
            ]
        )
    )

    if (
        team_manager_details[
            "manager_level"
        ]
        != "Department Head"
    ).any():
        raise ValueError(
            "employees: team managers must "
            "report to department heads."
        )

    # ---------------------------------------------
    # Span-of-control check
    # ---------------------------------------------

    direct_report_counts = (
        employees["manager_id"]
        .dropna()
        .astype(int)
        .value_counts()
    )

    if (
        direct_report_counts
        > MAX_DIRECT_REPORTS
    ).any():
        excessive_managers = (
            direct_report_counts[
                direct_report_counts
                > MAX_DIRECT_REPORTS
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "employees: managers exceed the "
            f"{MAX_DIRECT_REPORTS}-report limit. "
            f"Manager IDs: {excessive_managers}"
        )


def save_employees(
    employees: pd.DataFrame,
) -> None:
    """Save the improved employee sample."""

    output_path = (
        RAW_DATA_DIR
        / "employees_sample.csv"
    )

    employees.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved employees_sample.csv: "
        f"{len(employees)} rows and "
        f"{len(employees.columns)} columns"
    )


def print_summary(
    employees: pd.DataFrame,
) -> None:
    """Print hierarchy and workforce summaries."""

    print("\nEmployment status:")
    print(
        employees[
            "employment_status"
        ].value_counts()
    )

    print("\nOrganizational levels:")
    print(
        employees[
            "organizational_level"
        ].value_counts()
    )

    print("\nEmployees by department_id:")
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

    print("\nLargest number of direct reports:")
    print(
        int(
            direct_report_counts.max()
        )
    )


def main() -> None:
    """Run the improved employee-generation process."""

    departments = load_reference_table(
        "departments.csv"
    )

    locations = load_reference_table(
        "locations.csv"
    )

    job_roles = load_reference_table(
        "job_roles.csv"
    )

    employees = create_employees(
        departments,
        locations,
        job_roles,
    )

    validate_employees(
        employees,
        departments,
        locations,
        job_roles,
    )

    save_employees(employees)

    print_summary(employees)

    print(
        "\nImproved employee hierarchy "
        "generated successfully."
    )


if __name__ == "__main__":
    main()