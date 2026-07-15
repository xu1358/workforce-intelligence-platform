from datetime import date, timedelta
from pathlib import Path
import random

import pandas as pd
from faker import Faker


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder containing our generated CSV files.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Use a fixed analysis date so results are reproducible.
AS_OF_DATE = date(2026, 6, 30)

# Using the same seed produces the same random results each time.
RANDOM_SEED = 42

# Begin with a small sample.
NUM_EMPLOYEES = 50


# Define which job roles are reasonable for each department.
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


# Control the approximate percentage of employees in each department.
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


# Assign one initial manager role to each department.
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


# Control the approximate percentage of employees at each location.
LOCATION_WEIGHTS = {
    1: 0.28,
    2: 0.24,
    3: 0.18,
    4: 0.12,
    5: 0.18,
}


# These roles will be classified as hourly.
HOURLY_ROLE_IDS = {7, 20}


def load_reference_table(filename: str) -> pd.DataFrame:
    """Load one of the previously generated reference tables."""

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

    total_days = (end_date - start_date).days

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
    """Select an ID using predefined probabilities."""

    values = list(weight_map.keys())
    weights = list(weight_map.values())

    selected_value = random.choices(
        values,
        weights=weights,
        k=1,
    )[0]

    return selected_value


def choose_education(job_level: int) -> str:
    """Choose an education level based partly on job level."""

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

    # Employees hired very recently will remain active.
    if days_since_hire < 120:
        return "Active", None, None

    # Approximately 22% of eligible employees will be terminated.
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


def create_employees(
    departments: pd.DataFrame,
    locations: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> pd.DataFrame:
    """Create the synthetic employee table."""

    # Reset random generators for reproducibility.
    random.seed(RANDOM_SEED)
    Faker.seed(RANDOM_SEED)

    fake = Faker("en_US")

    valid_department_ids = sorted(
        departments["department_id"]
        .astype(int)
        .tolist()
    )

    valid_location_ids = sorted(
        locations["location_id"]
        .astype(int)
        .tolist()
    )

    job_level_by_role = (
        job_roles
        .set_index("job_role_id")["job_level"]
        .astype(int)
        .to_dict()
    )

    employees = []

    manager_id_by_department = {}

    next_employee_id = 10001

    # First, create one active manager for each department.
    for department_id in valid_department_ids:
        job_role_id = (
            MANAGER_ROLE_BY_DEPARTMENT[
                department_id
            ]
        )

        location_id = random.choice(
            valid_location_ids
        )

        hire_date = random_date(
            date(2021, 1, 1),
            date(2022, 12, 31),
        )

        manager_record = {
            "employee_id": next_employee_id,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "hire_date": hire_date,
            "termination_date": None,
            "employment_status": "Active",
            "termination_type": None,
            "department_id": department_id,
            "location_id": location_id,
            "job_role_id": job_role_id,
            "manager_id": None,
            "employment_type": (
                "Hourly"
                if job_role_id in HOURLY_ROLE_IDS
                else "Salaried"
            ),
            "birth_year": random.randint(
                1965,
                1998,
            ),
            "education_level": choose_education(
                job_level_by_role[job_role_id]
            ),
        }

        employees.append(manager_record)

        manager_id_by_department[
            department_id
        ] = next_employee_id

        next_employee_id += 1

    # Generate the remaining employees.
    while len(employees) < NUM_EMPLOYEES:
        department_id = choose_from_weight_map(
            DEPARTMENT_WEIGHTS
        )

        job_role_id = random.choice(
            DEPARTMENT_ROLE_MAP[
                department_id
            ]
        )

        location_id = choose_from_weight_map(
            LOCATION_WEIGHTS
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

        employee_record = {
            "employee_id": next_employee_id,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "hire_date": hire_date,
            "termination_date": termination_date,
            "employment_status": employment_status,
            "termination_type": termination_type,
            "department_id": department_id,
            "location_id": location_id,
            "job_role_id": job_role_id,
            "manager_id": (
                manager_id_by_department[
                    department_id
                ]
            ),
            "employment_type": (
                "Hourly"
                if job_role_id in HOURLY_ROLE_IDS
                else "Salaried"
            ),
            "birth_year": random.randint(
                1962,
                2003,
            ),
            "education_level": choose_education(
                job_level_by_role[job_role_id]
            ),
        }

        employees.append(employee_record)

        next_employee_id += 1

    employee_df = pd.DataFrame(employees)

    employee_df["hire_date"] = pd.to_datetime(
        employee_df["hire_date"]
    )

    employee_df["termination_date"] = pd.to_datetime(
        employee_df["termination_date"]
    )

    # Int64 allows integer values together with missing values.
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
    """Check whether the employee data follows our rules."""

    if employees["employee_id"].isna().any():
        raise ValueError(
            "employees: employee_id contains "
            "missing values."
        )

    if employees["employee_id"].duplicated().any():
        raise ValueError(
            "employees: employee_id contains duplicates."
        )

    valid_department_ids = set(
        departments["department_id"].astype(int)
    )

    valid_location_ids = set(
        locations["location_id"].astype(int)
    )

    valid_job_role_ids = set(
        job_roles["job_role_id"].astype(int)
    )

    valid_employee_ids = set(
        employees["employee_id"].astype(int)
    )

    if not set(
        employees["department_id"]
    ).issubset(valid_department_ids):
        raise ValueError(
            "employees: one or more department_id "
            "values are invalid."
        )

    if not set(
        employees["location_id"]
    ).issubset(valid_location_ids):
        raise ValueError(
            "employees: one or more location_id "
            "values are invalid."
        )

    if not set(
        employees["job_role_id"]
    ).issubset(valid_job_role_ids):
        raise ValueError(
            "employees: one or more job_role_id "
            "values are invalid."
        )

    manager_ids = set(
        employees["manager_id"]
        .dropna()
        .astype(int)
    )

    if not manager_ids.issubset(
        valid_employee_ids
    ):
        raise ValueError(
            "employees: one or more manager_id "
            "values are invalid."
        )

    self_managed = (
        employees["manager_id"].notna()
        & (
            employees["manager_id"].astype("Int64")
            == employees["employee_id"].astype("Int64")
        )
    )

    if self_managed.any():
        raise ValueError(
            "employees: an employee cannot be "
            "their own manager."
        )

    invalid_role_assignment = employees.apply(
        lambda row: int(row["job_role_id"])
        not in DEPARTMENT_ROLE_MAP[
            int(row["department_id"])
        ],
        axis=1,
    )

    if invalid_role_assignment.any():
        raise ValueError(
            "employees: one or more job roles "
            "do not match their assigned department."
        )

    as_of_timestamp = pd.Timestamp(AS_OF_DATE)

    if (
        employees["hire_date"]
        > as_of_timestamp
    ).any():
        raise ValueError(
            "employees: a hire date occurs "
            "after the analysis date."
        )

    terminated = (
        employees["employment_status"]
        == "Terminated"
    )

    active = (
        employees["employment_status"]
        == "Active"
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

    if employees.loc[
        active,
        "termination_date",
    ].notna().any():
        raise ValueError(
            "employees: an active employee "
            "has a termination date."
        )

    if employees.loc[
        active,
        "termination_type",
    ].notna().any():
        raise ValueError(
            "employees: an active employee "
            "has a termination type."
        )

    invalid_termination_dates = (
        terminated
        & (
            employees["termination_date"]
            < employees["hire_date"]
        )
    )

    if invalid_termination_dates.any():
        raise ValueError(
            "employees: a termination date occurs "
            "before a hire date."
        )


def save_employees(
    employees: pd.DataFrame,
) -> None:
    """Save the employee table as a CSV file."""

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
    """Print a simple summary of the generated table."""

    print("\nEmployment status:")
    print(
        employees["employment_status"]
        .value_counts()
    )

    print("\nEmployees by department_id:")
    print(
        employees["department_id"]
        .value_counts()
        .sort_index()
    )

    print("\nFirst five employees:")
    print(employees.head())


def main() -> None:
    """Run the complete employee-generation process."""

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
        "\nEmployee sample generated successfully."
    )


if __name__ == "__main__":
    main()