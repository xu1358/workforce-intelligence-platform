from datetime import date, timedelta
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

FIRST_REQUISITION_ID = 300_001

OPEN_REQUISITION_COUNT = 120
CANCELLED_REQUISITION_COUNT = 250


ALLOWED_STATUSES = {
    "Open",
    "Filled",
    "Cancelled",
}


RECRUITING_DEPARTMENT_KEYWORDS = (
    "human",
    "people",
    "talent",
)


# ---------------------------------------------------------
# File-loading helper
# ---------------------------------------------------------

def load_table(
    filename: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file from data/raw."""

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
# Date helper
# ---------------------------------------------------------

def random_date(
    start_date: date,
    end_date: date,
) -> date:
    """Choose a random date inside a valid range."""

    if start_date > end_date:
        raise ValueError(
            "start_date cannot occur after end_date."
        )

    number_of_days = (
        end_date - start_date
    ).days

    return start_date + timedelta(
        days=random.randint(
            0,
            number_of_days,
        )
    )


# ---------------------------------------------------------
# Recruiter helpers
# ---------------------------------------------------------

def build_recruiter_pool(
    employees: pd.DataFrame,
    departments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prefer employees from Human Resources,
    People, or Talent departments.
    """

    department_names = (
        departments[
            "department_name"
        ]
        .astype(str)
        .str.lower()
    )

    recruiting_department_mask = (
        pd.Series(
            False,
            index=departments.index,
        )
    )

    for keyword in (
        RECRUITING_DEPARTMENT_KEYWORDS
    ):
        recruiting_department_mask |= (
            department_names.str.contains(
                keyword,
                regex=False,
            )
        )

    recruiting_department_ids = set(
        departments.loc[
            recruiting_department_mask,
            "department_id",
        ].astype(int)
    )

    recruiter_pool = employees[
        employees[
            "department_id"
        ]
        .astype(int)
        .isin(
            recruiting_department_ids
        )
    ].copy()

    # This fallback makes the generator work even
    # if the company does not have a department
    # containing Human, People, or Talent.
    if recruiter_pool.empty:
        recruiter_pool = employees.copy()

    return recruiter_pool


def choose_recruiter(
    recruiter_pool: pd.DataFrame,
    employees: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> int:
    """
    Choose an employee who was employed throughout
    the entire requisition period.
    """

    start_timestamp = pd.Timestamp(
        start_date
    )

    end_timestamp = pd.Timestamp(
        end_date
    )

    def find_eligible_people(
        source: pd.DataFrame,
    ) -> pd.DataFrame:
        return source[
            source[
                "hire_date"
            ].le(
                start_timestamp
            )
            & (
                source[
                    "termination_date"
                ].isna()
                | source[
                    "termination_date"
                ].ge(
                    end_timestamp
                )
            )
        ]

    eligible_recruiters = (
        find_eligible_people(
            recruiter_pool
        )
    )

    # Use another employee only when no recruiting
    # department employee was available at that time.
    if eligible_recruiters.empty:
        eligible_recruiters = (
            find_eligible_people(
                employees
            )
        )

    if eligible_recruiters.empty:
        raise ValueError(
            "No employee is available to recruit "
            f"from {start_date} through {end_date}."
        )

    eligible_recruiter_ids = sorted(
        eligible_recruiters[
            "employee_id"
        ]
        .astype(int)
        .tolist()
    )

    return int(
        random.choice(
            eligible_recruiter_ids
        )
    )


# ---------------------------------------------------------
# Job-combination helper
# ---------------------------------------------------------

def get_valid_job_combinations(
    employees: pd.DataFrame,
    active_only: bool = False,
) -> list[dict]:
    """
    Return role, department, and location combinations
    that already exist in employee data.
    """

    source = employees

    if active_only:
        source = source[
            source[
                "employment_status"
            ].eq("Active")
        ]

    combinations = (
        source[
            [
                "job_role_id",
                "department_id",
                "location_id",
            ]
        ]
        .drop_duplicates()
        .astype(int)
    )

    if combinations.empty:
        raise ValueError(
            "No valid job combinations are available."
        )

    return combinations.to_dict(
        orient="records"
    )


# ---------------------------------------------------------
# Filled historical requisitions
# ---------------------------------------------------------

def add_filled_requisitions(
    employees: pd.DataFrame,
    recruiter_pool: pd.DataFrame,
    requisition_records: list[dict],
) -> None:
    """
    Create one filled requisition for every unique
    role-department-location-hiring-quarter group.
    """

    earliest_company_date = (
        pd.Timestamp(
            employees[
                "hire_date"
            ].min()
        ).date()
    )

    employee_groups = employees.copy()

    employee_groups[
        "_hire_quarter"
    ] = (
        employee_groups[
            "hire_date"
        ].dt.to_period("Q")
    )

    grouped_employees = (
        employee_groups.groupby(
            [
                "job_role_id",
                "department_id",
                "location_id",
                "_hire_quarter",
            ],
            sort=True,
        )
    )

    for (
        job_role_id,
        department_id,
        location_id,
        _,
    ), group in grouped_employees:

        earliest_hire_date = (
            pd.Timestamp(
                group[
                    "hire_date"
                ].min()
            ).date()
        )

        latest_hire_date = (
            pd.Timestamp(
                group[
                    "hire_date"
                ].max()
            ).date()
        )

        proposed_open_date = (
            earliest_hire_date
            - timedelta(
                days=random.randint(
                    30,
                    75,
                )
            )
        )

        # Do not create requisitions before the first
        # date represented in company history.
        open_date = max(
            proposed_open_date,
            earliest_company_date,
        )

        recruiter_id = choose_recruiter(
            recruiter_pool=(
                recruiter_pool
            ),
            employees=employees,
            start_date=open_date,
            end_date=latest_hire_date,
        )

        requisition_records.append(
            {
                "job_role_id": int(
                    job_role_id
                ),
                "department_id": int(
                    department_id
                ),
                "location_id": int(
                    location_id
                ),
                "open_date": open_date,
                "close_date": (
                    latest_hire_date
                ),
                "target_headcount": int(
                    len(group)
                ),
                "recruiter_id": (
                    recruiter_id
                ),
                "requisition_status": (
                    "Filled"
                ),
            }
        )


# ---------------------------------------------------------
# Cancelled requisitions
# ---------------------------------------------------------

def add_cancelled_requisitions(
    employees: pd.DataFrame,
    recruiter_pool: pd.DataFrame,
    requisition_records: list[dict],
) -> None:
    """Create requisitions that closed without hiring."""

    valid_combinations = (
        get_valid_job_combinations(
            employees
        )
    )

    earliest_open_date = (
        pd.Timestamp(
            employees[
                "hire_date"
            ].min()
        ).date()
    )

    latest_open_date = (
        AS_OF_DATE
        - timedelta(days=10)
    )

    used_keys = set()
    created_count = 0
    attempt_count = 0

    while (
        created_count
        < CANCELLED_REQUISITION_COUNT
    ):
        attempt_count += 1

        if attempt_count > 20_000:
            raise ValueError(
                "Unable to create enough "
                "cancelled requisitions."
            )

        combination = random.choice(
            valid_combinations
        )

        open_date = random_date(
            earliest_open_date,
            latest_open_date,
        )

        close_date = min(
            open_date
            + timedelta(
                days=random.randint(
                    10,
                    90,
                )
            ),
            AS_OF_DATE,
        )

        unique_key = (
            combination[
                "job_role_id"
            ],
            combination[
                "department_id"
            ],
            combination[
                "location_id"
            ],
            open_date,
        )

        if unique_key in used_keys:
            continue

        recruiter_id = choose_recruiter(
            recruiter_pool=(
                recruiter_pool
            ),
            employees=employees,
            start_date=open_date,
            end_date=close_date,
        )

        requisition_records.append(
            {
                **combination,
                "open_date": open_date,
                "close_date": close_date,
                "target_headcount": (
                    random.randint(
                        1,
                        6,
                    )
                ),
                "recruiter_id": (
                    recruiter_id
                ),
                "requisition_status": (
                    "Cancelled"
                ),
            }
        )

        used_keys.add(
            unique_key
        )

        created_count += 1


# ---------------------------------------------------------
# Current open requisitions
# ---------------------------------------------------------

def add_open_requisitions(
    employees: pd.DataFrame,
    recruiter_pool: pd.DataFrame,
    requisition_records: list[dict],
) -> None:
    """Create requisitions still open on June 30, 2026."""

    valid_combinations = (
        get_valid_job_combinations(
            employees,
            active_only=True,
        )
    )

    earliest_open_date = (
        AS_OF_DATE
        - timedelta(days=120)
    )

    used_keys = set()
    created_count = 0
    attempt_count = 0

    while (
        created_count
        < OPEN_REQUISITION_COUNT
    ):
        attempt_count += 1

        if attempt_count > 20_000:
            raise ValueError(
                "Unable to create enough "
                "open requisitions."
            )

        combination = random.choice(
            valid_combinations
        )

        open_date = random_date(
            earliest_open_date,
            AS_OF_DATE,
        )

        unique_key = (
            combination[
                "job_role_id"
            ],
            combination[
                "department_id"
            ],
            combination[
                "location_id"
            ],
            open_date,
        )

        if unique_key in used_keys:
            continue

        recruiter_id = choose_recruiter(
            recruiter_pool=(
                recruiter_pool
            ),
            employees=employees,
            start_date=open_date,
            end_date=AS_OF_DATE,
        )

        requisition_records.append(
            {
                **combination,
                "open_date": open_date,
                "close_date": None,
                "target_headcount": (
                    random.randint(
                        1,
                        8,
                    )
                ),
                "recruiter_id": (
                    recruiter_id
                ),
                "requisition_status": (
                    "Open"
                ),
            }
        )

        used_keys.add(
            unique_key
        )

        created_count += 1


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------

def create_job_requisitions(
    employees: pd.DataFrame,
    departments: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical and current requisitions."""

    random.seed(
        RANDOM_SEED
    )

    recruiter_pool = (
        build_recruiter_pool(
            employees,
            departments,
        )
    )

    requisition_records = []

    add_filled_requisitions(
        employees=employees,
        recruiter_pool=(
            recruiter_pool
        ),
        requisition_records=(
            requisition_records
        ),
    )

    add_cancelled_requisitions(
        employees=employees,
        recruiter_pool=(
            recruiter_pool
        ),
        requisition_records=(
            requisition_records
        ),
    )

    add_open_requisitions(
        employees=employees,
        recruiter_pool=(
            recruiter_pool
        ),
        requisition_records=(
            requisition_records
        ),
    )

    job_requisitions = pd.DataFrame(
        requisition_records
    )

    job_requisitions[
        "open_date"
    ] = pd.to_datetime(
        job_requisitions[
            "open_date"
        ]
    )

    job_requisitions[
        "close_date"
    ] = pd.to_datetime(
        job_requisitions[
            "close_date"
        ]
    )

    status_order = {
        "Filled": 1,
        "Cancelled": 2,
        "Open": 3,
    }

    job_requisitions[
        "_status_order"
    ] = (
        job_requisitions[
            "requisition_status"
        ].map(
            status_order
        )
    )

    job_requisitions = (
        job_requisitions
        .sort_values(
            [
                "open_date",
                "_status_order",
                "department_id",
                "job_role_id",
                "location_id",
            ]
        )
        .drop(
            columns=[
                "_status_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    job_requisitions.insert(
        0,
        "requisition_id",
        range(
            FIRST_REQUISITION_ID,
            FIRST_REQUISITION_ID
            + len(job_requisitions),
        ),
    )

    return job_requisitions[
        [
            "requisition_id",
            "job_role_id",
            "department_id",
            "location_id",
            "open_date",
            "close_date",
            "target_headcount",
            "recruiter_id",
            "requisition_status",
        ]
    ]


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_job_requisitions(
    job_requisitions: pd.DataFrame,
    employees: pd.DataFrame,
    departments: pd.DataFrame,
    locations: pd.DataFrame,
    job_roles: pd.DataFrame,
) -> None:
    """Validate job-requisition business rules."""

    expected_columns = [
        "requisition_id",
        "job_role_id",
        "department_id",
        "location_id",
        "open_date",
        "close_date",
        "target_headcount",
        "recruiter_id",
        "requisition_status",
    ]

    if (
        job_requisitions.columns.tolist()
        != expected_columns
    ):
        raise ValueError(
            "job_requisitions: unexpected "
            "columns or column order."
        )

    if job_requisitions.empty:
        raise ValueError(
            "job_requisitions: no records generated."
        )

    required_columns = [
        column
        for column in expected_columns
        if column != "close_date"
    ]

    if job_requisitions[
        required_columns
    ].isna().any().any():
        raise ValueError(
            "job_requisitions: missing "
            "required values."
        )

    if not job_requisitions[
        "requisition_id"
    ].is_unique:
        raise ValueError(
            "job_requisitions: requisition "
            "IDs are not unique."
        )

    expected_requisition_ids = list(
        range(
            FIRST_REQUISITION_ID,
            FIRST_REQUISITION_ID
            + len(job_requisitions),
        )
    )

    actual_requisition_ids = (
        job_requisitions[
            "requisition_id"
        ]
        .astype(int)
        .tolist()
    )

    if (
        actual_requisition_ids
        != expected_requisition_ids
    ):
        raise ValueError(
            "job_requisitions: requisition "
            "IDs are not sequential."
        )

    if not set(
        job_requisitions[
            "requisition_status"
        ]
    ).issubset(
        ALLOWED_STATUSES
    ):
        raise ValueError(
            "job_requisitions: invalid status."
        )

    if set(
        job_requisitions[
            "requisition_status"
        ]
    ) != ALLOWED_STATUSES:
        raise ValueError(
            "job_requisitions: not every "
            "status appears."
        )

    valid_job_role_ids = set(
        job_roles[
            "job_role_id"
        ].astype(int)
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

    valid_employee_ids = set(
        employees[
            "employee_id"
        ].astype(int)
    )

    if not set(
        job_requisitions[
            "job_role_id"
        ].astype(int)
    ).issubset(
        valid_job_role_ids
    ):
        raise ValueError(
            "job_requisitions: invalid "
            "job role ID."
        )

    if not set(
        job_requisitions[
            "department_id"
        ].astype(int)
    ).issubset(
        valid_department_ids
    ):
        raise ValueError(
            "job_requisitions: invalid "
            "department ID."
        )

    if not set(
        job_requisitions[
            "location_id"
        ].astype(int)
    ).issubset(
        valid_location_ids
    ):
        raise ValueError(
            "job_requisitions: invalid "
            "location ID."
        )

    if not set(
        job_requisitions[
            "recruiter_id"
        ].astype(int)
    ).issubset(
        valid_employee_ids
    ):
        raise ValueError(
            "job_requisitions: invalid "
            "recruiter ID."
        )

    if not job_requisitions[
        "target_headcount"
    ].ge(1).all():
        raise ValueError(
            "job_requisitions: target "
            "headcount must be positive."
        )

    if not job_requisitions[
        "target_headcount"
    ].astype(float).mod(1).eq(0).all():
        raise ValueError(
            "job_requisitions: target headcount "
            "must use whole numbers."
        )

    if job_requisitions[
        "open_date"
    ].gt(
        pd.Timestamp(
            AS_OF_DATE
        )
    ).any():
        raise ValueError(
            "job_requisitions: a requisition "
            "opens after the analysis date."
        )

    open_requisitions = (
        job_requisitions[
            job_requisitions[
                "requisition_status"
            ].eq("Open")
        ]
    )

    closed_requisitions = (
        job_requisitions[
            job_requisitions[
                "requisition_status"
            ].isin(
                [
                    "Filled",
                    "Cancelled",
                ]
            )
        ]
    )

    if open_requisitions[
        "close_date"
    ].notna().any():
        raise ValueError(
            "job_requisitions: an open "
            "requisition has a close date."
        )

    if closed_requisitions[
        "close_date"
    ].isna().any():
        raise ValueError(
            "job_requisitions: a closed "
            "requisition is missing a close date."
        )

    if closed_requisitions[
        "close_date"
    ].lt(
        closed_requisitions[
            "open_date"
        ]
    ).any():
        raise ValueError(
            "job_requisitions: close date "
            "occurs before open date."
        )

    if closed_requisitions[
        "close_date"
    ].gt(
        pd.Timestamp(
            AS_OF_DATE
        )
    ).any():
        raise ValueError(
            "job_requisitions: close date "
            "occurs after the analysis date."
        )

    # Validate that recruiters were employed during
    # the complete requisition period.
    recruiter_details = (
        employees[
            [
                "employee_id",
                "hire_date",
                "termination_date",
            ]
        ]
        .rename(
            columns={
                "employee_id": (
                    "recruiter_id"
                ),
                "hire_date": (
                    "recruiter_hire_date"
                ),
                "termination_date": (
                    "recruiter_termination_date"
                ),
            }
        )
    )

    requisition_details = (
        job_requisitions
        .merge(
            recruiter_details,
            on="recruiter_id",
            how="left",
        )
    )

    requisition_details[
        "requisition_end_date"
    ] = (
        requisition_details[
            "close_date"
        ]
        .fillna(
            pd.Timestamp(
                AS_OF_DATE
            )
        )
    )

    if requisition_details[
        "recruiter_hire_date"
    ].gt(
        requisition_details[
            "open_date"
        ]
    ).any():
        raise ValueError(
            "job_requisitions: recruiter was "
            "hired after requisition opened."
        )

    recruiter_left_too_early = (
        requisition_details[
            "recruiter_termination_date"
        ].notna()
        & requisition_details[
            "recruiter_termination_date"
        ].lt(
            requisition_details[
                "requisition_end_date"
            ]
        )
    )

    if recruiter_left_too_early.any():
        raise ValueError(
            "job_requisitions: recruiter left "
            "before requisition ended."
        )

    # Every requisition must use a role, department,
    # and location combination found in employee data.
    valid_combinations = set(
        employees[
            [
                "job_role_id",
                "department_id",
                "location_id",
            ]
        ]
        .astype(int)
        .itertuples(
            index=False,
            name=None,
        )
    )

    used_combinations = set(
        job_requisitions[
            [
                "job_role_id",
                "department_id",
                "location_id",
            ]
        ]
        .astype(int)
        .itertuples(
            index=False,
            name=None,
        )
    )

    if not used_combinations.issubset(
        valid_combinations
    ):
        raise ValueError(
            "job_requisitions: an invalid "
            "role-department-location "
            "combination appears."
        )

    # Rebuild the expected quarterly hiring groups.
    expected_groups = employees.copy()

    expected_groups[
        "hire_quarter"
    ] = (
        expected_groups[
            "hire_date"
        ].dt.to_period("Q")
    )

    expected_groups = (
        expected_groups
        .groupby(
            [
                "job_role_id",
                "department_id",
                "location_id",
                "hire_quarter",
            ],
            as_index=False,
        )
        .agg(
            expected_headcount=(
                "employee_id",
                "size",
            ),
            earliest_hire_date=(
                "hire_date",
                "min",
            ),
            latest_hire_date=(
                "hire_date",
                "max",
            ),
        )
    )

    filled_requisitions = (
        job_requisitions[
            job_requisitions[
                "requisition_status"
            ].eq("Filled")
        ]
        .copy()
    )

    filled_requisitions[
        "hire_quarter"
    ] = (
        filled_requisitions[
            "close_date"
        ].dt.to_period("Q")
    )

    group_keys = [
        "job_role_id",
        "department_id",
        "location_id",
        "hire_quarter",
    ]

    if filled_requisitions.duplicated(
        subset=group_keys
    ).any():
        raise ValueError(
            "job_requisitions: duplicate filled "
            "hiring-quarter group."
        )

    alignment = (
        expected_groups
        .merge(
            filled_requisitions[
                group_keys
                + [
                    "open_date",
                    "close_date",
                    "target_headcount",
                ]
            ],
            on=group_keys,
            how="outer",
            indicator=True,
        )
    )

    if not alignment[
        "_merge"
    ].eq("both").all():
        raise ValueError(
            "job_requisitions: filled requisitions "
            "do not cover every employee "
            "hiring group."
        )

    if not alignment[
        "target_headcount"
    ].eq(
        alignment[
            "expected_headcount"
        ]
    ).all():
        raise ValueError(
            "job_requisitions: filled target "
            "headcount does not match hires."
        )

    if not alignment[
        "open_date"
    ].le(
        alignment[
            "earliest_hire_date"
        ]
    ).all():
        raise ValueError(
            "job_requisitions: a filled "
            "requisition opens after its first hire."
        )

    if not alignment[
        "close_date"
    ].eq(
        alignment[
            "latest_hire_date"
        ]
    ).all():
        raise ValueError(
            "job_requisitions: filled close date "
            "does not match its last hire."
        )

    if (
        len(open_requisitions)
        != OPEN_REQUISITION_COUNT
    ):
        raise ValueError(
            "job_requisitions: unexpected "
            "open requisition count."
        )

    cancelled_count = (
        job_requisitions[
            "requisition_status"
        ].eq("Cancelled").sum()
    )

    if (
        cancelled_count
        != CANCELLED_REQUISITION_COUNT
    ):
        raise ValueError(
            "job_requisitions: unexpected "
            "cancelled requisition count."
        )

    filled_target_total = int(
        filled_requisitions[
            "target_headcount"
        ].sum()
    )

    if filled_target_total != len(
        employees
    ):
        raise ValueError(
            "job_requisitions: filled target "
            "headcount does not sum to "
            "the employee count."
        )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_job_requisitions(
    job_requisitions: pd.DataFrame,
) -> None:
    """Save job_requisitions.csv."""

    output_path = (
        RAW_DATA_DIR
        / "job_requisitions.csv"
    )

    job_requisitions.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved job_requisitions.csv: "
        f"{len(job_requisitions)} rows and "
        f"{len(job_requisitions.columns)} columns"
    )


def print_summary(
    job_requisitions: pd.DataFrame,
) -> None:
    """Print basic requisition statistics."""

    print(
        "\nRequisitions by status:"
    )

    print(
        job_requisitions[
            "requisition_status"
        ].value_counts()
    )

    filled_requisitions = (
        job_requisitions[
            job_requisitions[
                "requisition_status"
            ].eq("Filled")
        ]
    )

    print(
        "\nFilled target headcount:"
    )

    print(
        int(
            filled_requisitions[
                "target_headcount"
            ].sum()
        )
    )

    print(
        "\nAverage target headcount:"
    )

    print(
        round(
            job_requisitions[
                "target_headcount"
            ].mean(),
            2,
        )
    )

    print(
        "\nLargest target headcount:"
    )

    print(
        int(
            job_requisitions[
                "target_headcount"
            ].max()
        )
    )

    print(
        "\nRequisition open-date range:"
    )

    print(
        job_requisitions[
            "open_date"
        ].min().date(),
        "to",
        job_requisitions[
            "open_date"
        ].max().date(),
    )


def main() -> None:
    """Generate and validate job requisitions."""

    employees = load_table(
        "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    departments = load_table(
        "departments.csv"
    )

    locations = load_table(
        "locations.csv"
    )

    job_roles = load_table(
        "job_roles.csv"
    )

    job_requisitions = (
        create_job_requisitions(
            employees=employees,
            departments=departments,
        )
    )

    validate_job_requisitions(
        job_requisitions=(
            job_requisitions
        ),
        employees=employees,
        departments=departments,
        locations=locations,
        job_roles=job_roles,
    )

    save_job_requisitions(
        job_requisitions
    )

    print_summary(
        job_requisitions
    )

    print(
        "\nJob requisitions generated "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()