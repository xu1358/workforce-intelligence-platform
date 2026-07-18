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

FIRST_APPLICATION_ID = 400_001


ALLOWED_APPLICATION_STATUSES = {
    "Hired",
    "Rejected",
    "Withdrawn",
    "Offer Declined",
    "In Process",
    "Position Cancelled",
}


REQUISITION_STATUS_CHOICES = [
    "Filled",
    "Cancelled",
    "Open",
]


REQUISITION_STATUS_WEIGHTS = [
    0.55,
    0.20,
    0.25,
]


GENERAL_APPLICATION_COUNTS = [
    1,
    2,
    3,
]


GENERAL_APPLICATION_COUNT_WEIGHTS = [
    0.65,
    0.28,
    0.07,
]


# ---------------------------------------------------------
# File-loading helper
# ---------------------------------------------------------

def load_table(
    filename: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load one CSV table from data/raw."""

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

def random_date(
    start_date: date,
    end_date: date,
) -> date:
    """Choose one random date inside a date range."""

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


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Keep a number inside a given range."""

    return max(
        minimum,
        min(value, maximum),
    )


def append_application(
    application_records: list[dict],
    candidate_id: int,
    requisition_id: int,
    application_date: date,
    application_status: str,
    interview_score: float | None,
    offer_date: date | None,
    decision_date: date | None,
    employee_id: int | None,
) -> None:
    """Add one application record."""

    application_records.append(
        {
            "candidate_id": int(
                candidate_id
            ),
            "requisition_id": int(
                requisition_id
            ),
            "application_date": (
                application_date
            ),
            "application_status": (
                application_status
            ),
            "interview_score": (
                interview_score
            ),
            "offer_date": offer_date,
            "decision_date": (
                decision_date
            ),
            "employee_id": (
                employee_id
            ),
        }
    )


# ---------------------------------------------------------
# Filled-requisition lookup
# ---------------------------------------------------------

def build_filled_requisition_lookup(
    job_requisitions: pd.DataFrame,
) -> dict[tuple, object]:
    """
    Build a lookup using:

    job role
    + department
    + location
    + hiring quarter
    """

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
        ]
        .dt.to_period("Q")
        .astype(str)
    )

    lookup = {}

    for requisition in (
        filled_requisitions.itertuples(
            index=False
        )
    ):
        key = (
            int(
                requisition.job_role_id
            ),
            int(
                requisition.department_id
            ),
            int(
                requisition.location_id
            ),
            str(
                requisition.hire_quarter
            ),
        )

        if key in lookup:
            raise ValueError(
                "job_requisitions: duplicate "
                "filled hiring group."
            )

        lookup[key] = requisition

    if not lookup:
        raise ValueError(
            "job_requisitions: no filled "
            "requisitions were found."
        )

    return lookup


# ---------------------------------------------------------
# Hired applications
# ---------------------------------------------------------

def add_hired_applications(
    employees: pd.DataFrame,
    candidates: pd.DataFrame,
    job_requisitions: pd.DataFrame,
    application_records: list[dict],
) -> None:
    """
    Connect the first candidate block to employees.

    Candidate 200001 aligns with employee 100001,
    candidate 200002 aligns with employee 100002,
    and so on.
    """

    employee_order = (
        employees
        .sort_values(
            "employee_id"
        )
        .reset_index(
            drop=True
        )
    )

    reserved_candidates = (
        candidates
        .sort_values(
            "candidate_id"
        )
        .head(
            len(employee_order)
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(reserved_candidates)
        != len(employee_order)
    ):
        raise ValueError(
            "candidates: not enough reserved "
            "candidates for employees."
        )

    filled_lookup = (
        build_filled_requisition_lookup(
            job_requisitions
        )
    )

    for candidate, employee in zip(
        reserved_candidates.itertuples(
            index=False
        ),
        employee_order.itertuples(
            index=False
        ),
    ):
        hire_date = pd.Timestamp(
            employee.hire_date
        ).date()

        hire_quarter = str(
            pd.Timestamp(
                employee.hire_date
            ).to_period("Q")
        )

        hiring_group_key = (
            int(
                employee.job_role_id
            ),
            int(
                employee.department_id
            ),
            int(
                employee.location_id
            ),
            hire_quarter,
        )

        if (
            hiring_group_key
            not in filled_lookup
        ):
            raise ValueError(
                "No filled requisition matches "
                f"employee {employee.employee_id}."
            )

        requisition = (
            filled_lookup[
                hiring_group_key
            ]
        )

        requisition_open_date = (
            pd.Timestamp(
                requisition.open_date
            ).date()
        )

        earliest_application_date = max(
            requisition_open_date,
            hire_date
            - timedelta(days=90),
        )

        application_date = random_date(
            earliest_application_date,
            hire_date,
        )

        offer_date = random_date(
            application_date,
            hire_date,
        )

        interview_score = round(
            clamp(
                random.gauss(
                    88,
                    6,
                ),
                75,
                100,
            ),
            1,
        )

        append_application(
            application_records=(
                application_records
            ),
            candidate_id=int(
                candidate.candidate_id
            ),
            requisition_id=int(
                requisition.requisition_id
            ),
            application_date=(
                application_date
            ),
            application_status="Hired",
            interview_score=(
                interview_score
            ),
            offer_date=offer_date,
            decision_date=hire_date,
            employee_id=int(
                employee.employee_id
            ),
        )


# ---------------------------------------------------------
# General application outcomes
# ---------------------------------------------------------

def choose_general_outcome(
    requisition_status: str,
) -> str:
    """Choose a non-hired application result."""

    if requisition_status == "Filled":
        return random.choices(
            [
                "Rejected",
                "Withdrawn",
                "Offer Declined",
            ],
            weights=[
                0.70,
                0.15,
                0.15,
            ],
            k=1,
        )[0]

    if requisition_status == "Cancelled":
        return random.choices(
            [
                "Position Cancelled",
                "Withdrawn",
            ],
            weights=[
                0.85,
                0.15,
            ],
            k=1,
        )[0]

    if requisition_status == "Open":
        return random.choices(
            [
                "In Process",
                "Rejected",
                "Withdrawn",
            ],
            weights=[
                0.70,
                0.20,
                0.10,
            ],
            k=1,
        )[0]

    raise ValueError(
        f"Unexpected requisition status: "
        f"{requisition_status!r}."
    )


def choose_interview_score(
    application_status: str,
) -> float | None:
    """Generate an interview score when appropriate."""

    if application_status == "Hired":
        return round(
            random.uniform(
                75,
                100,
            ),
            1,
        )

    if application_status == "Offer Declined":
        return round(
            random.uniform(
                70,
                100,
            ),
            1,
        )

    if application_status == "Rejected":
        if random.random() < 0.65:
            return round(
                random.uniform(
                    35,
                    88,
                ),
                1,
            )

        return None

    if application_status == "In Process":
        if random.random() < 0.50:
            return round(
                random.uniform(
                    50,
                    95,
                ),
                1,
            )

        return None

    if application_status == "Withdrawn":
        if random.random() < 0.20:
            return round(
                random.uniform(
                    45,
                    95,
                ),
                1,
            )

        return None

    if (
        application_status
        == "Position Cancelled"
    ):
        if random.random() < 0.10:
            return round(
                random.uniform(
                    45,
                    95,
                ),
                1,
            )

        return None

    raise ValueError(
        f"Unexpected application status: "
        f"{application_status!r}."
    )


# ---------------------------------------------------------
# Requisition selection
# ---------------------------------------------------------

def build_requisition_pools(
    job_requisitions: pd.DataFrame,
) -> dict[str, list]:
    """Group requisition rows by status."""

    pools = {}

    for status in (
        REQUISITION_STATUS_CHOICES
    ):
        status_rows = (
            job_requisitions[
                job_requisitions[
                    "requisition_status"
                ].eq(status)
            ]
        )

        if status_rows.empty:
            raise ValueError(
                "job_requisitions: missing "
                f"{status} requisitions."
            )

        pools[status] = list(
            status_rows.itertuples(
                index=False
            )
        )

    return pools


def choose_unused_requisition(
    requisition_pools: dict[str, list],
    used_requisition_ids: set[int],
):
    """Choose a requisition not already used by a candidate."""

    for _ in range(200):
        selected_status = (
            random.choices(
                REQUISITION_STATUS_CHOICES,
                weights=(
                    REQUISITION_STATUS_WEIGHTS
                ),
                k=1,
            )[0]
        )

        requisition = random.choice(
            requisition_pools[
                selected_status
            ]
        )

        requisition_id = int(
            requisition.requisition_id
        )

        if (
            requisition_id
            not in used_requisition_ids
        ):
            return requisition

    raise ValueError(
        "Unable to choose an unused "
        "requisition for a candidate."
    )


# ---------------------------------------------------------
# General candidate applications
# ---------------------------------------------------------

def create_general_application_result(
    requisition,
) -> dict:
    """Create dates and results for a non-hired application."""

    requisition_status = str(
        requisition.requisition_status
    )

    requisition_open_date = (
        pd.Timestamp(
            requisition.open_date
        ).date()
    )

    if pd.isna(
        requisition.close_date
    ):
        requisition_end_date = (
            AS_OF_DATE
        )
    else:
        requisition_end_date = (
            pd.Timestamp(
                requisition.close_date
            ).date()
        )

    application_date = random_date(
        requisition_open_date,
        requisition_end_date,
    )

    application_status = (
        choose_general_outcome(
            requisition_status
        )
    )

    interview_score = (
        choose_interview_score(
            application_status
        )
    )

    offer_date = None
    decision_date = None

    if (
        application_status
        == "In Process"
    ):
        pass

    elif (
        application_status
        == "Position Cancelled"
    ):
        decision_date = (
            requisition_end_date
        )

    else:
        decision_date = random_date(
            application_date,
            requisition_end_date,
        )

    if (
        application_status
        == "Offer Declined"
    ):
        offer_date = random_date(
            application_date,
            decision_date,
        )

    return {
        "application_date": (
            application_date
        ),
        "application_status": (
            application_status
        ),
        "interview_score": (
            interview_score
        ),
        "offer_date": offer_date,
        "decision_date": decision_date,
    }


def add_general_applications(
    candidates: pd.DataFrame,
    employee_count: int,
    job_requisitions: pd.DataFrame,
    application_records: list[dict],
) -> None:
    """Create applications for candidates who were not hired."""

    general_candidates = (
        candidates
        .sort_values(
            "candidate_id"
        )
        .iloc[
            employee_count:
        ]
    )

    requisition_pools = (
        build_requisition_pools(
            job_requisitions
        )
    )

    for candidate in (
        general_candidates.itertuples(
            index=False
        )
    ):
        number_of_applications = (
            random.choices(
                GENERAL_APPLICATION_COUNTS,
                weights=(
                    GENERAL_APPLICATION_COUNT_WEIGHTS
                ),
                k=1,
            )[0]
        )

        used_requisition_ids = set()

        for _ in range(
            number_of_applications
        ):
            requisition = (
                choose_unused_requisition(
                    requisition_pools=(
                        requisition_pools
                    ),
                    used_requisition_ids=(
                        used_requisition_ids
                    ),
                )
            )

            requisition_id = int(
                requisition.requisition_id
            )

            result = (
                create_general_application_result(
                    requisition
                )
            )

            append_application(
                application_records=(
                    application_records
                ),
                candidate_id=int(
                    candidate.candidate_id
                ),
                requisition_id=(
                    requisition_id
                ),
                application_date=(
                    result[
                        "application_date"
                    ]
                ),
                application_status=(
                    result[
                        "application_status"
                    ]
                ),
                interview_score=(
                    result[
                        "interview_score"
                    ]
                ),
                offer_date=(
                    result[
                        "offer_date"
                    ]
                ),
                decision_date=(
                    result[
                        "decision_date"
                    ]
                ),
                employee_id=None,
            )

            used_requisition_ids.add(
                requisition_id
            )


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------

def create_applications(
    employees: pd.DataFrame,
    candidates: pd.DataFrame,
    job_requisitions: pd.DataFrame,
) -> pd.DataFrame:
    """Create the complete application table."""

    random.seed(
        RANDOM_SEED
    )

    application_records = []

    add_hired_applications(
        employees=employees,
        candidates=candidates,
        job_requisitions=(
            job_requisitions
        ),
        application_records=(
            application_records
        ),
    )

    add_general_applications(
        candidates=candidates,
        employee_count=len(
            employees
        ),
        job_requisitions=(
            job_requisitions
        ),
        application_records=(
            application_records
        ),
    )

    applications = pd.DataFrame(
        application_records
    )

    for date_column in [
        "application_date",
        "offer_date",
        "decision_date",
    ]:
        applications[
            date_column
        ] = pd.to_datetime(
            applications[
                date_column
            ]
        )

    applications = (
        applications
        .sort_values(
            [
                "application_date",
                "candidate_id",
                "requisition_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    applications.insert(
        0,
        "application_id",
        range(
            FIRST_APPLICATION_ID,
            FIRST_APPLICATION_ID
            + len(applications),
        ),
    )

    applications[
        "employee_id"
    ] = (
        applications[
            "employee_id"
        ]
        .astype("Int64")
    )

    return applications[
        [
            "application_id",
            "candidate_id",
            "requisition_id",
            "application_date",
            "application_status",
            "interview_score",
            "offer_date",
            "decision_date",
            "employee_id",
        ]
    ]


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_applications(
    applications: pd.DataFrame,
    employees: pd.DataFrame,
    candidates: pd.DataFrame,
    job_requisitions: pd.DataFrame,
) -> None:
    """Validate application-table business rules."""

    expected_columns = [
        "application_id",
        "candidate_id",
        "requisition_id",
        "application_date",
        "application_status",
        "interview_score",
        "offer_date",
        "decision_date",
        "employee_id",
    ]

    if (
        applications.columns.tolist()
        != expected_columns
    ):
        raise ValueError(
            "applications: unexpected columns "
            "or column order."
        )

    if applications.empty:
        raise ValueError(
            "applications: no records generated."
        )

    required_columns = [
        "application_id",
        "candidate_id",
        "requisition_id",
        "application_date",
        "application_status",
    ]

    if applications[
        required_columns
    ].isna().any().any():
        raise ValueError(
            "applications: missing "
            "required values."
        )

    if not applications[
        "application_id"
    ].is_unique:
        raise ValueError(
            "applications: application IDs "
            "are not unique."
        )

    expected_application_ids = list(
        range(
            FIRST_APPLICATION_ID,
            FIRST_APPLICATION_ID
            + len(applications),
        )
    )

    actual_application_ids = (
        applications[
            "application_id"
        ]
        .astype(int)
        .tolist()
    )

    if (
        actual_application_ids
        != expected_application_ids
    ):
        raise ValueError(
            "applications: application IDs "
            "are not sequential."
        )

    valid_candidate_ids = set(
        candidates[
            "candidate_id"
        ].astype(int)
    )

    valid_requisition_ids = set(
        job_requisitions[
            "requisition_id"
        ].astype(int)
    )

    valid_employee_ids = set(
        employees[
            "employee_id"
        ].astype(int)
    )

    if not set(
        applications[
            "candidate_id"
        ].astype(int)
    ).issubset(
        valid_candidate_ids
    ):
        raise ValueError(
            "applications: invalid "
            "candidate ID."
        )

    if not set(
        applications[
            "requisition_id"
        ].astype(int)
    ).issubset(
        valid_requisition_ids
    ):
        raise ValueError(
            "applications: invalid "
            "requisition ID."
        )

    used_employee_ids = set(
        applications[
            "employee_id"
        ]
        .dropna()
        .astype(int)
    )

    if not used_employee_ids.issubset(
        valid_employee_ids
    ):
        raise ValueError(
            "applications: invalid "
            "employee ID."
        )

    if set(
        applications[
            "candidate_id"
        ].astype(int)
    ) != valid_candidate_ids:
        raise ValueError(
            "applications: not every candidate "
            "has an application."
        )

    if applications.duplicated(
        subset=[
            "candidate_id",
            "requisition_id",
        ]
    ).any():
        raise ValueError(
            "applications: duplicate "
            "candidate-requisition pair."
        )

    if not set(
        applications[
            "application_status"
        ]
    ).issubset(
        ALLOWED_APPLICATION_STATUSES
    ):
        raise ValueError(
            "applications: invalid "
            "application status."
        )

    if set(
        applications[
            "application_status"
        ]
    ) != ALLOWED_APPLICATION_STATUSES:
        raise ValueError(
            "applications: not every "
            "application status appears."
        )

    # -----------------------------------------------------
    # Connect applications to requisitions
    # -----------------------------------------------------

    requisition_details = (
        job_requisitions[
            [
                "requisition_id",
                "job_role_id",
                "department_id",
                "location_id",
                "open_date",
                "close_date",
                "target_headcount",
                "requisition_status",
            ]
        ]
        .rename(
            columns={
                "job_role_id": (
                    "requisition_job_role_id"
                ),
                "department_id": (
                    "requisition_department_id"
                ),
                "location_id": (
                    "requisition_location_id"
                ),
                "open_date": (
                    "requisition_open_date"
                ),
                "close_date": (
                    "requisition_close_date"
                ),
            }
        )
    )

    application_details = (
        applications
        .merge(
            requisition_details,
            on="requisition_id",
            how="left",
        )
    )

    application_details[
        "requisition_end_date"
    ] = (
        application_details[
            "requisition_close_date"
        ]
        .fillna(
            pd.Timestamp(
                AS_OF_DATE
            )
        )
    )

    if application_details[
        "requisition_open_date"
    ].isna().any():
        raise ValueError(
            "applications: missing "
            "requisition details."
        )

    if application_details[
        "application_date"
    ].lt(
        application_details[
            "requisition_open_date"
        ]
    ).any():
        raise ValueError(
            "applications: application occurs "
            "before requisition opens."
        )

    if application_details[
        "application_date"
    ].gt(
        application_details[
            "requisition_end_date"
        ]
    ).any():
        raise ValueError(
            "applications: application occurs "
            "after requisition closes."
        )

    decided_applications = (
        application_details[
            application_details[
                "decision_date"
            ].notna()
        ]
    )

    if decided_applications[
        "decision_date"
    ].lt(
        decided_applications[
            "application_date"
        ]
    ).any():
        raise ValueError(
            "applications: decision occurs "
            "before application."
        )

    if decided_applications[
        "decision_date"
    ].gt(
        decided_applications[
            "requisition_end_date"
        ]
    ).any():
        raise ValueError(
            "applications: decision occurs "
            "after requisition ends."
        )

    offered_applications = (
        application_details[
            application_details[
                "offer_date"
            ].notna()
        ]
    )

    if offered_applications[
        "offer_date"
    ].lt(
        offered_applications[
            "application_date"
        ]
    ).any():
        raise ValueError(
            "applications: offer occurs "
            "before application."
        )

    if offered_applications[
        "offer_date"
    ].gt(
        offered_applications[
            "decision_date"
        ]
    ).any():
        raise ValueError(
            "applications: offer occurs "
            "after the decision."
        )

    # -----------------------------------------------------
    # Status rules
    # -----------------------------------------------------

    in_process = (
        application_details[
            "application_status"
        ].eq("In Process")
    )

    final_status = (
        ~in_process
    )

    if application_details.loc[
        in_process,
        "decision_date",
    ].notna().any():
        raise ValueError(
            "applications: an in-process "
            "application has a decision date."
        )

    if application_details.loc[
        final_status,
        "decision_date",
    ].isna().any():
        raise ValueError(
            "applications: a final application "
            "is missing a decision date."
        )

    if not application_details.loc[
        in_process,
        "requisition_status",
    ].eq("Open").all():
        raise ValueError(
            "applications: in-process applications "
            "must belong to open requisitions."
        )

    hired_mask = (
        application_details[
            "application_status"
        ].eq("Hired")
    )

    offer_declined_mask = (
        application_details[
            "application_status"
        ].eq("Offer Declined")
    )

    position_cancelled_mask = (
        application_details[
            "application_status"
        ].eq("Position Cancelled")
    )

    offer_status_mask = (
        hired_mask
        | offer_declined_mask
    )

    if application_details.loc[
        offer_status_mask,
        "offer_date",
    ].isna().any():
        raise ValueError(
            "applications: hired or declined "
            "offers must have offer dates."
        )

    if application_details.loc[
        ~offer_status_mask,
        "offer_date",
    ].notna().any():
        raise ValueError(
            "applications: an application "
            "without an offer has an offer date."
        )

    if not application_details.loc[
        hired_mask,
        "requisition_status",
    ].eq("Filled").all():
        raise ValueError(
            "applications: hired applications "
            "must use filled requisitions."
        )

    if not application_details.loc[
        position_cancelled_mask,
        "requisition_status",
    ].eq("Cancelled").all():
        raise ValueError(
            "applications: position-cancelled "
            "applications must use cancelled "
            "requisitions."
        )

    # -----------------------------------------------------
    # Interview-score rules
    # -----------------------------------------------------

    scored_applications = (
        applications[
            applications[
                "interview_score"
            ].notna()
        ]
    )

    if not scored_applications[
        "interview_score"
    ].between(
        0,
        100,
    ).all():
        raise ValueError(
            "applications: interview score "
            "is outside 0-100."
        )

    hired_applications = (
        applications[
            applications[
                "application_status"
            ].eq("Hired")
        ]
    )

    offer_declined_applications = (
        applications[
            applications[
                "application_status"
            ].eq("Offer Declined")
        ]
    )

    if hired_applications[
        "interview_score"
    ].isna().any():
        raise ValueError(
            "applications: a hired application "
            "is missing an interview score."
        )

    if not hired_applications[
        "interview_score"
    ].ge(75).all():
        raise ValueError(
            "applications: hired interview "
            "score is below 75."
        )

    if offer_declined_applications[
        "interview_score"
    ].isna().any():
        raise ValueError(
            "applications: an offer-declined "
            "application is missing a score."
        )

    if not offer_declined_applications[
        "interview_score"
    ].ge(70).all():
        raise ValueError(
            "applications: offer-declined "
            "score is below 70."
        )

    # -----------------------------------------------------
    # Employee-mapping rules
    # -----------------------------------------------------

    non_hired_applications = (
        applications[
            applications[
                "application_status"
            ].ne("Hired")
        ]
    )

    if non_hired_applications[
        "employee_id"
    ].notna().any():
        raise ValueError(
            "applications: a non-hired "
            "application has an employee ID."
        )

    if hired_applications[
        "employee_id"
    ].isna().any():
        raise ValueError(
            "applications: a hired application "
            "is missing an employee ID."
        )

    if (
        len(hired_applications)
        != len(employees)
    ):
        raise ValueError(
            "applications: hired application "
            "count does not match employees."
        )

    if not hired_applications[
        "employee_id"
    ].is_unique:
        raise ValueError(
            "applications: an employee appears "
            "in multiple hired applications."
        )

    if set(
        hired_applications[
            "employee_id"
        ].astype(int)
    ) != valid_employee_ids:
        raise ValueError(
            "applications: hired employee IDs "
            "do not cover every employee."
        )

    employee_order = (
        employees
        .sort_values(
            "employee_id"
        )
        .reset_index(
            drop=True
        )
    )

    reserved_candidates = (
        candidates
        .sort_values(
            "candidate_id"
        )
        .head(
            len(employees)
        )
        .reset_index(
            drop=True
        )
    )

    expected_hire_pairs = set(
        zip(
            reserved_candidates[
                "candidate_id"
            ].astype(int),
            employee_order[
                "employee_id"
            ].astype(int),
        )
    )

    actual_hire_pairs = set(
        hired_applications[
            [
                "candidate_id",
                "employee_id",
            ]
        ]
        .astype(int)
        .itertuples(
            index=False,
            name=None,
        )
    )

    if (
        actual_hire_pairs
        != expected_hire_pairs
    ):
        raise ValueError(
            "applications: candidate-to-employee "
            "mapping is incorrect."
        )

    # -----------------------------------------------------
    # Hired application and employee alignment
    # -----------------------------------------------------

    hired_details = (
        application_details[
            application_details[
                "application_status"
            ].eq("Hired")
        ]
        .merge(
            employees[
                [
                    "employee_id",
                    "hire_date",
                    "job_role_id",
                    "department_id",
                    "location_id",
                ]
            ],
            on="employee_id",
            how="left",
        )
    )

    if not hired_details[
        "decision_date"
    ].eq(
        hired_details[
            "hire_date"
        ]
    ).all():
        raise ValueError(
            "applications: hired decision date "
            "does not match employee hire date."
        )

    if not hired_details[
        "requisition_job_role_id"
    ].astype(int).eq(
        hired_details[
            "job_role_id"
        ].astype(int)
    ).all():
        raise ValueError(
            "applications: hired job role "
            "does not match employee."
        )

    if not hired_details[
        "requisition_department_id"
    ].astype(int).eq(
        hired_details[
            "department_id"
        ].astype(int)
    ).all():
        raise ValueError(
            "applications: hired department "
            "does not match employee."
        )

    if not hired_details[
        "requisition_location_id"
    ].astype(int).eq(
        hired_details[
            "location_id"
        ].astype(int)
    ).all():
        raise ValueError(
            "applications: hired location "
            "does not match employee."
        )

    # -----------------------------------------------------
    # Filled-requisition headcount alignment
    # -----------------------------------------------------

    hired_counts = (
        hired_applications
        .groupby(
            "requisition_id"
        )
        .size()
        .rename(
            "actual_hires"
        )
        .reset_index()
    )

    filled_requisitions = (
        job_requisitions[
            job_requisitions[
                "requisition_status"
            ].eq("Filled")
        ][
            [
                "requisition_id",
                "target_headcount",
            ]
        ]
    )

    filled_alignment = (
        filled_requisitions
        .merge(
            hired_counts,
            on="requisition_id",
            how="left",
        )
    )

    filled_alignment[
        "actual_hires"
    ] = (
        filled_alignment[
            "actual_hires"
        ]
        .fillna(0)
        .astype(int)
    )

    if not filled_alignment[
        "actual_hires"
    ].eq(
        filled_alignment[
            "target_headcount"
        ]
    ).all():
        raise ValueError(
            "applications: hired counts do not "
            "match filled target headcounts."
        )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_applications(
    applications: pd.DataFrame,
) -> None:
    """Save applications.csv."""

    output_path = (
        RAW_DATA_DIR
        / "applications.csv"
    )

    applications.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved applications.csv: "
        f"{len(applications)} rows and "
        f"{len(applications.columns)} columns"
    )


def print_summary(
    applications: pd.DataFrame,
) -> None:
    """Print useful application statistics."""

    print(
        "\nCandidates with applications:"
    )

    print(
        applications[
            "candidate_id"
        ].nunique()
    )

    print(
        "\nApplications by status:"
    )

    print(
        applications[
            "application_status"
        ].value_counts()
    )

    print(
        "\nHired applications:"
    )

    print(
        applications[
            "application_status"
        ].eq("Hired").sum()
    )

    print(
        "\nEmployees connected to applications:"
    )

    print(
        applications[
            "employee_id"
        ].nunique()
    )

    applications_per_candidate = (
        applications
        .groupby(
            "candidate_id"
        )
        .size()
    )

    print(
        "\nAverage applications per candidate:"
    )

    print(
        round(
            applications_per_candidate.mean(),
            2,
        )
    )

    print(
        "\nMaximum applications for one candidate:"
    )

    print(
        int(
            applications_per_candidate.max()
        )
    )


def main() -> None:
    """Generate and validate applications."""

    employees = load_table(
        "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    candidates = load_table(
        "candidates.csv"
    )

    job_requisitions = load_table(
        "job_requisitions.csv",
        parse_dates=[
            "open_date",
            "close_date",
        ],
    )

    applications = (
        create_applications(
            employees=employees,
            candidates=candidates,
            job_requisitions=(
                job_requisitions
            ),
        )
    )

    validate_applications(
        applications=applications,
        employees=employees,
        candidates=candidates,
        job_requisitions=(
            job_requisitions
        ),
    )

    save_applications(
        applications
    )

    print_summary(
        applications
    )

    print(
        "\nApplications generated "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()