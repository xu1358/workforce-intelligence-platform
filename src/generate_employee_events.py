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

FIRST_EVENT_ID = 1_100_001


ALLOWED_EVENT_TYPES = {
    "Hire",
    "Promotion",
    "Transfer",
    "Manager Change",
    "Leave",
    "Termination",
}


MANAGER_LEVELS = {
    "Department Head",
    "Senior Manager",
    "Team Manager",
}


# These rules identify which manager levels may supervise
# each employee level.
VALID_MANAGER_LEVELS_BY_EMPLOYEE_LEVEL = {
    "Senior Manager": {
        "Department Head",
    },
    "Team Manager": {
        "Department Head",
        "Senior Manager",
    },
    "Individual Contributor": {
        "Department Head",
        "Team Manager",
    },
}


# Probabilities for optional historical events.
TRANSFER_PROBABILITY = 0.07
MANAGER_CHANGE_PROBABILITY = 0.08
LEAVE_PROBABILITY = 0.06


LEAVE_DURATIONS = [
    30,
    45,
    60,
    84,
    90,
]


LEAVE_TYPES = [
    "Parental Leave",
    "Medical Leave",
    "Family Leave",
    "Personal Leave",
]


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
# Date helpers
# ---------------------------------------------------------

def random_date(
    start_date: date,
    end_date: date,
) -> date:
    """Generate a random date inside a date range."""

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


def get_employment_end_date(
    employee,
) -> date:
    """Return termination date or the analysis date."""

    if pd.isna(
        employee.termination_date
    ):
        return AS_OF_DATE

    return pd.Timestamp(
        employee.termination_date
    ).date()


def choose_available_date(
    start_date: date,
    end_date: date,
    occupied_dates: set[date],
) -> date | None:
    """Choose a date that is not already used."""

    if start_date > end_date:
        return None

    for _ in range(200):
        candidate_date = random_date(
            start_date,
            end_date,
        )

        if (
            candidate_date
            not in occupied_dates
        ):
            return candidate_date

    return None


# ---------------------------------------------------------
# Event-record helper
# ---------------------------------------------------------

def append_event(
    event_records: list[dict],
    employee_id: int,
    event_date: date,
    event_type: str,
    old_value,
    new_value,
    notes: str,
) -> None:
    """Add one event to the event-record list."""

    event_records.append(
        {
            "employee_id": int(
                employee_id
            ),
            "event_date": event_date,
            "event_type": event_type,
            "old_value": (
                None
                if old_value is None
                else str(old_value)
            ),
            "new_value": (
                None
                if new_value is None
                else str(new_value)
            ),
            "notes": notes,
        }
    )


# ---------------------------------------------------------
# Hire events
# ---------------------------------------------------------

def add_hire_events(
    employees: pd.DataFrame,
    event_records: list[dict],
    occupied_dates: dict[int, set[date]],
) -> None:
    """Create one hire event for every employee."""

    for employee in employees.itertuples(
        index=False
    ):
        employee_id = int(
            employee.employee_id
        )

        hire_date = pd.Timestamp(
            employee.hire_date
        ).date()

        occupied_dates[
            employee_id
        ].add(
            hire_date
        )

        append_event(
            event_records=event_records,
            employee_id=employee_id,
            event_date=hire_date,
            event_type="Hire",
            old_value=None,
            new_value="Active",
            notes=(
                "Hired into "
                f"department_id={int(employee.department_id)}, "
                f"location_id={int(employee.location_id)}, "
                f"job_role_id={int(employee.job_role_id)}."
            ),
        )


# ---------------------------------------------------------
# Promotion events
# ---------------------------------------------------------

def add_promotion_events(
    compensation_history: pd.DataFrame,
    event_records: list[dict],
    occupied_dates: dict[int, set[date]],
) -> None:
    """
    Create promotion events from compensation records whose
    change reason is Promotion.
    """

    compensation_history = (
        compensation_history
        .sort_values(
            [
                "employee_id",
                "effective_date",
            ]
        )
        .copy()
    )

    compensation_history[
        "previous_salary"
    ] = (
        compensation_history
        .groupby("employee_id")[
            "base_salary"
        ]
        .shift(1)
    )

    promotion_records = (
        compensation_history[
            compensation_history[
                "change_reason"
            ]
            == "Promotion"
        ]
    )

    for promotion in (
        promotion_records.itertuples(
            index=False
        )
    ):
        if pd.isna(
            promotion.previous_salary
        ):
            raise ValueError(
                "A promotion record is missing "
                "its previous salary."
            )

        employee_id = int(
            promotion.employee_id
        )

        event_date = pd.Timestamp(
            promotion.effective_date
        ).date()

        append_event(
            event_records=event_records,
            employee_id=employee_id,
            event_date=event_date,
            event_type="Promotion",
            old_value=int(
                promotion.previous_salary
            ),
            new_value=int(
                promotion.base_salary
            ),
            notes=(
                "Promotion-related compensation change; "
                "old_value and new_value represent "
                "annual base salary."
            ),
        )

        occupied_dates[
            employee_id
        ].add(
            event_date
        )


# ---------------------------------------------------------
# Transfer events
# ---------------------------------------------------------

def add_transfer_events(
    employees: pd.DataFrame,
    locations: pd.DataFrame,
    event_records: list[dict],
    occupied_dates: dict[int, set[date]],
) -> None:
    """Create a location transfer for some employees."""

    valid_location_ids = sorted(
        locations[
            "location_id"
        ]
        .astype(int)
        .tolist()
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

        employment_end_date = (
            get_employment_end_date(
                employee
            )
        )

        employment_days = (
            employment_end_date
            - hire_date
        ).days

        if employment_days < 365:
            continue

        if (
            random.random()
            >= TRANSFER_PROBABILITY
        ):
            continue

        current_location_id = int(
            employee.location_id
        )

        possible_old_locations = [
            location_id
            for location_id
            in valid_location_ids
            if location_id
            != current_location_id
        ]

        if not possible_old_locations:
            continue

        earliest_event_date = (
            hire_date
            + timedelta(days=180)
        )

        latest_event_date = (
            employment_end_date
            - timedelta(days=30)
        )

        event_date = (
            choose_available_date(
                start_date=(
                    earliest_event_date
                ),
                end_date=(
                    latest_event_date
                ),
                occupied_dates=(
                    occupied_dates[
                        employee_id
                    ]
                ),
            )
        )

        if event_date is None:
            continue

        old_location_id = random.choice(
            possible_old_locations
        )

        append_event(
            event_records=event_records,
            employee_id=employee_id,
            event_date=event_date,
            event_type="Transfer",
            old_value=old_location_id,
            new_value=(
                current_location_id
            ),
            notes=(
                "Internal location transfer; "
                "old_value and new_value represent "
                "location_id."
            ),
        )

        occupied_dates[
            employee_id
        ].add(
            event_date
        )


# ---------------------------------------------------------
# Manager-change events
# ---------------------------------------------------------

def add_manager_change_events(
    employees: pd.DataFrame,
    event_records: list[dict],
    occupied_dates: dict[int, set[date]],
) -> None:
    """Create a historical manager change for some employees."""

    employee_lookup = (
        employees
        .set_index("employee_id")
    )

    active_managers = employees[
        employees[
            "organizational_level"
        ].isin(
            MANAGER_LEVELS
        )
        & employees[
            "employment_status"
        ].eq("Active")
    ].copy()

    for employee in employees.itertuples(
        index=False
    ):
        if pd.isna(
            employee.manager_id
        ):
            continue

        employee_level = str(
            employee.organizational_level
        )

        if (
            employee_level
            not in
            VALID_MANAGER_LEVELS_BY_EMPLOYEE_LEVEL
        ):
            continue

        employee_id = int(
            employee.employee_id
        )

        current_manager_id = int(
            employee.manager_id
        )

        hire_date = pd.Timestamp(
            employee.hire_date
        ).date()

        employment_end_date = (
            get_employment_end_date(
                employee
            )
        )

        employment_days = (
            employment_end_date
            - hire_date
        ).days

        if employment_days < 365:
            continue

        if (
            random.random()
            >= MANAGER_CHANGE_PROBABILITY
        ):
            continue

        allowed_manager_levels = (
            VALID_MANAGER_LEVELS_BY_EMPLOYEE_LEVEL[
                employee_level
            ]
        )

        possible_old_managers = (
            active_managers[
                active_managers[
                    "department_id"
                ].eq(
                    int(
                        employee.department_id
                    )
                )
                & active_managers[
                    "organizational_level"
                ].isin(
                    allowed_manager_levels
                )
                & active_managers[
                    "employee_id"
                ].ne(
                    current_manager_id
                )
                & active_managers[
                    "employee_id"
                ].ne(
                    employee_id
                )
            ]
        )

        if possible_old_managers.empty:
            continue

        possible_old_manager_ids = (
            possible_old_managers[
                "employee_id"
            ]
            .astype(int)
            .tolist()
        )

        old_manager_id = random.choice(
            possible_old_manager_ids
        )

        current_manager_hire_date = (
            pd.Timestamp(
                employee_lookup.loc[
                    current_manager_id,
                    "hire_date",
                ]
            ).date()
        )

        old_manager_hire_date = (
            pd.Timestamp(
                employee_lookup.loc[
                    old_manager_id,
                    "hire_date",
                ]
            ).date()
        )

        earliest_event_date = max(
            hire_date
            + timedelta(days=180),
            current_manager_hire_date,
            old_manager_hire_date,
        )

        latest_event_date = (
            employment_end_date
            - timedelta(days=30)
        )

        event_date = (
            choose_available_date(
                start_date=(
                    earliest_event_date
                ),
                end_date=(
                    latest_event_date
                ),
                occupied_dates=(
                    occupied_dates[
                        employee_id
                    ]
                ),
            )
        )

        if event_date is None:
            continue

        append_event(
            event_records=event_records,
            employee_id=employee_id,
            event_date=event_date,
            event_type="Manager Change",
            old_value=old_manager_id,
            new_value=(
                current_manager_id
            ),
            notes=(
                "Manager reassignment; old_value "
                "and new_value represent manager_id."
            ),
        )

        occupied_dates[
            employee_id
        ].add(
            event_date
        )


# ---------------------------------------------------------
# Leave events
# ---------------------------------------------------------

def add_leave_events(
    employees: pd.DataFrame,
    event_records: list[dict],
    occupied_dates: dict[int, set[date]],
) -> None:
    """
    Create leave start and return events.

    Each selected employee receives two Leave rows:
    one for leaving and one for returning.
    """

    for employee in employees.itertuples(
        index=False
    ):
        employee_id = int(
            employee.employee_id
        )

        hire_date = pd.Timestamp(
            employee.hire_date
        ).date()

        employment_end_date = (
            get_employment_end_date(
                employee
            )
        )

        employment_days = (
            employment_end_date
            - hire_date
        ).days

        if employment_days < 365:
            continue

        if (
            random.random()
            >= LEAVE_PROBABILITY
        ):
            continue

        leave_duration = random.choice(
            LEAVE_DURATIONS
        )

        earliest_start_date = (
            hire_date
            + timedelta(days=180)
        )

        latest_start_date = (
            employment_end_date
            - timedelta(
                days=(
                    leave_duration
                    + 30
                )
            )
        )

        if (
            earliest_start_date
            > latest_start_date
        ):
            continue

        leave_start_date = None
        leave_return_date = None

        for _ in range(200):
            possible_start_date = (
                random_date(
                    earliest_start_date,
                    latest_start_date,
                )
            )

            possible_return_date = (
                possible_start_date
                + timedelta(
                    days=leave_duration
                )
            )

            employee_occupied_dates = (
                occupied_dates[
                    employee_id
                ]
            )

            if (
                possible_start_date
                not in employee_occupied_dates
                and possible_return_date
                not in employee_occupied_dates
            ):
                leave_start_date = (
                    possible_start_date
                )

                leave_return_date = (
                    possible_return_date
                )

                break

        if leave_start_date is None:
            continue

        leave_type = random.choice(
            LEAVE_TYPES
        )

        leave_notes = (
            f"{leave_type}; planned duration "
            f"{leave_duration} days."
        )

        append_event(
            event_records=event_records,
            employee_id=employee_id,
            event_date=leave_start_date,
            event_type="Leave",
            old_value="Active",
            new_value="Leave",
            notes=leave_notes,
        )

        append_event(
            event_records=event_records,
            employee_id=employee_id,
            event_date=leave_return_date,
            event_type="Leave",
            old_value="Leave",
            new_value="Active",
            notes=leave_notes,
        )

        occupied_dates[
            employee_id
        ].add(
            leave_start_date
        )

        occupied_dates[
            employee_id
        ].add(
            leave_return_date
        )


# ---------------------------------------------------------
# Termination events
# ---------------------------------------------------------

def add_termination_events(
    employees: pd.DataFrame,
    event_records: list[dict],
) -> None:
    """Create one termination event per terminated employee."""

    for employee in employees.itertuples(
        index=False
    ):
        if pd.isna(
            employee.termination_date
        ):
            continue

        employee_id = int(
            employee.employee_id
        )

        termination_date = pd.Timestamp(
            employee.termination_date
        ).date()

        termination_type = str(
            employee.termination_type
        )

        append_event(
            event_records=event_records,
            employee_id=employee_id,
            event_date=termination_date,
            event_type="Termination",
            old_value="Active",
            new_value="Terminated",
            notes=(
                f"{termination_type} termination."
            ),
        )


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------

def create_employee_events(
    employees: pd.DataFrame,
    compensation_history: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    """Create the complete employee-event table."""

    random.seed(RANDOM_SEED)

    employees = employees.sort_values(
        "employee_id"
    ).copy()

    employees["manager_id"] = (
        employees["manager_id"]
        .astype("Int64")
    )

    event_records = []

    occupied_dates = {
        int(employee_id): set()
        for employee_id
        in employees[
            "employee_id"
        ]
    }

    # Reserve termination dates before generating
    # optional events.
    for employee in employees.itertuples(
        index=False
    ):
        if not pd.isna(
            employee.termination_date
        ):
            occupied_dates[
                int(employee.employee_id)
            ].add(
                pd.Timestamp(
                    employee.termination_date
                ).date()
            )

    add_hire_events(
        employees=employees,
        event_records=event_records,
        occupied_dates=occupied_dates,
    )

    add_promotion_events(
        compensation_history=(
            compensation_history
        ),
        event_records=event_records,
        occupied_dates=occupied_dates,
    )

    add_transfer_events(
        employees=employees,
        locations=locations,
        event_records=event_records,
        occupied_dates=occupied_dates,
    )

    add_manager_change_events(
        employees=employees,
        event_records=event_records,
        occupied_dates=occupied_dates,
    )

    add_leave_events(
        employees=employees,
        event_records=event_records,
        occupied_dates=occupied_dates,
    )

    add_termination_events(
        employees=employees,
        event_records=event_records,
    )

    employee_events = pd.DataFrame(
        event_records
    )

    event_type_order = {
        "Hire": 1,
        "Promotion": 2,
        "Transfer": 3,
        "Manager Change": 4,
        "Leave": 5,
        "Termination": 6,
    }

    employee_events[
        "_event_type_order"
    ] = (
        employee_events[
            "event_type"
        ].map(
            event_type_order
        )
    )

    employee_events = (
        employee_events
        .sort_values(
            [
                "employee_id",
                "event_date",
                "_event_type_order",
            ]
        )
        .drop(
            columns=[
                "_event_type_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    employee_events.insert(
        0,
        "event_id",
        range(
            FIRST_EVENT_ID,
            FIRST_EVENT_ID
            + len(employee_events),
        ),
    )

    employee_events[
        "event_date"
    ] = pd.to_datetime(
        employee_events[
            "event_date"
        ]
    )

    return employee_events


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_employee_events(
    employee_events: pd.DataFrame,
    employees: pd.DataFrame,
    compensation_history: pd.DataFrame,
    locations: pd.DataFrame,
) -> None:
    """Validate employee-event business rules."""

    expected_columns = {
        "event_id",
        "employee_id",
        "event_date",
        "event_type",
        "old_value",
        "new_value",
        "notes",
    }

    if (
        set(employee_events.columns)
        != expected_columns
    ):
        raise ValueError(
            "employee_events: unexpected columns."
        )

    if employee_events.empty:
        raise ValueError(
            "employee_events: no records generated."
        )

    if employee_events[
        "event_id"
    ].isna().any():
        raise ValueError(
            "employee_events: missing event IDs."
        )

    if not employee_events[
        "event_id"
    ].is_unique:
        raise ValueError(
            "employee_events: event IDs "
            "are not unique."
        )

    if not set(
        employee_events[
            "event_type"
        ]
    ).issubset(
        ALLOWED_EVENT_TYPES
    ):
        raise ValueError(
            "employee_events: invalid event type."
        )

    valid_employee_ids = set(
        employees[
            "employee_id"
        ].astype(int)
    )

    used_employee_ids = set(
        employee_events[
            "employee_id"
        ].astype(int)
    )

    if not used_employee_ids.issubset(
        valid_employee_ids
    ):
        raise ValueError(
            "employee_events: invalid employee ID."
        )

    if employee_events.duplicated(
        subset=[
            "employee_id",
            "event_date",
            "event_type",
        ]
    ).any():
        raise ValueError(
            "employee_events: duplicate "
            "employee-date-type event."
        )

    non_hire_events = (
        employee_events[
            employee_events[
                "event_type"
            ]
            != "Hire"
        ]
    )

    if non_hire_events[
        "old_value"
    ].isna().any():
        raise ValueError(
            "employee_events: a non-hire event "
            "is missing old_value."
        )

    if employee_events[
        "new_value"
    ].isna().any():
        raise ValueError(
            "employee_events: an event is "
            "missing new_value."
        )

    employee_details = (
        employees[
            [
                "employee_id",
                "hire_date",
                "termination_date",
                "termination_type",
                "employment_status",
                "department_id",
                "location_id",
                "manager_id",
                "organizational_level",
            ]
        ]
        .copy()
    )

    employee_details[
        "employment_end_date"
    ] = (
        employee_details[
            "termination_date"
        ]
        .fillna(
            pd.Timestamp(
                AS_OF_DATE
            )
        )
    )

    detailed_events = (
        employee_events
        .merge(
            employee_details,
            on="employee_id",
            how="left",
        )
    )

    if detailed_events[
        "hire_date"
    ].isna().any():
        raise ValueError(
            "employee_events: missing "
            "employee details."
        )

    if (
        detailed_events[
            "event_date"
        ]
        < detailed_events[
            "hire_date"
        ]
    ).any():
        raise ValueError(
            "employee_events: an event occurs "
            "before hire."
        )

    if (
        detailed_events[
            "event_date"
        ]
        > detailed_events[
            "employment_end_date"
        ]
    ).any():
        raise ValueError(
            "employee_events: an event occurs "
            "after employment ends."
        )

    # -----------------------------------------------------
    # Hire-event checks
    # -----------------------------------------------------

    hire_events = employee_events[
        employee_events[
            "event_type"
        ]
        == "Hire"
    ]

    hire_counts = (
        hire_events
        .groupby("employee_id")
        .size()
        .reindex(
            employees[
                "employee_id"
            ],
            fill_value=0,
        )
    )

    if not hire_counts.eq(1).all():
        raise ValueError(
            "employee_events: every employee must "
            "have exactly one hire event."
        )

    hire_details = (
        hire_events
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

    if not hire_details[
        "event_date"
    ].eq(
        hire_details[
            "hire_date"
        ]
    ).all():
        raise ValueError(
            "employee_events: a hire-event date "
            "does not match the hire date."
        )

    if hire_events[
        "old_value"
    ].notna().any():
        raise ValueError(
            "employee_events: hire events "
            "should have blank old_value."
        )

    if not hire_events[
        "new_value"
    ].eq("Active").all():
        raise ValueError(
            "employee_events: hire-event "
            "new_value must be Active."
        )

    # -----------------------------------------------------
    # Termination-event checks
    # -----------------------------------------------------

    termination_events = (
        employee_events[
            employee_events[
                "event_type"
            ]
            == "Termination"
        ]
    )

    termination_counts = (
        termination_events
        .groupby("employee_id")
        .size()
        .reindex(
            employees[
                "employee_id"
            ],
            fill_value=0,
        )
    )

    terminated_employee_ids = (
        employees.loc[
            employees[
                "employment_status"
            ]
            == "Terminated",
            "employee_id",
        ]
        .astype(int)
        .tolist()
    )

    active_employee_ids = (
        employees.loc[
            employees[
                "employment_status"
            ]
            == "Active",
            "employee_id",
        ]
        .astype(int)
        .tolist()
    )

    if not termination_counts.loc[
        terminated_employee_ids
    ].eq(1).all():
        raise ValueError(
            "employee_events: a terminated employee "
            "is missing a termination event."
        )

    if not termination_counts.loc[
        active_employee_ids
    ].eq(0).all():
        raise ValueError(
            "employee_events: an active employee "
            "has a termination event."
        )

    termination_details = (
        termination_events
        .merge(
            employees[
                [
                    "employee_id",
                    "termination_date",
                ]
            ],
            on="employee_id",
            how="left",
        )
    )

    if not termination_details[
        "event_date"
    ].eq(
        termination_details[
            "termination_date"
        ]
    ).all():
        raise ValueError(
            "employee_events: termination-event "
            "date does not match termination date."
        )

    if not termination_events[
        "old_value"
    ].eq("Active").all():
        raise ValueError(
            "employee_events: termination "
            "old_value must be Active."
        )

    if not termination_events[
        "new_value"
    ].eq("Terminated").all():
        raise ValueError(
            "employee_events: termination "
            "new_value must be Terminated."
        )

    # -----------------------------------------------------
    # Promotion checks
    # -----------------------------------------------------

    expected_promotions = (
        compensation_history.loc[
            compensation_history[
                "change_reason"
            ]
            == "Promotion",
            [
                "employee_id",
                "effective_date",
            ],
        ]
        .copy()
    )

    expected_promotions[
        "effective_date"
    ] = pd.to_datetime(
        expected_promotions[
            "effective_date"
        ]
    )

    expected_promotion_pairs = set(
        expected_promotions[
            [
                "employee_id",
                "effective_date",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    promotion_events = employee_events[
        employee_events[
            "event_type"
        ]
        == "Promotion"
    ]

    actual_promotion_pairs = set(
        promotion_events[
            [
                "employee_id",
                "event_date",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if (
        actual_promotion_pairs
        != expected_promotion_pairs
    ):
        raise ValueError(
            "employee_events: promotion events "
            "do not match compensation history."
        )

    if not promotion_events.empty:
        promotion_old_salary = (
            promotion_events[
                "old_value"
            ].astype(int)
        )

        promotion_new_salary = (
            promotion_events[
                "new_value"
            ].astype(int)
        )

        if not promotion_new_salary.ge(
            promotion_old_salary
        ).all():
            raise ValueError(
                "employee_events: promotion salary "
                "is below previous salary."
            )

    # -----------------------------------------------------
    # Transfer checks
    # -----------------------------------------------------

    transfer_events = (
        employee_events[
            employee_events[
                "event_type"
            ]
            == "Transfer"
        ]
        .merge(
            employees[
                [
                    "employee_id",
                    "location_id",
                ]
            ],
            on="employee_id",
            how="left",
        )
    )

    if not transfer_events.empty:
        old_location_ids = (
            transfer_events[
                "old_value"
            ].astype(int)
        )

        new_location_ids = (
            transfer_events[
                "new_value"
            ].astype(int)
        )

        valid_location_ids = set(
            locations[
                "location_id"
            ].astype(int)
        )

        if not set(
            old_location_ids
        ).issubset(
            valid_location_ids
        ):
            raise ValueError(
                "employee_events: a transfer has "
                "an invalid old location."
            )

        if not set(
            new_location_ids
        ).issubset(
            valid_location_ids
        ):
            raise ValueError(
                "employee_events: a transfer has "
                "an invalid new location."
            )

        if old_location_ids.eq(
            new_location_ids
        ).any():
            raise ValueError(
                "employee_events: transfer locations "
                "must be different."
            )

        if not new_location_ids.eq(
            transfer_events[
                "location_id"
            ].astype(int)
        ).all():
            raise ValueError(
                "employee_events: transfer new location "
                "does not match current employee data."
            )

    # -----------------------------------------------------
    # Manager-change checks
    # -----------------------------------------------------

    manager_change_events = (
        employee_events[
            employee_events[
                "event_type"
            ]
            == "Manager Change"
        ]
        .copy()
    )

    if not manager_change_events.empty:
        old_manager_ids = (
            manager_change_events[
                "old_value"
            ].astype(int)
        )

        new_manager_ids = (
            manager_change_events[
                "new_value"
            ].astype(int)
        )

        if not set(
            old_manager_ids
        ).issubset(
            valid_employee_ids
        ):
            raise ValueError(
                "employee_events: invalid "
                "old manager ID."
            )

        if not set(
            new_manager_ids
        ).issubset(
            valid_employee_ids
        ):
            raise ValueError(
                "employee_events: invalid "
                "new manager ID."
            )

        if old_manager_ids.eq(
            new_manager_ids
        ).any():
            raise ValueError(
                "employee_events: old and new "
                "manager IDs are equal."
            )

        manager_change_details = (
            manager_change_events
            .merge(
                employees[
                    [
                        "employee_id",
                        "manager_id",
                        "department_id",
                        "organizational_level",
                    ]
                ],
                on="employee_id",
                how="left",
            )
        )

        manager_change_details[
            "old_manager_id"
        ] = old_manager_ids.values

        manager_change_details[
            "new_manager_id"
        ] = new_manager_ids.values

        manager_lookup = (
            employees
            .set_index("employee_id")
        )

        manager_change_details[
            "old_manager_department"
        ] = (
            manager_change_details[
                "old_manager_id"
            ].map(
                manager_lookup[
                    "department_id"
                ]
            )
        )

        manager_change_details[
            "new_manager_department"
        ] = (
            manager_change_details[
                "new_manager_id"
            ].map(
                manager_lookup[
                    "department_id"
                ]
            )
        )

        manager_change_details[
            "old_manager_status"
        ] = (
            manager_change_details[
                "old_manager_id"
            ].map(
                manager_lookup[
                    "employment_status"
                ]
            )
        )

        manager_change_details[
            "new_manager_status"
        ] = (
            manager_change_details[
                "new_manager_id"
            ].map(
                manager_lookup[
                    "employment_status"
                ]
            )
        )

        manager_change_details[
            "old_manager_level"
        ] = (
            manager_change_details[
                "old_manager_id"
            ].map(
                manager_lookup[
                    "organizational_level"
                ]
            )
        )

        manager_change_details[
            "new_manager_level"
        ] = (
            manager_change_details[
                "new_manager_id"
            ].map(
                manager_lookup[
                    "organizational_level"
                ]
            )
        )

        if not manager_change_details[
            "new_manager_id"
        ].eq(
            manager_change_details[
                "manager_id"
            ].astype(int)
        ).all():
            raise ValueError(
                "employee_events: new manager does "
                "not match current employee data."
            )

        if not manager_change_details[
            "old_manager_department"
        ].eq(
            manager_change_details[
                "department_id"
            ]
        ).all():
            raise ValueError(
                "employee_events: employee and old "
                "manager departments differ."
            )

        if not manager_change_details[
            "new_manager_department"
        ].eq(
            manager_change_details[
                "department_id"
            ]
        ).all():
            raise ValueError(
                "employee_events: employee and new "
                "manager departments differ."
            )

        if not manager_change_details[
            "old_manager_status"
        ].eq("Active").all():
            raise ValueError(
                "employee_events: an old manager "
                "is inactive."
            )

        if not manager_change_details[
            "new_manager_status"
        ].eq("Active").all():
            raise ValueError(
                "employee_events: a new manager "
                "is inactive."
            )

        for row in (
            manager_change_details.itertuples(
                index=False
            )
        ):
            valid_manager_levels = (
                VALID_MANAGER_LEVELS_BY_EMPLOYEE_LEVEL[
                    row.organizational_level
                ]
            )

            if (
                row.old_manager_level
                not in valid_manager_levels
                or row.new_manager_level
                not in valid_manager_levels
            ):
                raise ValueError(
                    "employee_events: manager level "
                    "is invalid for the employee."
                )

    # -----------------------------------------------------
    # Leave-event checks
    # -----------------------------------------------------

    leave_events = employee_events[
        employee_events[
            "event_type"
        ]
        == "Leave"
    ]

    if not leave_events.empty:
        leave_counts = (
            leave_events
            .groupby("employee_id")
            .size()
        )

        if not leave_counts.eq(2).all():
            raise ValueError(
                "employee_events: every leave period "
                "must contain two events."
            )

        leave_start_events = (
            leave_events[
                leave_events[
                    "old_value"
                ].eq("Active")
                & leave_events[
                    "new_value"
                ].eq("Leave")
            ]
        )

        leave_return_events = (
            leave_events[
                leave_events[
                    "old_value"
                ].eq("Leave")
                & leave_events[
                    "new_value"
                ].eq("Active")
            ]
        )

        leave_employee_ids = set(
            leave_counts.index
        )

        if set(
            leave_start_events[
                "employee_id"
            ]
        ) != leave_employee_ids:
            raise ValueError(
                "employee_events: a leave period "
                "is missing its start event."
            )

        if set(
            leave_return_events[
                "employee_id"
            ]
        ) != leave_employee_ids:
            raise ValueError(
                "employee_events: a leave period "
                "is missing its return event."
            )

        leave_periods = (
            leave_start_events[
                [
                    "employee_id",
                    "event_date",
                ]
            ]
            .merge(
                leave_return_events[
                    [
                        "employee_id",
                        "event_date",
                    ]
                ],
                on="employee_id",
                suffixes=(
                    "_start",
                    "_return",
                ),
            )
        )

        if not leave_periods[
            "event_date_start"
        ].lt(
            leave_periods[
                "event_date_return"
            ]
        ).all():
            raise ValueError(
                "employee_events: a leave return "
                "does not follow its start."
            )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_employee_events(
    employee_events: pd.DataFrame,
) -> None:
    """Save employee events as a CSV."""

    output_path = (
        RAW_DATA_DIR
        / "employee_events.csv"
    )

    employee_events.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved employee_events.csv: "
        f"{len(employee_events)} rows and "
        f"{len(employee_events.columns)} columns"
    )


def print_summary(
    employee_events: pd.DataFrame,
) -> None:
    """Print a basic event summary."""

    print(
        "\nEmployees with events:"
    )

    print(
        employee_events[
            "employee_id"
        ].nunique()
    )

    print(
        "\nEvents by type:"
    )

    print(
        employee_events[
            "event_type"
        ].value_counts()
    )

    events_per_employee = (
        employee_events
        .groupby("employee_id")
        .size()
    )

    print(
        "\nAverage events per employee:"
    )

    print(
        round(
            events_per_employee.mean(),
            2,
        )
    )

    print(
        "\nMaximum events for one employee:"
    )

    print(
        int(
            events_per_employee.max()
        )
    )


def main() -> None:
    """Generate and validate employee events."""

    employees = load_table(
        "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    compensation_history = load_table(
        "compensation_history.csv",
        parse_dates=[
            "effective_date",
        ],
    )

    locations = load_table(
        "locations.csv"
    )

    employee_events = (
        create_employee_events(
            employees=employees,
            compensation_history=(
                compensation_history
            ),
            locations=locations,
        )
    )

    validate_employee_events(
        employee_events=employee_events,
        employees=employees,
        compensation_history=(
            compensation_history
        ),
        locations=locations,
    )

    save_employee_events(
        employee_events
    )

    print_summary(
        employee_events
    )

    print(
        "\nEmployee events generated "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()