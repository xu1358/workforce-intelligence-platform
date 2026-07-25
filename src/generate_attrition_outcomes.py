"""Apply Version 2 attrition outcomes to potential workforce histories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from attrition_hazard import load_hazard_config, simulate_attrition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "config" / "attrition_hazard_config.yaml"

FIRST_EVENT_ID = 1_100_001

VALID_MANAGER_LEVELS = {
    "Senior Manager": ["Department Head"],
    "Team Manager": ["Senior Manager", "Department Head"],
    "Individual Contributor": ["Team Manager", "Department Head"],
}


def load_table(
    filename: str,
    date_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one required raw CSV table."""

    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

    frame = pd.read_csv(path)

    for date_column in date_columns or []:
        frame[date_column] = pd.to_datetime(frame[date_column])

    return frame


def _employment_end_lookup(
    employees: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """Map each employee to termination or administrative censor date."""

    return (
        employees.set_index("employee_id")["termination_date"]
        .fillna(as_of_date)
    )


def apply_outcomes(
    employees: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Write the simulated status and exit fields to employees."""

    revised = employees.copy()
    outcome_lookup = outcomes.set_index("employee_id")

    for column in [
        "employment_status",
        "termination_date",
        "termination_type",
    ]:
        revised[column] = revised["employee_id"].map(
            outcome_lookup[column]
        )

    revised["termination_date"] = pd.to_datetime(
        revised["termination_date"]
    )
    revised["manager_id"] = revised["manager_id"].astype("Int64")

    return revised


def censor_dated_table(
    frame: pd.DataFrame,
    date_column: str,
    end_dates: pd.Series,
) -> pd.DataFrame:
    """Remove records that occur after employment ends."""

    revised = frame.copy()
    revised["_employment_end_date"] = revised["employee_id"].map(end_dates)

    revised = revised.loc[
        revised[date_column] <= revised["_employment_end_date"]
    ].drop(columns="_employment_end_date")

    return revised.reset_index(drop=True)


def censor_training_records(
    training_records: pd.DataFrame,
    training_programs: pd.DataFrame,
    end_dates: pd.Series,
) -> pd.DataFrame:
    """Censor training starts and unfinished completions at exit."""

    revised = training_records.copy()
    revised["_employment_end_date"] = revised["employee_id"].map(end_dates)

    revised = revised.loc[
        revised["start_date"] <= revised["_employment_end_date"]
    ].copy()

    completion_after_exit = (
        revised["completion_date"].notna()
        & (
            revised["completion_date"]
            > revised["_employment_end_date"]
        )
    )

    if completion_after_exit.any():
        required_hours = training_programs.set_index("program_id")[
            "required_hours"
        ]

        revised.loc[
            completion_after_exit,
            "completion_status",
        ] = "Incomplete"
        revised.loc[
            completion_after_exit,
            "completion_date",
        ] = pd.NaT
        revised.loc[
            completion_after_exit,
            "score",
        ] = pd.NA

        maximum_incomplete_hours = (
            revised.loc[completion_after_exit, "program_id"]
            .map(required_hours)
            .astype(float)
            .sub(0.1)
            .clip(lower=0.0)
        )

        revised.loc[
            completion_after_exit,
            "training_hours",
        ] = (
            revised.loc[
                completion_after_exit,
                "training_hours",
            ]
            .astype(float)
            .clip(upper=maximum_incomplete_hours)
        )

    return (
        revised.drop(columns="_employment_end_date")
        .reset_index(drop=True)
    )


def reconstruct_assignments_at_exit(
    employees: pd.DataFrame,
    employee_events: pd.DataFrame,
) -> pd.DataFrame:
    """Undo potential transfers or manager changes dated after exit."""

    revised = employees.copy()
    terminated = revised.loc[
        revised["employment_status"].eq("Terminated")
    ]

    for employee in terminated.itertuples(index=False):
        employee_id = int(employee.employee_id)
        termination_date = pd.Timestamp(employee.termination_date)
        employee_history = employee_events.loc[
            employee_events["employee_id"].eq(employee_id)
        ]

        transfers = employee_history.loc[
            employee_history["event_type"].eq("Transfer")
        ].sort_values("event_date")

        if not transfers.empty:
            location_id = int(float(transfers.iloc[0]["old_value"]))

            for transfer in transfers.itertuples(index=False):
                if pd.Timestamp(transfer.event_date) > termination_date:
                    break
                location_id = int(float(transfer.new_value))

            revised.loc[
                revised["employee_id"].eq(employee_id),
                "location_id",
            ] = location_id

        manager_changes = employee_history.loc[
            employee_history["event_type"].eq("Manager Change")
        ].sort_values("event_date")

        if not manager_changes.empty:
            manager_id = int(float(manager_changes.iloc[0]["old_value"]))

            for manager_change in manager_changes.itertuples(index=False):
                if pd.Timestamp(manager_change.event_date) > termination_date:
                    break
                manager_id = int(float(manager_change.new_value))

            revised.loc[
                revised["employee_id"].eq(employee_id),
                "manager_id",
            ] = manager_id

    revised["manager_id"] = revised["manager_id"].astype("Int64")
    return revised


def repair_potential_manager_changes(
    employees: pd.DataFrame,
    employee_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove manager changes that became impossible after simulation."""

    revised_employees = employees.copy()
    revised_events = employee_events.copy()
    manager_changes = revised_events.loc[
        revised_events["event_type"].eq("Manager Change")
    ].sort_values(["employee_id", "event_date"])

    if manager_changes.empty:
        return revised_employees, revised_events

    employee_lookup = revised_employees.set_index("employee_id")
    manager_end_dates = employee_lookup["termination_date"]

    old_manager_end = pd.to_numeric(
        manager_changes["old_value"]
    ).astype(int).map(manager_end_dates)
    new_manager_end = pd.to_numeric(
        manager_changes["new_value"]
    ).astype(int).map(manager_end_dates)

    old_manager_available = (
        old_manager_end.isna()
        | old_manager_end.gt(manager_changes["event_date"])
    )
    new_manager_available = (
        new_manager_end.isna()
        | new_manager_end.gt(manager_changes["event_date"])
    )
    valid_changes = manager_changes.loc[
        old_manager_available & new_manager_available
    ]

    invalid_change_indices = manager_changes.index.difference(
        valid_changes.index
    )
    revised_events = revised_events.drop(index=invalid_change_indices)

    for employee_id, original_history in manager_changes.groupby(
        "employee_id"
    ):
        baseline_manager_id = int(
            float(original_history.iloc[0]["old_value"])
        )
        valid_history = valid_changes.loc[
            valid_changes["employee_id"].eq(employee_id)
        ]

        if valid_history.empty:
            final_manager_id = baseline_manager_id
        else:
            final_manager_id = int(
                float(valid_history.iloc[-1]["new_value"])
            )

        revised_employees.loc[
            revised_employees["employee_id"].eq(int(employee_id)),
            "manager_id",
        ] = final_manager_id

    revised_employees["manager_id"] = revised_employees[
        "manager_id"
    ].astype("Int64")

    return revised_employees, revised_events


def censor_employee_events(
    employee_events: pd.DataFrame,
    employees: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Censor potential events and add sampled termination events."""

    end_dates = _employment_end_lookup(employees, as_of_date)
    revised = employee_events.loc[
        ~employee_events["event_type"].eq("Termination")
    ].copy()
    revised["_employment_end_date"] = revised["employee_id"].map(end_dates)

    revised = revised.loc[
        revised["event_date"] <= revised["_employment_end_date"]
    ].drop(columns="_employment_end_date")

    # A leave is represented by a start and a return.  If attrition
    # censors either half, remove the partial leave pair.
    leave_counts = (
        revised.loc[revised["event_type"].eq("Leave")]
        .groupby("employee_id")
        .size()
    )
    incomplete_leave_ids = set(
        leave_counts.loc[~leave_counts.eq(2)].index.astype(int)
    )

    if incomplete_leave_ids:
        revised = revised.loc[
            ~(
                revised["event_type"].eq("Leave")
                & revised["employee_id"].isin(incomplete_leave_ids)
            )
        ]

    termination_records = []

    for employee in employees.loc[
        employees["employment_status"].eq("Terminated")
    ].itertuples(index=False):
        termination_records.append(
            {
                "employee_id": int(employee.employee_id),
                "event_date": pd.Timestamp(employee.termination_date),
                "event_type": "Termination",
                "old_value": "Active",
                "new_value": "Terminated",
                "notes": f"{employee.termination_type} termination.",
            }
        )

    if termination_records:
        revised = pd.concat(
            [revised, pd.DataFrame(termination_records)],
            ignore_index=True,
        )

    return revised


def _manager_capacity(
    organizational_level: str,
    hierarchy_config: dict[str, Any],
) -> int:
    """Return the configured direct-report capacity for a manager."""

    if organizational_level == "Department Head":
        return int(hierarchy_config["maximum_department_head_reports"])

    return int(hierarchy_config["maximum_other_manager_reports"])


def reassign_reports_after_manager_exit(
    employees: pd.DataFrame,
    employee_events: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reassign employees whose manager exits before they do."""

    revised_employees = employees.copy()
    revised_events = employee_events.copy()
    hierarchy_config = config["hierarchy"]

    if not hierarchy_config["reassign_reports_after_manager_exit"]:
        return revised_employees, revised_events

    employee_lookup = revised_employees.set_index("employee_id")
    as_of_date = pd.Timestamp(config["simulation"]["as_of_date"])

    active_managers = revised_employees.loc[
        revised_employees["employment_status"].eq("Active")
        & revised_employees["organizational_level"].isin(
            ["Department Head", "Senior Manager", "Team Manager"]
        )
    ].copy()

    active_employee_ids = set(
        revised_employees.loc[
            revised_employees["employment_status"].eq("Active"),
            "employee_id",
        ].astype(int)
    )
    current_loads = (
        revised_employees.loc[
            revised_employees["employee_id"].isin(active_employee_ids),
            "manager_id",
        ]
        .dropna()
        .astype(int)
        .value_counts()
        .to_dict()
    )

    manager_end_dates = employee_lookup["termination_date"].to_dict()
    manager_statuses = employee_lookup["employment_status"].to_dict()
    reassignment_records: list[dict[str, Any]] = []

    for employee in revised_employees.sort_values(
        ["department_id", "organizational_level", "employee_id"]
    ).itertuples(index=False):
        if pd.isna(employee.manager_id):
            continue

        old_manager_id = int(employee.manager_id)

        if manager_statuses[old_manager_id] == "Active":
            continue

        manager_exit_date = pd.Timestamp(manager_end_dates[old_manager_id])
        employee_end_date = (
            pd.Timestamp(employee.termination_date)
            if not pd.isna(employee.termination_date)
            else as_of_date
        )

        if employee_end_date <= manager_exit_date:
            continue

        valid_levels = VALID_MANAGER_LEVELS[str(employee.organizational_level)]
        candidates = active_managers.loc[
            active_managers["department_id"].eq(int(employee.department_id))
            & active_managers["organizational_level"].isin(valid_levels)
            & active_managers["employee_id"].ne(int(employee.employee_id))
        ].copy()

        candidates["_current_load"] = (
            candidates["employee_id"]
            .astype(int)
            .map(current_loads)
            .fillna(0)
            .astype(int)
        )
        candidates["_capacity"] = candidates["organizational_level"].map(
            lambda level: _manager_capacity(level, hierarchy_config)
        )
        candidates = candidates.loc[
            candidates["_current_load"] < candidates["_capacity"]
        ]

        if candidates.empty:
            raise ValueError(
                "No active manager with capacity is available for "
                f"employee {int(employee.employee_id)}."
            )

        level_priority = {
            level: position
            for position, level in enumerate(valid_levels)
        }
        candidates["_level_priority"] = candidates[
            "organizational_level"
        ].map(level_priority)

        replacement = candidates.sort_values(
            ["_level_priority", "_current_load", "employee_id"]
        ).iloc[0]
        new_manager_id = int(replacement["employee_id"])

        revised_employees.loc[
            revised_employees["employee_id"].eq(int(employee.employee_id)),
            "manager_id",
        ] = new_manager_id

        if int(employee.employee_id) in active_employee_ids:
            current_loads[old_manager_id] = max(
                0,
                current_loads.get(old_manager_id, 0) - 1,
            )
            current_loads[new_manager_id] = (
                current_loads.get(new_manager_id, 0) + 1
            )

        duplicate_mask = (
            revised_events["employee_id"].eq(int(employee.employee_id))
            & revised_events["event_type"].eq("Manager Change")
            & revised_events["event_date"].eq(manager_exit_date)
        )
        revised_events = revised_events.loc[~duplicate_mask]

        reassignment_records.append(
            {
                "employee_id": int(employee.employee_id),
                "event_date": manager_exit_date,
                "event_type": "Manager Change",
                "old_value": str(old_manager_id),
                "new_value": str(new_manager_id),
                "notes": (
                    "Manager reassignment following a simulated "
                    "manager termination."
                ),
            }
        )

    if reassignment_records:
        revised_events = pd.concat(
            [revised_events, pd.DataFrame(reassignment_records)],
            ignore_index=True,
        )

    revised_employees["manager_id"] = revised_employees[
        "manager_id"
    ].astype("Int64")

    return revised_employees, revised_events


def finalize_event_ids(employee_events: pd.DataFrame) -> pd.DataFrame:
    """Sort the final event table and assign stable sequential IDs."""

    event_order = {
        "Hire": 1,
        "Promotion": 2,
        "Transfer": 3,
        "Manager Change": 4,
        "Leave": 5,
        "Termination": 6,
    }

    revised = employee_events.drop(
        columns=["event_id"],
        errors="ignore",
    ).copy()
    revised["_event_order"] = revised["event_type"].map(event_order)

    revised = (
        revised.sort_values(
            ["employee_id", "event_date", "_event_order"]
        )
        .drop(columns="_event_order")
        .reset_index(drop=True)
    )

    revised.insert(
        0,
        "event_id",
        range(FIRST_EVENT_ID, FIRST_EVENT_ID + len(revised)),
    )

    return revised


def validate_final_outputs(
    employees: pd.DataFrame,
    compensation_history: pd.DataFrame,
    performance_reviews: pd.DataFrame,
    training_records: pd.DataFrame,
    employee_events: pd.DataFrame,
    outcomes: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate the most important post-hazard temporal rules."""

    as_of_date = pd.Timestamp(config["simulation"]["as_of_date"])
    end_dates = _employment_end_lookup(employees, as_of_date)

    terminated = employees["employment_status"].eq("Terminated")
    active = employees["employment_status"].eq("Active")

    if employees.loc[terminated, "termination_date"].isna().any():
        raise ValueError("A terminated employee is missing an exit date.")
    if employees.loc[terminated, "termination_type"].isna().any():
        raise ValueError("A terminated employee is missing an exit type.")
    if employees.loc[active, "termination_date"].notna().any():
        raise ValueError("An active employee has an exit date.")

    dated_tables = [
        (compensation_history, "effective_date", "compensation"),
        (performance_reviews, "review_date", "performance"),
        (training_records, "start_date", "training"),
        (employee_events, "event_date", "events"),
    ]

    for frame, date_column, table_name in dated_tables:
        record_end_dates = frame["employee_id"].map(end_dates)
        if (frame[date_column] > record_end_dates).any():
            raise ValueError(
                f"{table_name}: a record occurs after employment ends."
            )

    completed_training = training_records["completion_date"].notna()
    if (
        training_records.loc[completed_training, "completion_date"]
        > training_records.loc[completed_training, "employee_id"].map(
            end_dates
        )
    ).any():
        raise ValueError("A training completion occurs after employment ends.")

    termination_counts = (
        employee_events.loc[
            employee_events["event_type"].eq("Termination")
        ]
        .groupby("employee_id")
        .size()
        .reindex(employees["employee_id"], fill_value=0)
    )
    terminated_ids = employees.loc[terminated, "employee_id"].astype(int)
    active_ids = employees.loc[active, "employee_id"].astype(int)

    if not termination_counts.loc[terminated_ids].eq(1).all():
        raise ValueError("A terminated employee lacks one termination event.")
    if not termination_counts.loc[active_ids].eq(0).all():
        raise ValueError("An active employee has a termination event.")

    employee_lookup = employees.set_index("employee_id")
    active_with_manager = employees.loc[
        active & employees["manager_id"].notna()
    ].copy()
    active_with_manager["manager_status"] = (
        active_with_manager["manager_id"]
        .astype(int)
        .map(employee_lookup["employment_status"])
    )
    active_with_manager["manager_department_id"] = (
        active_with_manager["manager_id"]
        .astype(int)
        .map(employee_lookup["department_id"])
    )
    active_with_manager["manager_level"] = (
        active_with_manager["manager_id"]
        .astype(int)
        .map(employee_lookup["organizational_level"])
    )

    if not active_with_manager["manager_status"].eq("Active").all():
        raise ValueError("An active employee reports to an inactive manager.")
    if not active_with_manager["manager_department_id"].eq(
        active_with_manager["department_id"]
    ).all():
        raise ValueError("An active employee and manager differ by department.")

    for employee_level, valid_manager_levels in VALID_MANAGER_LEVELS.items():
        level_relationships = active_with_manager.loc[
            active_with_manager["organizational_level"].eq(employee_level)
        ]
        if not level_relationships["manager_level"].isin(
            valid_manager_levels
        ).all():
            raise ValueError(
                f"An active {employee_level} has an invalid manager level."
            )

    active_loads = (
        active_with_manager["manager_id"]
        .astype(int)
        .value_counts()
        .rename("direct_reports")
        .to_frame()
    )
    active_loads["manager_level"] = active_loads.index.map(
        employee_lookup["organizational_level"]
    )
    active_loads["capacity"] = active_loads["manager_level"].map(
        lambda level: _manager_capacity(level, config["hierarchy"])
    )

    if (active_loads["direct_reports"] > active_loads["capacity"]).any():
        raise ValueError("An active manager exceeds the configured capacity.")

    manager_changes = employee_events.loc[
        employee_events["event_type"].eq("Manager Change")
    ].sort_values(["employee_id", "event_date"])

    if not manager_changes.empty:
        latest_manager_changes = manager_changes.groupby(
            "employee_id",
            as_index=False,
        ).last()
        latest_manager_changes["current_manager_id"] = (
            latest_manager_changes["employee_id"].map(
                employee_lookup["manager_id"]
            )
        )

        if not pd.to_numeric(
            latest_manager_changes["new_value"]
        ).astype(int).eq(
            latest_manager_changes["current_manager_id"].astype(int)
        ).all():
            raise ValueError(
                "A latest manager-change event does not match "
                "the employee's current manager."
            )

        for _, change_history in manager_changes.groupby("employee_id"):
            if len(change_history) < 2:
                continue
            previous_new = pd.to_numeric(
                change_history["new_value"]
            ).astype(int).iloc[:-1].reset_index(drop=True)
            following_old = pd.to_numeric(
                change_history["old_value"]
            ).astype(int).iloc[1:].reset_index(drop=True)
            if not previous_new.eq(following_old).all():
                raise ValueError(
                    "A manager-change history has a broken transition chain."
                )

    promotion_compensation_pairs = set(
        compensation_history.loc[
            compensation_history["change_reason"].eq("Promotion"),
            ["employee_id", "effective_date"],
        ].itertuples(index=False, name=None)
    )
    promotion_event_pairs = set(
        employee_events.loc[
            employee_events["event_type"].eq("Promotion"),
            ["employee_id", "event_date"],
        ].itertuples(index=False, name=None)
    )

    if promotion_compensation_pairs != promotion_event_pairs:
        raise ValueError(
            "Promotion events do not match promotion compensation records."
        )

    protected_levels = set(config["hierarchy"]["protected_levels"])
    if employees.loc[
        employees["organizational_level"].isin(protected_levels),
        "employment_status",
    ].ne("Active").any():
        raise ValueError("A protected organizational level was terminated.")

    manager_exits = employees.loc[
        employees["organizational_level"].isin(
            ["Senior Manager", "Team Manager"]
        )
        & terminated
    ]

    if manager_exits.empty:
        raise ValueError(
            "The hazard produced no leadership attrition outside "
            "the protected department-head level."
        )

    if not outcomes["employee_id"].is_unique:
        raise ValueError("Attrition outcomes contain duplicate employees.")
    if len(outcomes) != len(employees):
        raise ValueError("Attrition outcomes do not cover every employee.")

    probability_columns = [
        "average_voluntary_probability",
        "average_involuntary_probability",
    ]
    for column in probability_columns:
        if not diagnostics[column].between(0.0, 1.0).all():
            raise ValueError(f"Invalid monthly probability in {column}.")


def save_outputs(
    employees: pd.DataFrame,
    compensation_history: pd.DataFrame,
    performance_reviews: pd.DataFrame,
    training_records: pd.DataFrame,
    employee_events: pd.DataFrame,
    outcomes: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    """Save revised raw tables and transparent simulation artifacts."""

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_outputs = {
        "employees.csv": employees,
        "compensation_history.csv": compensation_history,
        "performance_reviews.csv": performance_reviews,
        "training_records.csv": training_records,
        "employee_events.csv": employee_events,
    }

    for filename, frame in raw_outputs.items():
        frame.to_csv(
            RAW_DATA_DIR / filename,
            index=False,
            date_format="%Y-%m-%d",
        )

    outcomes.to_csv(
        PROCESSED_DATA_DIR / "attrition_hazard_outcomes.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    diagnostics.to_csv(
        INTERIM_DATA_DIR / "attrition_hazard_monthly_diagnostics.csv",
        index=False,
        date_format="%Y-%m-%d",
    )


def print_summary(
    employees: pd.DataFrame,
    employee_events: pd.DataFrame,
) -> None:
    """Print a concise simulation summary."""

    terminated = employees.loc[
        employees["employment_status"].eq("Terminated")
    ]
    manager_exits = terminated.loc[
        terminated["organizational_level"].isin(
            ["Senior Manager", "Team Manager"]
        )
    ]
    manager_reassignments = employee_events.loc[
        employee_events["event_type"].eq("Manager Change")
        & employee_events["notes"].str.contains(
            "simulated manager termination",
            na=False,
        )
    ]

    print("\nATTRITION HAZARD SUMMARY")
    print(f"Employees simulated: {len(employees):,}")
    print(f"Terminations: {len(terminated):,}")
    print(f"Cumulative termination rate: {len(terminated) / len(employees):.2%}")
    print(
        "Voluntary share: "
        f"{terminated['termination_type'].eq('Voluntary').mean():.2%}"
    )
    print(f"Senior/team manager exits: {len(manager_exits):,}")
    print(f"Manager reassignments: {len(manager_reassignments):,}")


def main() -> None:
    """Run the hazard and materialize the final Version 2 histories."""

    config = load_hazard_config(CONFIG_PATH)
    as_of_date = pd.Timestamp(config["simulation"]["as_of_date"])

    employees = load_table("employees.csv", ["hire_date", "termination_date"])
    job_roles = load_table("job_roles.csv")
    compensation_history = load_table(
        "compensation_history.csv",
        ["effective_date"],
    )
    performance_reviews = load_table(
        "performance_reviews.csv",
        ["review_date"],
    )
    training_records = load_table(
        "training_records.csv",
        ["start_date", "completion_date"],
    )
    training_programs = load_table("training_programs.csv")
    employee_events = load_table("employee_events.csv", ["event_date"])

    outcomes, diagnostics = simulate_attrition(
        employees=employees,
        job_roles=job_roles,
        compensation_history=compensation_history,
        performance_reviews=performance_reviews,
        training_records=training_records,
        employee_events=employee_events,
        config=config,
    )

    employees = apply_outcomes(employees, outcomes)
    employees, employee_events = repair_potential_manager_changes(
        employees,
        employee_events,
    )
    employees = reconstruct_assignments_at_exit(
        employees,
        employee_events,
    )
    end_dates = _employment_end_lookup(employees, as_of_date)

    compensation_history = censor_dated_table(
        compensation_history,
        "effective_date",
        end_dates,
    )
    performance_reviews = censor_dated_table(
        performance_reviews,
        "review_date",
        end_dates,
    )
    training_records = censor_training_records(
        training_records,
        training_programs,
        end_dates,
    )
    employee_events = censor_employee_events(
        employee_events,
        employees,
        as_of_date,
    )

    employees, employee_events = reassign_reports_after_manager_exit(
        employees,
        employee_events,
        config,
    )
    employee_events = finalize_event_ids(employee_events)

    validate_final_outputs(
        employees,
        compensation_history,
        performance_reviews,
        training_records,
        employee_events,
        outcomes,
        diagnostics,
        config,
    )

    save_outputs(
        employees,
        compensation_history,
        performance_reviews,
        training_records,
        employee_events,
        outcomes,
        diagnostics,
    )
    print_summary(employees, employee_events)

    print("\nAttrition outcomes generated and validated successfully.")


if __name__ == "__main__":
    main()
