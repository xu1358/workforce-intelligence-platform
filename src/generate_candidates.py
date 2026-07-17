from pathlib import Path
import random

import pandas as pd


# ---------------------------------------------------------
# Project settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RANDOM_SEED = 42

FIRST_CANDIDATE_ID = 200_001
TOTAL_CANDIDATES = 40_000


APPLICATION_SOURCES = [
    "LinkedIn",
    "Employee Referral",
    "Company Careers Page",
    "Indeed",
    "University Recruiting",
    "Staffing Agency",
    "Professional Association",
    "Job Fair",
]


FUTURE_HIRE_SOURCE_WEIGHTS = [
    0.24,
    0.23,
    0.20,
    0.12,
    0.08,
    0.05,
    0.05,
    0.03,
]


GENERAL_SOURCE_WEIGHTS = [
    0.29,
    0.10,
    0.21,
    0.18,
    0.07,
    0.07,
    0.04,
    0.04,
]


EXTERNAL_CANDIDATE_LOCATIONS = [
    "Atlanta, GA",
    "Austin, TX",
    "Boston, MA",
    "Charlotte, NC",
    "Chicago, IL",
    "Columbus, OH",
    "Dallas, TX",
    "Denver, CO",
    "Detroit, MI",
    "Houston, TX",
    "Los Angeles, CA",
    "Miami, FL",
    "Minneapolis, MN",
    "Nashville, TN",
    "New York, NY",
    "Phoenix, AZ",
    "Pittsburgh, PA",
    "Portland, OR",
    "Raleigh, NC",
    "San Diego, CA",
    "San Francisco, CA",
    "Seattle, WA",
    "St. Louis, MO",
    "Washington, DC",
]


EXPERIENCE_RANGES_BY_LEVEL = {
    "Department Head": (
        12,
        30,
    ),
    "Senior Manager": (
        8,
        24,
    ),
    "Team Manager": (
        4,
        18,
    ),
    "Individual Contributor": (
        0,
        15,
    ),
}


# ---------------------------------------------------------
# File-loading helper
# ---------------------------------------------------------

def load_table(
    filename: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load one CSV file from the raw-data directory."""

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
# Location helpers
# ---------------------------------------------------------

def build_company_location_lookup(
    locations: pd.DataFrame,
) -> tuple[
    dict[int, str],
    list[str],
]:
    """Create formatted company-location values."""

    required_columns = {
        "location_id",
        "city",
        "state",
    }

    if not required_columns.issubset(
        locations.columns
    ):
        raise ValueError(
            "locations: missing location_id, "
            "city, or state."
        )

    location_copy = locations.copy()

    location_copy[
        "candidate_location"
    ] = (
        location_copy[
            "city"
        ]
        .astype(str)
        .str.strip()
        + ", "
        + location_copy[
            "state"
        ]
        .astype(str)
        .str.strip()
    )

    location_lookup = dict(
        zip(
            location_copy[
                "location_id"
            ].astype(int),
            location_copy[
                "candidate_location"
            ],
        )
    )

    company_locations = sorted(
        location_copy[
            "candidate_location"
        ]
        .unique()
        .tolist()
    )

    return (
        location_lookup,
        company_locations,
    )


# ---------------------------------------------------------
# Candidate value helpers
# ---------------------------------------------------------

def choose_application_source(
    future_hire: bool,
) -> str:
    """Choose the candidate's application source."""

    if future_hire:
        weights = (
            FUTURE_HIRE_SOURCE_WEIGHTS
        )
    else:
        weights = (
            GENERAL_SOURCE_WEIGHTS
        )

    return random.choices(
        APPLICATION_SOURCES,
        weights=weights,
        k=1,
    )[0]


def choose_future_hire_experience(
    organizational_level: str,
) -> int:
    """Generate experience appropriate for an employee level."""

    if (
        organizational_level
        not in EXPERIENCE_RANGES_BY_LEVEL
    ):
        raise ValueError(
            "employees: unexpected "
            "organizational level "
            f"{organizational_level!r}."
        )

    minimum, maximum = (
        EXPERIENCE_RANGES_BY_LEVEL[
            organizational_level
        ]
    )

    return random.randint(
        minimum,
        maximum,
    )


def choose_general_experience(
    education_level: str,
) -> int:
    """Generate experience for a general candidate."""

    education_text = (
        education_level.lower()
    )

    if "doctor" in education_text:
        minimum, maximum = (
            2,
            25,
        )

    elif "master" in education_text:
        minimum, maximum = (
            1,
            22,
        )

    elif "bachelor" in education_text:
        minimum, maximum = (
            0,
            20,
        )

    elif "associate" in education_text:
        minimum, maximum = (
            0,
            17,
        )

    else:
        minimum, maximum = (
            0,
            15,
        )

    return random.randint(
        minimum,
        maximum,
    )


def choose_future_hire_location(
    employee_location: str,
    all_candidate_locations: list[str],
) -> str:
    """
    Most future hires already live near their eventual
    work location. Some candidates relocate.
    """

    if random.random() < 0.80:
        return employee_location

    alternative_locations = [
        location
        for location
        in all_candidate_locations
        if location
        != employee_location
    ]

    return random.choice(
        alternative_locations
    )


def choose_general_location(
    company_locations: list[str],
    all_candidate_locations: list[str],
) -> str:
    """Choose a location for a general candidate."""

    if random.random() < 0.45:
        return random.choice(
            company_locations
        )

    return random.choice(
        all_candidate_locations
    )


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------

def create_candidates(
    employees: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    """Create candidate profiles for the recruiting system."""

    random.seed(
        RANDOM_SEED
    )

    if (
        len(employees)
        > TOTAL_CANDIDATES
    ):
        raise ValueError(
            "TOTAL_CANDIDATES must be at "
            "least the employee count."
        )

    required_employee_columns = {
        "employee_id",
        "education_level",
        "organizational_level",
        "location_id",
    }

    if not required_employee_columns.issubset(
        employees.columns
    ):
        raise ValueError(
            "employees: missing columns "
            "required for candidates."
        )

    if employees[
        "education_level"
    ].isna().any():
        raise ValueError(
            "employees: missing "
            "education levels."
        )

    (
        location_lookup,
        company_locations,
    ) = build_company_location_lookup(
        locations
    )

    all_candidate_locations = sorted(
        set(
            company_locations
        )
        | set(
            EXTERNAL_CANDIDATE_LOCATIONS
        )
    )

    employees = (
        employees
        .sort_values(
            "employee_id"
        )
        .reset_index(
            drop=True
        )
    )

    candidate_records = []

    next_candidate_id = (
        FIRST_CANDIDATE_ID
    )

    # -----------------------------------------------------
    # First candidate block:
    # candidates who will later be connected to employees
    # through accepted applications.
    # -----------------------------------------------------

    for employee in employees.itertuples(
        index=False
    ):
        employee_location_id = int(
            employee.location_id
        )

        if (
            employee_location_id
            not in location_lookup
        ):
            raise ValueError(
                "employees: invalid "
                f"location_id "
                f"{employee_location_id}."
            )

        employee_location = (
            location_lookup[
                employee_location_id
            ]
        )

        candidate_records.append(
            {
                "candidate_id": (
                    next_candidate_id
                ),
                "application_source": (
                    choose_application_source(
                        future_hire=True
                    )
                ),
                "education_level": str(
                    employee.education_level
                ),
                "years_experience": (
                    choose_future_hire_experience(
                        str(
                            employee
                            .organizational_level
                        )
                    )
                ),
                "candidate_location": (
                    choose_future_hire_location(
                        employee_location,
                        all_candidate_locations,
                    )
                ),
            }
        )

        next_candidate_id += 1

    # -----------------------------------------------------
    # Remaining candidate block:
    # candidates who will not necessarily be hired.
    # -----------------------------------------------------

    education_distribution = (
        employees[
            "education_level"
        ]
        .astype(str)
        .value_counts(
            normalize=True
        )
    )

    education_levels = (
        education_distribution
        .index
        .tolist()
    )

    education_weights = (
        education_distribution
        .values
        .tolist()
    )

    remaining_candidate_count = (
        TOTAL_CANDIDATES
        - len(employees)
    )

    for _ in range(
        remaining_candidate_count
    ):
        education_level = (
            random.choices(
                education_levels,
                weights=(
                    education_weights
                ),
                k=1,
            )[0]
        )

        candidate_records.append(
            {
                "candidate_id": (
                    next_candidate_id
                ),
                "application_source": (
                    choose_application_source(
                        future_hire=False
                    )
                ),
                "education_level": (
                    education_level
                ),
                "years_experience": (
                    choose_general_experience(
                        education_level
                    )
                ),
                "candidate_location": (
                    choose_general_location(
                        company_locations,
                        all_candidate_locations,
                    )
                ),
            }
        )

        next_candidate_id += 1

    candidates = pd.DataFrame(
        candidate_records
    )

    return candidates


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_candidates(
    candidates: pd.DataFrame,
    employees: pd.DataFrame,
    locations: pd.DataFrame,
) -> None:
    """Validate candidate-table business rules."""

    expected_columns = [
        "candidate_id",
        "application_source",
        "education_level",
        "years_experience",
        "candidate_location",
    ]

    if (
        candidates.columns.tolist()
        != expected_columns
    ):
        raise ValueError(
            "candidates: unexpected "
            "columns or column order."
        )

    if (
        len(candidates)
        != TOTAL_CANDIDATES
    ):
        raise ValueError(
            "candidates: unexpected row count."
        )

    if candidates.isna().any().any():
        raise ValueError(
            "candidates: missing values found."
        )

    if not candidates[
        "candidate_id"
    ].is_unique:
        raise ValueError(
            "candidates: candidate IDs "
            "are not unique."
        )

    expected_candidate_ids = list(
        range(
            FIRST_CANDIDATE_ID,
            FIRST_CANDIDATE_ID
            + TOTAL_CANDIDATES,
        )
    )

    actual_candidate_ids = (
        candidates
        .sort_values(
            "candidate_id"
        )[
            "candidate_id"
        ]
        .astype(int)
        .tolist()
    )

    if (
        actual_candidate_ids
        != expected_candidate_ids
    ):
        raise ValueError(
            "candidates: candidate IDs "
            "are not sequential."
        )

    if not set(
        candidates[
            "application_source"
        ]
    ).issubset(
        set(
            APPLICATION_SOURCES
        )
    ):
        raise ValueError(
            "candidates: invalid "
            "application source."
        )

    valid_education_levels = set(
        employees[
            "education_level"
        ].astype(str)
    )

    if not set(
        candidates[
            "education_level"
        ].astype(str)
    ).issubset(
        valid_education_levels
    ):
        raise ValueError(
            "candidates: invalid "
            "education level."
        )

    if not candidates[
        "years_experience"
    ].between(
        0,
        30,
    ).all():
        raise ValueError(
            "candidates: years_experience "
            "is outside 0-30."
        )

    experience_is_integer = (
        candidates[
            "years_experience"
        ]
        .astype(float)
        .mod(1)
        .eq(0)
        .all()
    )

    if not experience_is_integer:
        raise ValueError(
            "candidates: years_experience "
            "must use whole numbers."
        )

    (
        _,
        company_locations,
    ) = build_company_location_lookup(
        locations
    )

    allowed_locations = (
        set(
            company_locations
        )
        | set(
            EXTERNAL_CANDIDATE_LOCATIONS
        )
    )

    if not set(
        candidates[
            "candidate_location"
        ]
    ).issubset(
        allowed_locations
    ):
        raise ValueError(
            "candidates: invalid "
            "candidate location."
        )

    valid_location_format = (
        candidates[
            "candidate_location"
        ]
        .str.contains(
            r",\s*\S+",
            regex=True,
        )
        .all()
    )

    if not valid_location_format:
        raise ValueError(
            "candidates: candidate locations "
            "must use 'City, State' format."
        )

    # The first candidate block aligns with employees
    # so later accepted applications can link them.
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

    employee_order = (
        employees
        .sort_values(
            "employee_id"
        )
        .reset_index(
            drop=True
        )
    )

    education_matches = (
        reserved_candidates[
            "education_level"
        ]
        .astype(str)
        .eq(
            employee_order[
                "education_level"
            ].astype(str)
        )
        .all()
    )

    if not education_matches:
        raise ValueError(
            "candidates: reserved "
            "hire-candidate education does "
            "not match employee data."
        )


# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------

def save_candidates(
    candidates: pd.DataFrame,
) -> None:
    """Save candidates.csv."""

    output_path = (
        RAW_DATA_DIR
        / "candidates.csv"
    )

    candidates.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved candidates.csv: "
        f"{len(candidates)} rows and "
        f"{len(candidates.columns)} columns"
    )


def print_summary(
    candidates: pd.DataFrame,
    employee_count: int,
) -> None:
    """Print useful candidate statistics."""

    print(
        "\nCandidate ID range:"
    )

    print(
        f"{candidates['candidate_id'].min()} "
        f"to "
        f"{candidates['candidate_id'].max()}"
    )

    print(
        "\nReserved future-hire candidates:"
    )

    print(
        employee_count
    )

    print(
        "\nCandidates by application source:"
    )

    print(
        candidates[
            "application_source"
        ].value_counts()
    )

    print(
        "\nCandidates by education level:"
    )

    print(
        candidates[
            "education_level"
        ].value_counts()
    )

    print(
        "\nYears of experience:"
    )

    print(
        candidates[
            "years_experience"
        ]
        .describe()
        .round(2)
    )

    print(
        "\nTop candidate locations:"
    )

    print(
        candidates[
            "candidate_location"
        ]
        .value_counts()
        .head(10)
    )


def main() -> None:
    """Generate and validate candidate profiles."""

    employees = load_table(
        "employees.csv"
    )

    locations = load_table(
        "locations.csv"
    )

    candidates = create_candidates(
        employees=employees,
        locations=locations,
    )

    validate_candidates(
        candidates=candidates,
        employees=employees,
        locations=locations,
    )

    save_candidates(
        candidates
    )

    print_summary(
        candidates,
        employee_count=len(employees),
    )

    print(
        "\nCandidates generated and "
        "validated successfully."
    )


if __name__ == "__main__":
    main()