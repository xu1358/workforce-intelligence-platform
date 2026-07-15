from pathlib import Path

import pandas as pd


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define where the generated CSV files will be saved.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def create_departments() -> pd.DataFrame:
    """Create the department reference table."""

    departments = [
        {
            "department_id": 1,
            "department_name": "Engineering",
            "department_group": "Product and Technology",
            "cost_center": "ENG-100",
        },
        {
            "department_id": 2,
            "department_name": "Manufacturing",
            "department_group": "Operations",
            "cost_center": "MFG-200",
        },
        {
            "department_id": 3,
            "department_name": "Supply Chain",
            "department_group": "Operations",
            "cost_center": "SCM-300",
        },
        {
            "department_id": 4,
            "department_name": "Sales",
            "department_group": "Commercial",
            "cost_center": "SAL-400",
        },
        {
            "department_id": 5,
            "department_name": "Finance",
            "department_group": "Corporate",
            "cost_center": "FIN-500",
        },
        {
            "department_id": 6,
            "department_name": "Human Resources",
            "department_group": "Corporate",
            "cost_center": "HR-600",
        },
        {
            "department_id": 7,
            "department_name": "Information Technology",
            "department_group": "Product and Technology",
            "cost_center": "IT-700",
        },
        {
            "department_id": 8,
            "department_name": "Customer Support",
            "department_group": "Commercial",
            "cost_center": "SUP-800",
        },
    ]

    return pd.DataFrame(departments)


def create_locations() -> pd.DataFrame:
    """Create the location reference table."""

    locations = [
        {
            "location_id": 1,
            "city": "Austin",
            "state": "Texas",
            "region": "South",
            "location_type": "Office and Factory",
        },
        {
            "location_id": 2,
            "city": "Fremont",
            "state": "California",
            "region": "West",
            "location_type": "Factory",
        },
        {
            "location_id": 3,
            "city": "Reno",
            "state": "Nevada",
            "region": "West",
            "location_type": "Factory and Distribution Center",
        },
        {
            "location_id": 4,
            "city": "Buffalo",
            "state": "New York",
            "region": "Northeast",
            "location_type": "Factory",
        },
        {
            "location_id": 5,
            "city": "Phoenix",
            "state": "Arizona",
            "region": "Southwest",
            "location_type": "Office and Distribution Center",
        },
    ]

    return pd.DataFrame(locations)


def create_job_roles() -> pd.DataFrame:
    """Create the job-role reference table."""

    job_roles = [
        {
            "job_role_id": 1,
            "job_title": "Data Analyst",
            "job_family": "Analytics",
            "job_level": 2,
            "salary_band_min": 65000,
            "salary_band_max": 95000,
        },
        {
            "job_role_id": 2,
            "job_title": "Senior Data Analyst",
            "job_family": "Analytics",
            "job_level": 3,
            "salary_band_min": 90000,
            "salary_band_max": 130000,
        },
        {
            "job_role_id": 3,
            "job_title": "Data Scientist",
            "job_family": "Data Science",
            "job_level": 3,
            "salary_band_min": 105000,
            "salary_band_max": 155000,
        },
        {
            "job_role_id": 4,
            "job_title": "Software Engineer",
            "job_family": "Engineering",
            "job_level": 2,
            "salary_band_min": 90000,
            "salary_band_max": 135000,
        },
        {
            "job_role_id": 5,
            "job_title": "Senior Software Engineer",
            "job_family": "Engineering",
            "job_level": 4,
            "salary_band_min": 130000,
            "salary_band_max": 190000,
        },
        {
            "job_role_id": 6,
            "job_title": "Manufacturing Engineer",
            "job_family": "Engineering",
            "job_level": 3,
            "salary_band_min": 85000,
            "salary_band_max": 125000,
        },
        {
            "job_role_id": 7,
            "job_title": "Production Technician",
            "job_family": "Manufacturing",
            "job_level": 1,
            "salary_band_min": 42000,
            "salary_band_max": 65000,
        },
        {
            "job_role_id": 8,
            "job_title": "Production Supervisor",
            "job_family": "Manufacturing",
            "job_level": 3,
            "salary_band_min": 70000,
            "salary_band_max": 105000,
        },
        {
            "job_role_id": 9,
            "job_title": "Supply Chain Analyst",
            "job_family": "Supply Chain",
            "job_level": 2,
            "salary_band_min": 65000,
            "salary_band_max": 95000,
        },
        {
            "job_role_id": 10,
            "job_title": "Operations Research Analyst",
            "job_family": "Supply Chain",
            "job_level": 3,
            "salary_band_min": 85000,
            "salary_band_max": 125000,
        },
        {
            "job_role_id": 11,
            "job_title": "Procurement Specialist",
            "job_family": "Supply Chain",
            "job_level": 2,
            "salary_band_min": 60000,
            "salary_band_max": 90000,
        },
        {
            "job_role_id": 12,
            "job_title": "Sales Representative",
            "job_family": "Sales",
            "job_level": 2,
            "salary_band_min": 55000,
            "salary_band_max": 90000,
        },
        {
            "job_role_id": 13,
            "job_title": "Account Manager",
            "job_family": "Sales",
            "job_level": 3,
            "salary_band_min": 80000,
            "salary_band_max": 125000,
        },
        {
            "job_role_id": 14,
            "job_title": "Financial Analyst",
            "job_family": "Finance",
            "job_level": 2,
            "salary_band_min": 65000,
            "salary_band_max": 95000,
        },
        {
            "job_role_id": 15,
            "job_title": "Risk Analyst",
            "job_family": "Finance",
            "job_level": 2,
            "salary_band_min": 70000,
            "salary_band_max": 105000,
        },
        {
            "job_role_id": 16,
            "job_title": "Human Resources Specialist",
            "job_family": "Human Resources",
            "job_level": 2,
            "salary_band_min": 60000,
            "salary_band_max": 90000,
        },
        {
            "job_role_id": 17,
            "job_title": "Recruiter",
            "job_family": "Human Resources",
            "job_level": 2,
            "salary_band_min": 60000,
            "salary_band_max": 95000,
        },
        {
            "job_role_id": 18,
            "job_title": "IT Systems Analyst",
            "job_family": "Information Technology",
            "job_level": 2,
            "salary_band_min": 75000,
            "salary_band_max": 110000,
        },
        {
            "job_role_id": 19,
            "job_title": "Cybersecurity Risk Analyst",
            "job_family": "Information Technology",
            "job_level": 3,
            "salary_band_min": 90000,
            "salary_band_max": 135000,
        },
        {
            "job_role_id": 20,
            "job_title": "Customer Support Specialist",
            "job_family": "Customer Support",
            "job_level": 1,
            "salary_band_min": 42000,
            "salary_band_max": 65000,
        },
    ]

    return pd.DataFrame(job_roles)


def create_training_programs() -> pd.DataFrame:
    """Create the training-program reference table."""

    training_programs = [
        {
            "program_id": 1,
            "program_name": "New Employee Orientation",
            "program_category": "Onboarding",
            "required_hours": 8,
            "mandatory": True,
        },
        {
            "program_id": 2,
            "program_name": "Workplace Safety Fundamentals",
            "program_category": "Safety",
            "required_hours": 6,
            "mandatory": True,
        },
        {
            "program_id": 3,
            "program_name": "Data Privacy and Security",
            "program_category": "Compliance",
            "required_hours": 4,
            "mandatory": True,
        },
        {
            "program_id": 4,
            "program_name": "Python for Data Analysis",
            "program_category": "Technical",
            "required_hours": 20,
            "mandatory": False,
        },
        {
            "program_id": 5,
            "program_name": "SQL Fundamentals",
            "program_category": "Technical",
            "required_hours": 16,
            "mandatory": False,
        },
        {
            "program_id": 6,
            "program_name": "First-Time Manager Program",
            "program_category": "Leadership",
            "required_hours": 24,
            "mandatory": False,
        },
        {
            "program_id": 7,
            "program_name": "Advanced Leadership Development",
            "program_category": "Leadership",
            "required_hours": 32,
            "mandatory": False,
        },
        {
            "program_id": 8,
            "program_name": "Lean Manufacturing Fundamentals",
            "program_category": "Operations",
            "required_hours": 18,
            "mandatory": False,
        },
        {
            "program_id": 9,
            "program_name": "Supply Chain Risk Management",
            "program_category": "Operations",
            "required_hours": 14,
            "mandatory": False,
        },
        {
            "program_id": 10,
            "program_name": "Inclusive Workplace Practices",
            "program_category": "Compliance",
            "required_hours": 6,
            "mandatory": True,
        },
    ]

    return pd.DataFrame(training_programs)


def validate_primary_key(
    dataframe: pd.DataFrame,
    primary_key: str,
    table_name: str,
) -> None:
    """Check that a primary-key column is complete and unique."""

    if primary_key not in dataframe.columns:
        raise ValueError(
            f"{table_name}: primary-key column '{primary_key}' does not exist."
        )

    if dataframe[primary_key].isna().any():
        raise ValueError(
            f"{table_name}: primary key '{primary_key}' contains missing values."
        )

    if dataframe[primary_key].duplicated().any():
        raise ValueError(
            f"{table_name}: primary key '{primary_key}' contains duplicates."
        )


def validate_required_values(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Check that required columns do not contain missing values."""

    for column in required_columns:
        if column not in dataframe.columns:
            raise ValueError(
                f"{table_name}: required column '{column}' does not exist."
            )

        if dataframe[column].isna().any():
            raise ValueError(
                f"{table_name}: required column '{column}' contains missing values."
            )


def validate_job_roles(job_roles: pd.DataFrame) -> None:
    """Check that job-role salary bands are logically valid."""

    invalid_bands = (
        job_roles["salary_band_min"]
        >= job_roles["salary_band_max"]
    )

    if invalid_bands.any():
        invalid_ids = job_roles.loc[
            invalid_bands,
            "job_role_id",
        ].tolist()

        raise ValueError(
            "job_roles: salary_band_min must be lower than "
            f"salary_band_max. Invalid role IDs: {invalid_ids}"
        )


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save a DataFrame as a CSV file."""

    output_path = RAW_DATA_DIR / filename
    dataframe.to_csv(output_path, index=False)

    print(
        f"Saved {filename}: "
        f"{len(dataframe)} rows and {len(dataframe.columns)} columns"
    )


def main() -> None:
    """Create, validate, and save all reference tables."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    departments = create_departments()
    locations = create_locations()
    job_roles = create_job_roles()
    training_programs = create_training_programs()

    validate_primary_key(
        departments,
        "department_id",
        "departments",
    )
    validate_required_values(
        departments,
        [
            "department_name",
            "department_group",
            "cost_center",
        ],
        "departments",
    )

    validate_primary_key(
        locations,
        "location_id",
        "locations",
    )
    validate_required_values(
        locations,
        [
            "city",
            "state",
            "region",
            "location_type",
        ],
        "locations",
    )

    validate_primary_key(
        job_roles,
        "job_role_id",
        "job_roles",
    )
    validate_required_values(
        job_roles,
        [
            "job_title",
            "job_family",
            "job_level",
            "salary_band_min",
            "salary_band_max",
        ],
        "job_roles",
    )
    validate_job_roles(job_roles)

    validate_primary_key(
        training_programs,
        "program_id",
        "training_programs",
    )
    validate_required_values(
        training_programs,
        [
            "program_name",
            "program_category",
            "required_hours",
            "mandatory",
        ],
        "training_programs",
    )

    save_table(
        departments,
        "departments.csv",
    )
    save_table(
        locations,
        "locations.csv",
    )
    save_table(
        job_roles,
        "job_roles.csv",
    )
    save_table(
        training_programs,
        "training_programs.csv",
    )

    print("\nAll reference tables were generated successfully.")


if __name__ == "__main__":
    main()