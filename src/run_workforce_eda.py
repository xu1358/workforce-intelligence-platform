"""Run reproducible exploratory analysis for the Version 1 workforce data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "eda"
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_FILE = (
    PROCESSED_DIR / "retention_modeling_dataset.csv"
)

TARGET = "attrition_next_12m"
MINIMUM_GROUP_SIZE = 30


def grouped_attrition_rate(
    data: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """Calculate attrition rate and lift by category."""

    work = data[[feature, TARGET]].copy()

    work[feature] = (
        work[feature]
        .astype("object")
        .where(work[feature].notna(), "Missing")
        .astype(str)
    )

    baseline_rate = work[TARGET].mean()

    result = (
        work.groupby(feature, dropna=False)
        .agg(
            sample_size=(TARGET, "size"),
            attrition_cases=(TARGET, "sum"),
            attrition_rate=(TARGET, "mean"),
        )
        .reset_index()
        .rename(columns={feature: "category"})
    )

    result["baseline_attrition_rate"] = (
        baseline_rate
    )

    result["lift_vs_baseline"] = (
        result["attrition_rate"]
        / baseline_rate
    )

    result.insert(0, "feature", feature)

    return result


def composition_summary(
    data: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """Calculate workforce counts and shares."""

    values = (
        data[feature]
        .astype("object")
        .where(data[feature].notna(), "Missing")
        .astype(str)
    )

    result = (
        values.value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="headcount")
    )

    result["workforce_share"] = (
        result["headcount"] / len(data)
    )

    result.insert(0, "feature", feature)

    return result


def corrected_cramers_v(
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Calculate bias-corrected Cramer's V."""

    first = (
        first.astype("object")
        .where(first.notna(), "Missing")
        .astype(str)
    )

    second = (
        second.astype("object")
        .where(second.notna(), "Missing")
        .astype(str)
    )

    table = pd.crosstab(first, second)

    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    chi_squared = chi2_contingency(
        table,
        correction=False,
    )[0]

    observations = table.to_numpy().sum()
    rows, columns = table.shape

    phi_squared = chi_squared / observations

    correction = (
        (columns - 1) * (rows - 1)
        / (observations - 1)
    )

    corrected_phi_squared = max(
        0.0,
        phi_squared - correction,
    )

    corrected_rows = (
        rows
        - ((rows - 1) ** 2)
        / (observations - 1)
    )

    corrected_columns = (
        columns
        - ((columns - 1) ** 2)
        / (observations - 1)
    )

    denominator = min(
        corrected_rows - 1,
        corrected_columns - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sqrt(
            corrected_phi_squared
            / denominator
        )
    )


def save_rate_chart(
    table: pd.DataFrame,
    title: str,
    filename: str,
    category_order: list[str] | None = None,
) -> None:
    """Save a horizontal attrition-rate chart."""

    plot_data = table.loc[
        table["sample_size"].ge(
            MINIMUM_GROUP_SIZE
        )
    ].copy()

    if category_order is not None:
        plot_data["category"] = pd.Categorical(
            plot_data["category"],
            categories=category_order,
            ordered=True,
        )

        plot_data = plot_data.sort_values(
            "category"
        )
    else:
        plot_data = plot_data.sort_values(
            "attrition_rate"
        )

    plot_data["attrition_percent"] = (
        plot_data["attrition_rate"] * 100
    )

    baseline_percent = (
        plot_data["baseline_attrition_rate"]
        .iloc[0]
        * 100
    )

    figure_height = max(
        4.0,
        0.45 * len(plot_data),
    )

    figure, axis = plt.subplots(
        figsize=(10, figure_height)
    )

    axis.barh(
        plot_data["category"].astype(str),
        plot_data["attrition_percent"],
        color="#1565C0",
    )

    axis.axvline(
        baseline_percent,
        color="#C62828",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Overall rate: "
            f"{baseline_percent:.1f}%"
        ),
    )

    axis.set_title(title)
    axis.set_xlabel("Attrition in next 12 months (%)")
    axis.set_ylabel("")
    axis.grid(
        axis="x",
        alpha=0.25,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Run the complete exploratory analysis."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(MODEL_FILE)

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    data["prediction_end_date"] = pd.to_datetime(
        data["prediction_end_date"]
    )

    if set(data[TARGET].dropna().unique()) != {
        0,
        1,
    }:
        raise ValueError(
            f"{TARGET} must contain only 0 and 1."
        )

    employees = pd.read_csv(
        RAW_DIR / "employees.csv",
        parse_dates=[
            "hire_date",
            "termination_date",
        ],
    )

    departments = pd.read_csv(
        RAW_DIR / "departments.csv"
    )

    job_roles = pd.read_csv(
        RAW_DIR / "job_roles.csv"
    )

    locations = pd.read_csv(
        RAW_DIR / "locations.csv"
    )

    employees = (
        employees.merge(
            departments[
                [
                    "department_id",
                    "department_name",
                    "department_group",
                ]
            ],
            on="department_id",
            how="left",
        )
        .merge(
            job_roles[
                [
                    "job_role_id",
                    "job_title",
                    "job_family",
                    "job_level",
                ]
            ],
            on="job_role_id",
            how="left",
        )
        .merge(
            locations[
                [
                    "location_id",
                    "city",
                    "state",
                    "region",
                ]
            ],
            on="location_id",
            how="left",
        )
    )

    metadata = pd.DataFrame(
        {
            "metric": [
                "Workforce records",
                "Modeling records",
                "Modeling columns",
                "Unique employees in modeling data",
                "Snapshot dates",
                "Prediction end dates",
                "Overall attrition rate",
            ],
            "value": [
                len(employees),
                len(data),
                data.shape[1],
                data["employee_id"].nunique(),
                ", ".join(
                    data["snapshot_date"]
                    .dt.strftime("%Y-%m-%d")
                    .sort_values()
                    .unique()
                ),
                ", ".join(
                    data["prediction_end_date"]
                    .dt.strftime("%Y-%m-%d")
                    .sort_values()
                    .unique()
                ),
                f"{data[TARGET].mean():.6f}",
            ],
        }
    )

    target_balance = (
        data[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("target_value")
        .reset_index(name="record_count")
    )

    target_balance["target_label"] = (
        target_balance["target_value"].map(
            {
                0: "No attrition",
                1: "Attrition",
            }
        )
    )

    target_balance["record_share"] = (
        target_balance["record_count"]
        / len(data)
    )

    workforce_features = [
        "employment_status",
        "organizational_level",
        "employment_type",
        "department_name",
        "department_group",
        "city",
        "region",
        "job_family",
        "job_level",
        "education_level",
    ]

    workforce_composition = pd.concat(
        [
            composition_summary(
                employees,
                feature,
            )
            for feature in workforce_features
        ],
        ignore_index=True,
    )

    missingness = pd.DataFrame(
        {
            "feature": data.columns,
            "missing_count": (
                data.isna().sum().values
            ),
            "missing_rate": (
                data.isna().mean().values
            ),
        }
    )

    missing_target_rows = []

    for feature in data.columns:
        missing_mask = data[feature].isna()

        if not missing_mask.any():
            continue

        missing_target_rows.append(
            {
                "feature": feature,
                "missing_count": int(
                    missing_mask.sum()
                ),
                "attrition_rate_when_missing": (
                    data.loc[
                        missing_mask,
                        TARGET,
                    ].mean()
                ),
                "attrition_rate_when_present": (
                    data.loc[
                        ~missing_mask,
                        TARGET,
                    ].mean()
                ),
            }
        )

    missing_target_associations = pd.DataFrame(
        missing_target_rows
    )

    if not missing_target_associations.empty:
        missing_target_associations[
            "rate_difference"
        ] = (
            missing_target_associations[
                "attrition_rate_when_missing"
            ]
            - missing_target_associations[
                "attrition_rate_when_present"
            ]
        )

    excluded_numeric = {
        "employee_id",
        TARGET,
    }

    numeric_features = [
        feature
        for feature in data.select_dtypes(
            include=np.number
        ).columns
        if feature not in excluded_numeric
    ]

    numeric_summary = (
        data[numeric_features]
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    numeric_target_rows = []

    for feature in numeric_features:
        active_values = (
            data.loc[
                data[TARGET].eq(0),
                feature,
            ]
            .dropna()
        )

        attrition_values = (
            data.loc[
                data[TARGET].eq(1),
                feature,
            ]
            .dropna()
        )

        combined_standard_deviation = np.sqrt(
            (
                active_values.var(ddof=1)
                + attrition_values.var(ddof=1)
            )
            / 2
        )

        if (
            pd.isna(combined_standard_deviation)
            or combined_standard_deviation == 0
        ):
            standardized_mean_difference = np.nan
        else:
            standardized_mean_difference = (
                attrition_values.mean()
                - active_values.mean()
            ) / combined_standard_deviation

        valid_rows = data[
            [feature, TARGET]
        ].dropna()

        point_biserial_correlation = (
            valid_rows[feature].corr(
                valid_rows[TARGET]
            )
        )

        numeric_target_rows.append(
            {
                "feature": feature,
                "active_mean": (
                    active_values.mean()
                ),
                "attrition_mean": (
                    attrition_values.mean()
                ),
                "mean_difference": (
                    attrition_values.mean()
                    - active_values.mean()
                ),
                "standardized_mean_difference": (
                    standardized_mean_difference
                ),
                "point_biserial_correlation": (
                    point_biserial_correlation
                ),
            }
        )

    numeric_target_associations = pd.DataFrame(
        numeric_target_rows
    )

    numeric_target_associations[
        "absolute_standardized_difference"
    ] = (
        numeric_target_associations[
            "standardized_mean_difference"
        ].abs()
    )

    numeric_target_associations = (
        numeric_target_associations.sort_values(
            "absolute_standardized_difference",
            ascending=False,
        )
    )

    data["age_band"] = pd.cut(
        data["approx_age"],
        bins=[
            0,
            25,
            35,
            45,
            55,
            np.inf,
        ],
        labels=[
            "Under 25",
            "25–34",
            "35–44",
            "45–54",
            "55+",
        ],
        right=False,
    )

    data["tenure_band"] = pd.cut(
        data["tenure_years"],
        bins=[
            -np.inf,
            0.5,
            1,
            2,
            3,
            5,
            np.inf,
        ],
        labels=[
            "Under 0.5 years",
            "0.5–0.99 years",
            "1–1.99 years",
            "2–2.99 years",
            "3–4.99 years",
            "5+ years",
        ],
        right=False,
    )

    data["salary_quintile"] = pd.qcut(
        data["base_salary"],
        q=5,
        labels=[
            "Q1: lowest",
            "Q2",
            "Q3",
            "Q4",
            "Q5: highest",
        ],
    )

    data["salary_growth_band"] = pd.cut(
        data["salary_growth_percent"],
        bins=[
            -np.inf,
            0,
            5,
            10,
            20,
            np.inf,
        ],
        labels=[
            "0% or lower",
            "0–4.99%",
            "5–9.99%",
            "10–19.99%",
            "20%+",
        ],
        right=False,
    )

    data["performance_band"] = pd.cut(
        data["performance_rating"],
        bins=[
            -np.inf,
            3.0,
            3.5,
            4.0,
            np.inf,
        ],
        labels=[
            "Below 3.0",
            "3.0–3.49",
            "3.5–3.99",
            "4.0+",
        ],
        right=False,
    )

    data["completed_training_band"] = pd.cut(
        data["completed_training_programs"],
        bins=[
            -0.1,
            0,
            1,
            2,
            3,
            np.inf,
        ],
        labels=[
            "0",
            "1",
            "2",
            "3",
            "4+",
        ],
    )

    data["has_prior_review"] = np.where(
        data["performance_rating"].notna(),
        "Prior review available",
        "No prior review",
    )

    data["has_prior_change_event"] = np.where(
        data[
            "days_since_last_change_event"
        ].notna(),
        "Prior change event",
        "No prior change event",
    )

    categorical_features = [
        "employment_type",
        "education_level",
        "application_source",
        "hire_department_name",
        "hire_region",
        "hire_job_family",
        "hire_job_level",
        "promotion_recommended",
    ]

    categorical_rates = pd.concat(
        [
            grouped_attrition_rate(
                data,
                feature,
            )
            for feature in categorical_features
        ],
        ignore_index=True,
    )

    binned_features = [
        "age_band",
        "tenure_band",
        "salary_quintile",
        "salary_growth_band",
        "performance_band",
        "completed_training_band",
        "has_prior_review",
        "has_prior_change_event",
    ]

    binned_rates = pd.concat(
        [
            grouped_attrition_rate(
                data,
                feature,
            )
            for feature in binned_features
        ],
        ignore_index=True,
    )

    categorical_strength_rows = []

    for feature in categorical_features:
        feature_rates = categorical_rates.loc[
            categorical_rates["feature"].eq(
                feature
            )
            & categorical_rates[
                "sample_size"
            ].ge(MINIMUM_GROUP_SIZE)
        ]

        rate_spread = (
            feature_rates["attrition_rate"].max()
            - feature_rates["attrition_rate"].min()
        )

        categorical_strength_rows.append(
            {
                "feature": feature,
                "cramers_v_with_target": (
                    corrected_cramers_v(
                        data[feature],
                        data[TARGET],
                    )
                ),
                "attrition_rate_spread": (
                    rate_spread
                ),
                "eligible_category_count": len(
                    feature_rates
                ),
            }
        )

    categorical_target_strength = (
        pd.DataFrame(
            categorical_strength_rows
        )
        .sort_values(
            "cramers_v_with_target",
            ascending=False,
        )
    )

    numeric_correlation = (
        data[numeric_features].corr()
    )

    correlation_rows = []

    for first_index, first_feature in enumerate(
        numeric_features
    ):
        for second_feature in numeric_features[
            first_index + 1:
        ]:
            correlation = numeric_correlation.loc[
                first_feature,
                second_feature,
            ]

            if abs(correlation) >= 0.70:
                correlation_rows.append(
                    {
                        "first_feature": (
                            first_feature
                        ),
                        "second_feature": (
                            second_feature
                        ),
                        "correlation": correlation,
                        "absolute_correlation": abs(
                            correlation
                        ),
                    }
                )

    high_numeric_correlations = pd.DataFrame(
        correlation_rows,
        columns=[
            "first_feature",
            "second_feature",
            "correlation",
            "absolute_correlation",
        ],
    )

    if not high_numeric_correlations.empty:
        high_numeric_correlations = (
            high_numeric_correlations.sort_values(
                "absolute_correlation",
                ascending=False,
            )
        )

    categorical_pairs = [
        (
            "hire_department_name",
            "hire_job_family",
        ),
        (
            "hire_department_name",
            "employment_type",
        ),
        (
            "hire_department_name",
            "hire_region",
        ),
        (
            "hire_job_family",
            "employment_type",
        ),
        (
            "hire_job_family",
            "hire_job_level",
        ),
    ]

    categorical_redundancy = pd.DataFrame(
        [
            {
                "first_feature": first_feature,
                "second_feature": second_feature,
                "corrected_cramers_v": (
                    corrected_cramers_v(
                        data[first_feature],
                        data[second_feature],
                    )
                ),
            }
            for first_feature, second_feature
            in categorical_pairs
        ]
    ).sort_values(
        "corrected_cramers_v",
        ascending=False,
    )

    outputs = {
        "eda_metadata.csv": metadata,
        "target_balance.csv": target_balance,
        "workforce_composition.csv": (
            workforce_composition
        ),
        "missingness_summary.csv": (
            missingness.sort_values(
                "missing_rate",
                ascending=False,
            )
        ),
        "missingness_target_association.csv": (
            missing_target_associations
        ),
        "numeric_summary.csv": numeric_summary,
        "numeric_target_associations.csv": (
            numeric_target_associations
        ),
        "categorical_attrition_rates.csv": (
            categorical_rates
        ),
        "binned_attrition_rates.csv": (
            binned_rates
        ),
        "categorical_target_strength.csv": (
            categorical_target_strength
        ),
        "numeric_correlation_matrix.csv": (
            numeric_correlation.reset_index()
        ),
        "high_numeric_correlations.csv": (
            high_numeric_correlations
        ),
        "categorical_redundancy.csv": (
            categorical_redundancy
        ),
    }

    for filename, table in outputs.items():
        table.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    # Chart 1: target imbalance.
    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.bar(
        target_balance["target_label"],
        target_balance["record_count"],
        color=["#90CAF9", "#C62828"],
    )

    for position, row in (
        target_balance.reset_index(
            drop=True
        ).iterrows()
    ):
        axis.text(
            position,
            row["record_count"],
            (
                f'{row["record_count"]:,}\n'
                f'({row["record_share"]:.1%})'
            ),
            ha="center",
            va="bottom",
        )

    axis.set_title(
        "Attrition Is a Small Minority of "
        "12-Month Outcomes"
    )
    axis.set_ylabel("Employee snapshots")
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "01_class_imbalance.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Charts 2–5: categorical and binned rates.
    department_rates = categorical_rates.loc[
        categorical_rates["feature"].eq(
            "hire_department_name"
        )
    ]

    save_rate_chart(
        department_rates,
        (
            "Manufacturing and Customer Support "
            "Have the Highest Generated Risk"
        ),
        "02_attrition_by_department.png",
    )

    job_family_rates = categorical_rates.loc[
        categorical_rates["feature"].eq(
            "hire_job_family"
        )
    ]

    save_rate_chart(
        job_family_rates,
        (
            "12-Month Attrition Varies "
            "Across Job Families"
        ),
        "03_attrition_by_job_family.png",
    )

    tenure_rates = binned_rates.loc[
        binned_rates["feature"].eq(
            "tenure_band"
        )
    ]

    save_rate_chart(
        tenure_rates,
        (
            "Attrition Differences by Tenure "
            "Reflect Generator Timing Rules"
        ),
        "04_attrition_by_tenure.png",
        [
            "Under 0.5 years",
            "0.5–0.99 years",
            "1–1.99 years",
            "2–2.99 years",
            "3–4.99 years",
            "5+ years",
        ],
    )

    performance_rates = binned_rates.loc[
        binned_rates["feature"].eq(
            "performance_band"
        )
    ]

    save_rate_chart(
        performance_rates,
        (
            "Lower Performance Appears Predictive "
            "but Is Partly Outcome-Conditioned"
        ),
        "05_attrition_by_performance.png",
        [
            "Missing",
            "Below 3.0",
            "3.0–3.49",
            "3.5–3.99",
            "4.0+",
        ],
    )

    # Chart 6: salary distributions.
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.boxplot(
        [
            data.loc[
                data[TARGET].eq(0),
                "base_salary",
            ],
            data.loc[
                data[TARGET].eq(1),
                "base_salary",
            ],
        ],
        tick_labels=[
            "No attrition",
            "Attrition",
        ],
        showfliers=False,
    )

    axis.set_title(
        "Base Salary Distribution by "
        "12-Month Outcome"
    )
    axis.set_ylabel("Base salary (USD)")
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "06_salary_by_outcome.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Chart 7: missingness.
    missing_plot = (
        missingness.loc[
            missingness["missing_count"].gt(0)
        ]
        .sort_values("missing_rate")
        .tail(12)
        .copy()
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.barh(
        missing_plot["feature"],
        missing_plot["missing_rate"] * 100,
        color="#6A1B9A",
    )

    axis.set_title(
        "Most Missing Values Represent "
        "No Prior Event or Review"
    )
    axis.set_xlabel("Missing records (%)")
    axis.set_ylabel("")
    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "07_missingness.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Chart 8: selected numeric correlations.
    correlation_features = [
        feature
        for feature in [
            "approx_age",
            "tenure_years",
            "years_experience_at_hire",
            "initial_base_salary",
            "base_salary",
            "salary_growth_percent",
            "compensation_record_count",
            "performance_rating",
            "goal_completion",
            "review_count",
            "completed_training_programs",
            "completed_training_hours",
            "prior_promotion_events",
            "prior_change_events",
        ]
        if feature in data.columns
    ]

    selected_correlation = (
        data[correlation_features].corr()
    )

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    image = axis.imshow(
        selected_correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(correlation_features))
    )
    axis.set_yticks(
        range(len(correlation_features))
    )

    axis.set_xticklabels(
        correlation_features,
        rotation=60,
        ha="right",
    )
    axis.set_yticklabels(
        correlation_features
    )

    axis.set_title(
        "Selected Numeric Features Contain "
        "Several Redundant Relationships"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Pearson correlation",
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "08_numeric_correlation.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nEDA METADATA")
    print(metadata.to_string(index=False))

    print("\nTARGET BALANCE")
    print(target_balance.to_string(index=False))

    print("\nTOP MISSING FEATURES")
    print(
        missingness.sort_values(
            "missing_rate",
            ascending=False,
        )
        .head(12)
        .to_string(index=False)
    )

    print("\nDEPARTMENT ATTRITION RATES")
    print(
        department_rates.sort_values(
            "attrition_rate",
            ascending=False,
        ).to_string(index=False)
    )

    print("\nJOB-FAMILY ATTRITION RATES")
    print(
        job_family_rates.sort_values(
            "attrition_rate",
            ascending=False,
        ).to_string(index=False)
    )

    print("\nSTRONGEST NUMERIC ASSOCIATIONS")
    print(
        numeric_target_associations.head(12)
        .to_string(index=False)
    )

    print("\nCATEGORICAL TARGET STRENGTH")
    print(
        categorical_target_strength
        .to_string(index=False)
    )

    print("\nHIGH NUMERIC CORRELATIONS")
    if high_numeric_correlations.empty:
        print("No absolute correlations >= 0.70.")
    else:
        print(
            high_numeric_correlations
            .to_string(index=False)
        )

    print("\nCATEGORICAL REDUNDANCY")
    print(
        categorical_redundancy.to_string(
            index=False
        )
    )

    print(
        "\nEDA outputs saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()