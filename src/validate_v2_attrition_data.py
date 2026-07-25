"""Validate Version 2 attrition data and compare it with Version 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "v2_validation"

HAZARD_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "attrition_hazard_config.yaml"
)
VERSION1_BASELINE_PATH = (
    PROJECT_ROOT / "config" / "version1_validation_baseline.yaml"
)

NUMERIC_DIAGNOSTIC_FEATURES = [
    "tenure_years",
    "base_salary",
    "bonus_target",
    "equity_value",
    "salary_growth_percent",
    "performance_rating",
    "goal_completion",
    "average_performance_rating",
    "review_count",
    "completed_training_hours",
    "failed_training_count",
    "manager_changes_prior_12m",
    "months_since_promotion",
    "no_prior_promotion",
    "no_prior_review",
]

CATEGORICAL_DIAGNOSTIC_FEATURES = [
    "department_name",
    "city",
    "region",
    "job_family",
    "job_level",
    "employment_type",
    "organizational_level",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML configuration file."""

    with path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def load_table(
    filename: str,
    date_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one required raw table."""

    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")

    frame = pd.read_csv(path)

    for date_column in date_columns or []:
        frame[date_column] = pd.to_datetime(frame[date_column])

    return frame


def latest_before(
    frame: pd.DataFrame,
    date_column: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Return each employee's latest record on or before a cutoff."""

    return (
        frame.loc[frame[date_column].le(cutoff)]
        .sort_values(["employee_id", date_column])
        .groupby("employee_id", as_index=False)
        .tail(1)
    )


def create_snapshot_features(
    employees: pd.DataFrame,
    departments: pd.DataFrame,
    locations: pd.DataFrame,
    job_roles: pd.DataFrame,
    compensation: pd.DataFrame,
    performance: pd.DataFrame,
    training: pd.DataFrame,
    events: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    prediction_end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small leakage-safe diagnostic snapshot."""

    eligible = employees.loc[
        employees["hire_date"].le(snapshot_date)
        & (
            employees["termination_date"].isna()
            | employees["termination_date"].gt(snapshot_date)
        )
    ].copy()

    eligible["attrition_next_12m"] = (
        employees.loc[eligible.index, "termination_date"]
        .gt(snapshot_date)
        & employees.loc[eligible.index, "termination_date"].le(
            prediction_end_date
        )
    ).astype(int)

    eligible["tenure_years"] = (
        (snapshot_date - eligible["hire_date"]).dt.days / 365.25
    )

    eligible = (
        eligible.merge(departments, on="department_id", how="left")
        .merge(locations, on="location_id", how="left")
        .merge(job_roles, on="job_role_id", how="left")
    )

    compensation_before = compensation.loc[
        compensation["effective_date"].le(snapshot_date)
    ].sort_values(["employee_id", "effective_date"])

    initial_compensation = (
        compensation_before.groupby("employee_id", as_index=False)
        .first()[
            ["employee_id", "base_salary"]
        ]
        .rename(columns={"base_salary": "initial_base_salary"})
    )
    current_compensation = (
        compensation_before.groupby("employee_id", as_index=False)
        .last()[
            [
                "employee_id",
                "base_salary",
                "bonus_target",
                "equity_value",
            ]
        ]
    )

    compensation_features = current_compensation.merge(
        initial_compensation,
        on="employee_id",
        how="left",
    )
    compensation_features["salary_growth_percent"] = (
        (
            compensation_features["base_salary"]
            / compensation_features["initial_base_salary"]
            - 1.0
        )
        * 100.0
    )

    performance_before = performance.loc[
        performance["review_date"].le(snapshot_date)
    ].sort_values(["employee_id", "review_date"])

    latest_performance = (
        performance_before.groupby("employee_id", as_index=False)
        .last()[
            [
                "employee_id",
                "performance_rating",
                "goal_completion",
            ]
        ]
    )
    average_performance = (
        performance_before.groupby("employee_id", as_index=False)
        .agg(
            average_performance_rating=(
                "performance_rating",
                "mean",
            ),
            review_count=("review_id", "size"),
        )
    )
    performance_features = latest_performance.merge(
        average_performance,
        on="employee_id",
        how="outer",
    )
    performance_features["no_prior_review"] = (
        performance_features["review_count"].isna().astype(int)
    )

    trailing_start = snapshot_date - pd.DateOffset(years=1)
    training_before = training.loc[
        training["completion_date"].notna()
        & training["completion_date"].gt(trailing_start)
        & training["completion_date"].le(snapshot_date)
    ].copy()

    training_features = (
        training_before.groupby("employee_id", as_index=False)
        .agg(
            completed_training_hours=(
                "training_hours",
                lambda values: float(
                    values[
                        training_before.loc[
                            values.index,
                            "completion_status",
                        ].eq("Completed")
                    ].sum()
                ),
            ),
            failed_training_count=(
                "completion_status",
                lambda values: int(values.eq("Failed").sum()),
            ),
        )
    )

    prior_events = events.loc[
        events["event_date"].le(snapshot_date)
    ].copy()
    trailing_events = prior_events.loc[
        prior_events["event_date"].gt(trailing_start)
    ]

    manager_change_features = (
        trailing_events.loc[
            trailing_events["event_type"].eq("Manager Change")
        ]
        .groupby("employee_id")
        .size()
        .rename("manager_changes_prior_12m")
        .reset_index()
    )

    promotions = prior_events.loc[
        prior_events["event_type"].eq("Promotion")
    ].sort_values(["employee_id", "event_date"])
    latest_promotions = (
        promotions.groupby("employee_id", as_index=False)
        .last()[["employee_id", "event_date"]]
        .rename(columns={"event_date": "latest_promotion_date"})
    )

    snapshot = (
        eligible.merge(
            compensation_features,
            on="employee_id",
            how="left",
        )
        .merge(
            performance_features,
            on="employee_id",
            how="left",
        )
        .merge(
            training_features,
            on="employee_id",
            how="left",
        )
        .merge(
            manager_change_features,
            on="employee_id",
            how="left",
        )
        .merge(
            latest_promotions,
            on="employee_id",
            how="left",
        )
    )

    snapshot["months_since_promotion"] = (
        (
            snapshot_date.to_period("M")
            - snapshot["latest_promotion_date"].dt.to_period("M")
        )
        .apply(lambda value: value.n if pd.notna(value) else np.nan)
    )
    snapshot["no_prior_promotion"] = (
        snapshot["latest_promotion_date"].isna().astype(int)
    )
    snapshot["months_since_promotion"] = snapshot[
        "months_since_promotion"
    ].fillna(snapshot["tenure_years"] * 12.0)

    count_columns = [
        "review_count",
        "completed_training_hours",
        "failed_training_count",
        "manager_changes_prior_12m",
    ]
    snapshot[count_columns] = snapshot[count_columns].fillna(0)
    snapshot["no_prior_review"] = snapshot["no_prior_review"].fillna(1)

    source_cutoffs = pd.DataFrame(
        {
            "source": [
                "compensation_history",
                "performance_reviews",
                "training_records",
                "employee_events",
            ],
            "latest_date_used": [
                compensation_before["effective_date"].max(),
                performance_before["review_date"].max(),
                training_before["completion_date"].max(),
                prior_events["event_date"].max(),
            ],
            "required_cutoff": snapshot_date,
        }
    )

    return snapshot, source_cutoffs


def numeric_signal_table(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Measure simple numeric relationships with the future target."""

    numeric_features = [
        *NUMERIC_DIAGNOSTIC_FEATURES,
        "job_level",
    ]

    records = []

    for feature in numeric_features:
        values = pd.to_numeric(snapshot[feature], errors="coerce")
        valid = values.notna()
        active_values = values.loc[
            valid & snapshot["attrition_next_12m"].eq(0)
        ]
        attrition_values = values.loc[
            valid & snapshot["attrition_next_12m"].eq(1)
        ]

        if values.loc[valid].nunique() <= 1:
            correlation = 0.0
        else:
            correlation = values.loc[valid].corr(
                snapshot.loc[valid, "attrition_next_12m"]
            )

        pooled_standard_deviation = values.loc[valid].std()
        mean_difference = (
            attrition_values.mean() - active_values.mean()
        )
        standardized_difference = (
            mean_difference / pooled_standard_deviation
            if pooled_standard_deviation
            and not pd.isna(pooled_standard_deviation)
            else 0.0
        )

        records.append(
            {
                "feature": feature,
                "non_missing_records": int(valid.sum()),
                "no_attrition_mean": active_values.mean(),
                "attrition_mean": attrition_values.mean(),
                "mean_difference": mean_difference,
                "standardized_mean_difference": standardized_difference,
                "point_biserial_correlation": correlation,
                "absolute_correlation": abs(float(correlation)),
            }
        )

    return pd.DataFrame(records).sort_values(
        "absolute_correlation",
        ascending=False,
    )


def categorical_signal_table(
    snapshot: pd.DataFrame,
    minimum_group_size: int,
) -> pd.DataFrame:
    """Measure category-level attrition rates and baseline lift."""

    categorical_features = CATEGORICAL_DIAGNOSTIC_FEATURES
    baseline_rate = snapshot["attrition_next_12m"].mean()
    tables = []

    for feature in categorical_features:
        summary = (
            snapshot.assign(
                _category=snapshot[feature].astype(str)
            )
            .groupby("_category", dropna=False)
            .agg(
                sample_size=("employee_id", "size"),
                positive_cases=("attrition_next_12m", "sum"),
                attrition_rate=("attrition_next_12m", "mean"),
            )
            .reset_index()
            .rename(columns={"_category": "category"})
        )
        summary.insert(0, "feature", feature)
        summary["baseline_attrition_rate"] = baseline_rate
        summary["rate_ratio_vs_baseline"] = (
            summary["attrition_rate"] / baseline_rate
        )
        summary["included_in_rate_check"] = summary[
            "sample_size"
        ].ge(minimum_group_size)
        tables.append(summary)

    return pd.concat(tables, ignore_index=True).sort_values(
        ["feature", "attrition_rate"],
        ascending=[True, False],
    )


def diagnostic_model_evaluation(
    snapshot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure whether observable signal is moderate and detectable."""

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_DIAGNOSTIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_DIAGNOSTIC_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    feature_columns = (
        NUMERIC_DIAGNOSTIC_FEATURES
        + CATEGORICAL_DIAGNOSTIC_FEATURES
    )
    results = cross_validate(
        model,
        snapshot[feature_columns],
        snapshot["attrition_next_12m"],
        cv=folds,
        scoring={
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        },
        return_train_score=False,
    )

    fold_results = pd.DataFrame(
        {
            "fold": range(1, 6),
            "roc_auc": results["test_roc_auc"],
            "pr_auc": results["test_pr_auc"],
        }
    )
    baseline_pr_auc = float(
        snapshot["attrition_next_12m"].mean()
    )
    model_summary = pd.DataFrame(
        {
            "metric": [
                "Mean ROC-AUC",
                "ROC-AUC standard deviation",
                "Mean PR-AUC",
                "PR-AUC standard deviation",
                "No-skill PR-AUC",
                "PR-AUC lift over no-skill",
            ],
            "value": [
                fold_results["roc_auc"].mean(),
                fold_results["roc_auc"].std(ddof=0),
                fold_results["pr_auc"].mean(),
                fold_results["pr_auc"].std(ddof=0),
                baseline_pr_auc,
                (
                    fold_results["pr_auc"].mean()
                    / baseline_pr_auc
                ),
            ],
        }
    )

    return fold_results, model_summary


def table_hashes(paths: list[Path]) -> pd.DataFrame:
    """Create SHA-256 fingerprints for reproducibility checks."""

    records = []

    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )

    return pd.DataFrame(records)


def grouped_attrition(
    employees: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Summarize cumulative attrition for one grouping."""

    return (
        employees.groupby(group_column, dropna=False)
        .agg(
            headcount=("employee_id", "size"),
            terminations=(
                "employment_status",
                lambda values: int(values.eq("Terminated").sum()),
            ),
            cumulative_attrition_rate=(
                "employment_status",
                lambda values: float(values.eq("Terminated").mean()),
            ),
        )
        .reset_index()
        .sort_values(
            "cumulative_attrition_rate",
            ascending=False,
        )
    )


def main() -> None:
    """Run all Version 2 validation and comparison checks."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hazard_config = load_yaml(HAZARD_CONFIG_PATH)
    version1 = load_yaml(VERSION1_BASELINE_PATH)

    as_of_date = pd.Timestamp(hazard_config["simulation"]["as_of_date"])
    snapshot_date = pd.Timestamp(
        version1["modeling_snapshot"]["snapshot_date"]
    )
    prediction_end_date = pd.Timestamp(
        version1["modeling_snapshot"]["prediction_end_date"]
    )

    employees = load_table(
        "employees.csv",
        ["hire_date", "termination_date"],
    )
    departments = load_table("departments.csv")
    locations = load_table("locations.csv")
    job_roles = load_table("job_roles.csv")
    compensation = load_table(
        "compensation_history.csv",
        ["effective_date"],
    )
    performance = load_table(
        "performance_reviews.csv",
        ["review_date"],
    )
    training = load_table(
        "training_records.csv",
        ["start_date", "completion_date"],
    )
    events = load_table("employee_events.csv", ["event_date"])

    outcomes = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "attrition_hazard_outcomes.csv",
        parse_dates=["termination_date"],
    )
    monthly_diagnostics = pd.read_csv(
        INTERIM_DATA_DIR
        / "attrition_hazard_monthly_diagnostics.csv",
        parse_dates=["simulation_month"],
    )

    snapshot, source_cutoffs = create_snapshot_features(
        employees,
        departments,
        locations,
        job_roles,
        compensation,
        performance,
        training,
        events,
        snapshot_date,
        prediction_end_date,
    )

    numeric_signals = numeric_signal_table(snapshot)
    minimum_group_size = int(
        hazard_config["validation_rules"][
            "minimum_subgroup_size_for_rate_check"
        ]
    )
    categorical_signals = categorical_signal_table(
        snapshot,
        minimum_group_size,
    )
    diagnostic_model_folds, diagnostic_model_summary = (
        diagnostic_model_evaluation(snapshot)
    )

    employees_detailed = (
        employees.merge(departments, on="department_id", how="left")
        .merge(locations, on="location_id", how="left")
        .merge(job_roles, on="job_role_id", how="left")
    )
    organizational_summary = grouped_attrition(
        employees_detailed,
        "organizational_level",
    )
    department_summary = grouped_attrition(
        employees_detailed,
        "department_name",
    )
    location_summary = grouped_attrition(
        employees_detailed,
        "city",
    )

    terminated = employees["employment_status"].eq("Terminated")
    active = employees["employment_status"].eq("Active")
    voluntary_count = int(
        employees["termination_type"].eq("Voluntary").sum()
    )
    involuntary_count = int(
        employees["termination_type"].eq("Involuntary").sum()
    )
    termination_count = int(terminated.sum())
    cumulative_rate = float(terminated.mean())
    voluntary_share = voluntary_count / termination_count
    snapshot_positive_cases = int(snapshot["attrition_next_12m"].sum())
    snapshot_rate = float(snapshot["attrition_next_12m"].mean())

    protected_levels = set(
        hazard_config["hierarchy"]["protected_levels"]
    )
    protected_terminations = int(
        employees.loc[
            employees["organizational_level"].isin(protected_levels),
            "employment_status",
        ]
        .eq("Terminated")
        .sum()
    )
    team_manager_terminations = int(
        (
            employees["organizational_level"].eq("Team Manager")
            & terminated
        ).sum()
    )

    end_dates = (
        employees.set_index("employee_id")["termination_date"]
        .fillna(as_of_date)
    )
    post_employment_counts = {}

    for table_name, table, date_column in [
        ("compensation", compensation, "effective_date"),
        ("performance", performance, "review_date"),
        ("training_start", training, "start_date"),
        ("events", events, "event_date"),
    ]:
        post_employment_counts[table_name] = int(
            (
                table[date_column]
                > table["employee_id"].map(end_dates)
            ).sum()
        )

    post_employment_counts["training_completion"] = int(
        (
            training["completion_date"].notna()
            & (
                training["completion_date"]
                > training["employee_id"].map(end_dates)
            )
        ).sum()
    )

    termination_event_counts = (
        events.loc[events["event_type"].eq("Termination")]
        .groupby("employee_id")
        .size()
        .reindex(employees["employee_id"], fill_value=0)
    )

    employee_lookup = employees.set_index("employee_id")
    active_with_manager = employees.loc[
        active & employees["manager_id"].notna()
    ].copy()
    active_with_manager["manager_status"] = (
        active_with_manager["manager_id"]
        .astype(int)
        .map(employee_lookup["employment_status"])
    )
    active_with_manager["manager_department"] = (
        active_with_manager["manager_id"]
        .astype(int)
        .map(employee_lookup["department_id"])
    )

    manager_reassignments = int(
        events["notes"]
        .str.contains(
            "simulated manager termination",
            case=False,
            na=False,
        )
        .sum()
    )

    monthly_diagnostics["total_exits"] = (
        monthly_diagnostics["voluntary_exits"]
        + monthly_diagnostics["involuntary_exits"]
    )
    monthly_diagnostics["observed_monthly_hazard"] = (
        monthly_diagnostics["total_exits"]
        / monthly_diagnostics["risk_set_size"]
    )
    monthly_diagnostics["average_combined_probability"] = (
        monthly_diagnostics["average_voluntary_probability"]
        + monthly_diagnostics["average_involuntary_probability"]
    )

    max_numeric_correlation = float(
        numeric_signals["absolute_correlation"].max()
    )
    checked_categories = categorical_signals.loc[
        categorical_signals["included_in_rate_check"]
    ]
    max_category_rate_ratio = float(
        checked_categories["rate_ratio_vs_baseline"].max()
    )
    diagnostic_roc_auc = float(
        diagnostic_model_folds["roc_auc"].mean()
    )
    diagnostic_pr_auc = float(
        diagnostic_model_folds["pr_auc"].mean()
    )
    no_skill_pr_auc = snapshot_rate

    validation_records: list[dict[str, Any]] = []

    def add_check(
        check: str,
        passed: bool,
        observed: Any,
        requirement: str,
        details: str,
    ) -> None:
        validation_records.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "requirement": requirement,
                "details": details,
            }
        )

    targets = hazard_config["calibration_targets"]
    rules = hazard_config["validation_rules"]

    add_check(
        "Workforce row count",
        len(employees) == 10_000,
        len(employees),
        "10,000",
        "The Version 2 workforce preserves the designed cohort size.",
    )
    add_check(
        "Unique employee IDs",
        employees["employee_id"].is_unique,
        employees["employee_id"].nunique(),
        "10,000 unique IDs",
        "Each employee must have exactly one workforce record.",
    )
    add_check(
        "Valid termination dates",
        bool(
            employees.loc[
                terminated,
                "termination_date",
            ].between(
                employees.loc[terminated, "hire_date"],
                as_of_date,
            ).all()
        ),
        termination_count,
        "Every exit between hire and as-of date",
        "No employee may terminate before hire or after the simulation.",
    )
    add_check(
        "No post-employment records",
        sum(post_employment_counts.values()) == 0,
        sum(post_employment_counts.values()),
        "0",
        str(post_employment_counts),
    )
    add_check(
        "Termination event reconciliation",
        bool(
            termination_event_counts.loc[
                employees.loc[terminated, "employee_id"]
            ].eq(1).all()
            and termination_event_counts.loc[
                employees.loc[active, "employee_id"]
            ].eq(0).all()
        ),
        int(
            events["event_type"].eq("Termination").sum()
        ),
        f"{termination_count}",
        "Each terminated employee has one event; active employees have none.",
    )
    add_check(
        "Active manager status",
        active_with_manager["manager_status"].eq("Active").all(),
        int(
            active_with_manager["manager_status"].ne("Active").sum()
        ),
        "0 active employees with inactive managers",
        "Manager exits must trigger valid reassignment.",
    )
    add_check(
        "Manager department consistency",
        active_with_manager["manager_department"].eq(
            active_with_manager["department_id"]
        ).all(),
        int(
            active_with_manager["manager_department"].ne(
                active_with_manager["department_id"]
            ).sum()
        ),
        "0 mismatches",
        "Active employees and current managers remain in one department.",
    )
    add_check(
        "Protected hierarchy levels",
        protected_terminations == 0,
        protected_terminations,
        "0",
        f"Protected levels: {sorted(protected_levels)}",
    )
    add_check(
        "Nonzero team-manager attrition",
        team_manager_terminations > 0,
        team_manager_terminations,
        "> 0",
        "Version 2 removes the all-leaders-active Version 1 artifact.",
    )
    add_check(
        "Snapshot positive rate",
        (
            float(targets["annual_attrition_rate"]["minimum"])
            <= snapshot_rate
            <= float(targets["annual_attrition_rate"]["maximum"])
        ),
        f"{snapshot_rate:.4%}",
        (
            f"{targets['annual_attrition_rate']['minimum']:.0%}"
            f"–{targets['annual_attrition_rate']['maximum']:.0%}"
        ),
        "This is the twelve-month forward attrition rate, not cumulative.",
    )
    add_check(
        "Snapshot positive cases",
        (
            int(targets["twelve_month_positive_cases"]["minimum"])
            <= snapshot_positive_cases
            <= int(targets["twelve_month_positive_cases"]["maximum"])
        ),
        snapshot_positive_cases,
        (
            f"{targets['twelve_month_positive_cases']['minimum']}"
            f"–{targets['twelve_month_positive_cases']['maximum']}"
        ),
        "The future window contains enough minority-class examples.",
    )
    add_check(
        "Voluntary termination share",
        (
            float(targets["voluntary_share"]["minimum"])
            <= voluntary_share
            <= float(targets["voluntary_share"]["maximum"])
        ),
        f"{voluntary_share:.4%}",
        (
            f"{targets['voluntary_share']['minimum']:.0%}"
            f"–{targets['voluntary_share']['maximum']:.0%}"
        ),
        "Cause mix remains close to the configured business assumption.",
    )
    add_check(
        "Maximum numeric target correlation",
        max_numeric_correlation
        <= float(rules["maximum_feature_target_correlation"]),
        f"{max_numeric_correlation:.4f}",
        f"<= {rules['maximum_feature_target_correlation']}",
        "No single numeric diagnostic feature nearly determines attrition.",
    )
    add_check(
        "Maximum category rate ratio",
        max_category_rate_ratio
        <= float(rules["maximum_single_group_rate_ratio"]),
        f"{max_category_rate_ratio:.4f}",
        f"<= {rules['maximum_single_group_rate_ratio']}",
        (
            "Only categories meeting the configured minimum sample size "
            "are included."
        ),
    )
    add_check(
        "Detectable observable signal",
        (
            float(targets["model_roc_auc"]["minimum"])
            <= diagnostic_roc_auc
            <= float(targets["model_roc_auc"]["maximum"])
        ),
        f"{diagnostic_roc_auc:.4f}",
        (
            f"{targets['model_roc_auc']['minimum']:.2f}"
            f"–{targets['model_roc_auc']['maximum']:.2f}"
        ),
        (
            "Five-fold Logistic Regression diagnostic; this is a "
            "generator check, not final model selection."
        ),
    )
    add_check(
        "PR-AUC above no-skill baseline",
        diagnostic_pr_auc > no_skill_pr_auc,
        f"{diagnostic_pr_auc:.4f}",
        f"> {no_skill_pr_auc:.4f}",
        "Observable features must rank risk better than random ordering.",
    )
    add_check(
        "Snapshot feature cutoffs",
        source_cutoffs["latest_date_used"].le(
            source_cutoffs["required_cutoff"]
        ).all(),
        source_cutoffs["latest_date_used"].max().date(),
        f"<= {snapshot_date.date()}",
        "Every diagnostic feature uses information available by snapshot.",
    )
    add_check(
        "Monthly probability bounds",
        bool(
            monthly_diagnostics[
                "average_combined_probability"
            ].between(
                0.0,
                float(
                    hazard_config["probability_limits"][
                        "maximum_combined_probability"
                    ]
                ),
            ).all()
        ),
        (
            f"{monthly_diagnostics['average_combined_probability'].max():.4%}"
        ),
        (
            "<= "
            f"{hazard_config['probability_limits']['maximum_combined_probability']:.2%}"
        ),
        "Average monthly exit probability remains inside configured bounds.",
    )
    add_check(
        "Outcome coverage",
        (
            len(outcomes) == len(employees)
            and outcomes["employee_id"].is_unique
        ),
        len(outcomes),
        "10,000 unique employees",
        "The simulator produces one final outcome for every employee.",
    )

    validation_summary = pd.DataFrame(validation_records)

    version1_workforce = version1["workforce"]
    version1_snapshot = version1["modeling_snapshot"]
    version_comparison = pd.DataFrame(
        [
            {
                "metric": "Total employees",
                "version_1": version1_workforce["total_employees"],
                "version_2": len(employees),
            },
            {
                "metric": "Cumulative terminations",
                "version_1": version1_workforce["terminated_employees"],
                "version_2": termination_count,
            },
            {
                "metric": "Cumulative attrition rate",
                "version_1": version1_workforce[
                    "cumulative_attrition_rate"
                ],
                "version_2": cumulative_rate,
            },
            {
                "metric": "Voluntary share",
                "version_1": (
                    version1_workforce["voluntary_terminations"]
                    / version1_workforce["terminated_employees"]
                ),
                "version_2": voluntary_share,
            },
            {
                "metric": "Leadership terminations",
                "version_1": 0,
                "version_2": team_manager_terminations,
            },
            {
                "metric": "Snapshot eligible employees",
                "version_1": version1_snapshot["eligible_employees"],
                "version_2": len(snapshot),
            },
            {
                "metric": "Snapshot positive cases",
                "version_1": version1_snapshot["positive_cases"],
                "version_2": snapshot_positive_cases,
            },
            {
                "metric": "Snapshot positive rate",
                "version_1": version1_snapshot["positive_rate"],
                "version_2": snapshot_rate,
            },
            {
                "metric": "Manager reassignments after exit",
                "version_1": 0,
                "version_2": manager_reassignments,
            },
        ]
    )
    version_comparison["absolute_change"] = (
        version_comparison["version_2"]
        - version_comparison["version_1"]
    )
    version_comparison["relative_change"] = np.where(
        version_comparison["version_1"].ne(0),
        version_comparison["absolute_change"]
        / version_comparison["version_1"].abs(),
        np.nan,
    )

    snapshot_summary = pd.DataFrame(
        {
            "metric": [
                "Snapshot date",
                "Prediction end date",
                "Eligible employees",
                "Positive cases",
                "Negative cases",
                "Positive rate",
            ],
            "value": [
                snapshot_date.date(),
                prediction_end_date.date(),
                len(snapshot),
                snapshot_positive_cases,
                len(snapshot) - snapshot_positive_cases,
                snapshot_rate,
            ],
        }
    )

    row_counts = pd.DataFrame(
        {
            "table": [
                "employees",
                "compensation_history",
                "performance_reviews",
                "training_records",
                "employee_events",
            ],
            "row_count": [
                len(employees),
                len(compensation),
                len(performance),
                len(training),
                len(events),
            ],
        }
    )

    fingerprints = table_hashes(
        [
            RAW_DATA_DIR / "employees.csv",
            RAW_DATA_DIR / "compensation_history.csv",
            RAW_DATA_DIR / "performance_reviews.csv",
            RAW_DATA_DIR / "training_records.csv",
            RAW_DATA_DIR / "employee_events.csv",
            PROJECT_ROOT
            / "data"
            / "processed"
            / "attrition_hazard_outcomes.csv",
            INTERIM_DATA_DIR
            / "attrition_hazard_monthly_diagnostics.csv",
        ]
    )

    outputs = {
        "validation_summary.csv": validation_summary,
        "v1_v2_comparison.csv": version_comparison,
        "snapshot_summary.csv": snapshot_summary,
        "snapshot_feature_table.csv": snapshot,
        "feature_source_cutoffs.csv": source_cutoffs,
        "numeric_signal_checks.csv": numeric_signals,
        "categorical_signal_checks.csv": categorical_signals,
        "diagnostic_model_folds.csv": diagnostic_model_folds,
        "diagnostic_model_summary.csv": diagnostic_model_summary,
        "attrition_by_organizational_level.csv": organizational_summary,
        "attrition_by_department.csv": department_summary,
        "attrition_by_location.csv": location_summary,
        "monthly_hazard_summary.csv": monthly_diagnostics,
        "output_row_counts.csv": row_counts,
        "reproducibility_hashes.csv": fingerprints,
    }

    for filename, table in outputs.items():
        table.to_csv(
            OUTPUT_DIR / filename,
            index=False,
            date_format="%Y-%m-%d",
        )

    print("\nVERSION 2 VALIDATION SUMMARY")
    print(validation_summary.to_string(index=False))

    print("\nVERSION 1 VERSUS VERSION 2")
    print(version_comparison.to_string(index=False))

    print("\nSTRONGEST NUMERIC SIGNALS")
    print(
        numeric_signals.head(10).to_string(
            index=False
        )
    )

    print("\nLARGEST CHECKED CATEGORY RATE RATIOS")
    print(
        checked_categories.sort_values(
            "rate_ratio_vs_baseline",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nDIAGNOSTIC MODEL")
    print(diagnostic_model_summary.to_string(index=False))

    failed_checks = validation_summary.loc[
        validation_summary["status"].eq("FAIL")
    ]

    print(f"\nValidation outputs saved to: {OUTPUT_DIR}")

    if not failed_checks.empty:
        raise ValueError(
            "Version 2 validation failed: "
            + ", ".join(failed_checks["check"])
        )

    print("\nVERSION 2 DATA VALIDATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
