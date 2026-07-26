"""Create leakage-aware Version 2 model split assignments.

The primary split is chronological:

* 2023 snapshot: train
* 2024 snapshot: validation
* 2025 snapshot: reserved test

A separate employee-grouped cross-validation assignment uses development
rows only. The reserved test target is never read for summaries, fold
creation, or validation decisions in this checkpoint.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedGroupKFold

from retention_feature_policy import load_feature_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "retention_multi_snapshot.csv"
STRATEGY_PATH = (
    PROJECT_ROOT / "config" / "model_validation_strategy.yaml"
)
TEMPORAL_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "temporal_snapshots.yaml"
)

ASSIGNMENT_PATH = PROCESSED_DIR / "model_split_assignments.csv"
PRIMARY_SUMMARY_PATH = PROCESSED_DIR / "model_split_summary.csv"
GROUP_SUMMARY_PATH = PROCESSED_DIR / "group_cv_summary.csv"
VALIDATION_PATH = PROCESSED_DIR / "model_split_validation.csv"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")

    return loaded


def load_inputs() -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Load the temporal panel and the three governing policies."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing retention_multi_snapshot.csv. "
            "Run Checkpoint 38 first."
        )

    dataset = pd.read_csv(DATA_PATH)
    strategy = load_yaml(STRATEGY_PATH)
    temporal_policy = load_yaml(TEMPORAL_CONFIG_PATH)
    feature_policy = load_feature_policy()

    return dataset, strategy, temporal_policy, feature_policy


def add_check(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    """Append one validation result."""

    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def configured_split_dates(
    strategy: dict[str, Any],
) -> dict[str, str]:
    """Return split names and snapshot dates in configured order."""

    split_configuration = strategy["primary_temporal_split"]
    return {
        split: str(split_configuration[split]["snapshot_date"])
        for split in strategy["split_order"]
    }


def temporal_window_lookup(
    temporal_policy: dict[str, Any],
) -> dict[str, str]:
    """Map every historical snapshot to its prediction-window end."""

    return {
        str(item["snapshot_date"]): str(item["prediction_end_date"])
        for item in temporal_policy["historical_snapshots"]
    }


def create_primary_assignments(
    dataset: pd.DataFrame,
    strategy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> pd.DataFrame:
    """Filter the model population and assign chronological splits."""

    split_dates = configured_split_dates(strategy)
    date_to_split = {
        snapshot_date: split
        for split, snapshot_date in split_dates.items()
    }
    eligible_levels = list(
        feature_policy["model_population"][
            "eligible_organizational_levels"
        ]
    )

    snapshot_dates = dataset["snapshot_date"].astype(str)
    population = dataset.loc[
        dataset["organizational_level"].isin(eligible_levels)
        & snapshot_dates.isin(list(date_to_split))
    ].copy()
    population["snapshot_date"] = (
        population["snapshot_date"].astype(str)
    )
    population["prediction_end_date"] = (
        population["prediction_end_date"].astype(str)
    )
    population["primary_split"] = population[
        "snapshot_date"
    ].map(date_to_split)
    target_column = str(strategy["target_column"])
    test_mask = population["primary_split"].eq("test")
    development_target = population.loc[
        ~test_mask,
        target_column,
    ].copy()
    population[target_column] = pd.Series(
        pd.NA,
        index=population.index,
        dtype="Int64",
    )
    population.loc[
        ~test_mask,
        target_column,
    ] = development_target.astype(int)
    population["group_cv_fold"] = pd.Series(
        pd.NA,
        index=population.index,
        dtype="Int64",
    )

    return population


def assign_grouped_folds(
    population: pd.DataFrame,
    strategy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign employee-disjoint folds to development rows only."""

    grouped_policy = strategy["grouped_robustness"]
    development_splits = list(grouped_policy["development_splits"])
    target_column = str(strategy["target_column"])
    group_column = str(grouped_policy["group_column"])

    development = population.loc[
        population["primary_split"].isin(development_splits)
    ].copy()
    target = development[target_column].astype(int).to_numpy()
    groups = development[group_column].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=int(grouped_policy["folds"]),
        shuffle=bool(grouped_policy["shuffle"]),
        random_state=int(grouped_policy["random_seed"]),
    )
    placeholder_features = np.zeros((len(development), 1))

    for fold_number, (_, validation_positions) in enumerate(
        splitter.split(
            placeholder_features,
            target,
            groups,
        ),
        start=1,
    ):
        validation_index = development.iloc[
            validation_positions
        ].index
        population.loc[
            validation_index,
            "group_cv_fold",
        ] = fold_number

    population["group_cv_fold"] = population[
        "group_cv_fold"
    ].astype("Int64")

    development = population.loc[
        population["primary_split"].isin(development_splits)
    ].copy()
    fold_records: list[dict[str, Any]] = []

    for fold_number in range(1, int(grouped_policy["folds"]) + 1):
        validation_rows = development.loc[
            development["group_cv_fold"].eq(fold_number)
        ]
        training_rows = development.loc[
            development["group_cv_fold"].ne(fold_number)
        ]
        training_employees = set(training_rows[group_column])
        validation_employees = set(validation_rows[group_column])

        fold_records.append(
            {
                "fold": fold_number,
                "training_rows": len(training_rows),
                "validation_rows": len(validation_rows),
                "training_employees": len(training_employees),
                "validation_employees": len(validation_employees),
                "training_positive_cases": int(
                    training_rows[target_column].astype(int).sum()
                ),
                "validation_positive_cases": int(
                    validation_rows[target_column].astype(int).sum()
                ),
                "training_positive_rate": float(
                    training_rows[target_column].astype(int).mean()
                ),
                "validation_positive_rate": float(
                    validation_rows[target_column].astype(int).mean()
                ),
                "employee_overlap": len(
                    training_employees & validation_employees
                ),
            }
        )

    return population, pd.DataFrame(fold_records)


def create_primary_summary(
    population: pd.DataFrame,
    strategy: dict[str, Any],
) -> pd.DataFrame:
    """Summarize development targets without reading the test target."""

    target_column = str(strategy["target_column"])
    test_split = "test"
    records: list[dict[str, Any]] = []

    for split in strategy["split_order"]:
        split_rows = population.loc[
            population["primary_split"].eq(split)
        ]
        record: dict[str, Any] = {
            "primary_split": split,
            "snapshot_date": split_rows["snapshot_date"].iloc[0],
            "prediction_end_date": (
                split_rows["prediction_end_date"].iloc[0]
            ),
            "rows": len(split_rows),
            "unique_employees": split_rows["employee_id"].nunique(),
        }

        if split == test_split:
            record.update(
                {
                    "target_status": "RESERVED_NOT_ACCESSED",
                    "positive_cases": "RESERVED",
                    "positive_rate": "RESERVED",
                }
            )
        else:
            target = split_rows[target_column].astype(int)
            record.update(
                {
                    "target_status": "AVAILABLE_FOR_DEVELOPMENT",
                    "positive_cases": str(int(target.sum())),
                    "positive_rate": f"{target.mean():.6f}",
                }
            )

        records.append(record)

    return pd.DataFrame(records)


def create_assignment_output(
    population: pd.DataFrame,
    strategy: dict[str, Any],
) -> pd.DataFrame:
    """Create the target-free split assignment artifact."""

    split_rank = {
        split: index
        for index, split in enumerate(strategy["split_order"], start=1)
    }
    assignments = pd.DataFrame(
        population.loc[
            :,
            [
                "employee_id",
                "snapshot_date",
                "prediction_end_date",
                "primary_split",
                "group_cv_fold",
            ],
        ].copy()
    )
    primary_split = pd.Series(
        assignments["primary_split"],
        index=assignments.index,
    )
    assignments["split_sequence"] = primary_split.map(split_rank)
    assignments["reserved_test"] = primary_split.eq("test")
    assignments = pd.DataFrame(
        assignments.sort_values(
            by=["split_sequence", "employee_id"]
        ).reset_index(drop=True)
    )

    return pd.DataFrame(
        assignments.loc[
            :,
            [
                "employee_id",
                "snapshot_date",
                "prediction_end_date",
                "split_sequence",
                "primary_split",
                "group_cv_fold",
                "reserved_test",
            ],
        ]
    )


def validate_splits(
    population: pd.DataFrame,
    assignments: pd.DataFrame,
    primary_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    strategy: dict[str, Any],
    temporal_policy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> pd.DataFrame:
    """Validate temporal, target-access, and group-separation rules."""

    checks: list[dict[str, Any]] = []
    split_dates = configured_split_dates(strategy)
    split_order = list(strategy["split_order"])
    target_column = str(strategy["target_column"])
    validation_policy = strategy["validation"]
    grouped_policy = strategy["grouped_robustness"]
    development_splits = list(grouped_policy["development_splits"])
    test_split = "test"

    expected_development_dates = {
        split_dates[split] for split in development_splits
    }
    feature_development_dates = set(
        feature_policy["development_snapshots"]
    )
    add_check(
        checks,
        "Feature policy development periods align",
        expected_development_dates == feature_development_dates,
        sorted(feature_development_dates),
        str(sorted(expected_development_dates)),
        "Feature selection and model development use the same periods.",
    )

    feature_holdout = str(
        feature_policy["reserved_holdout_snapshot"]
    )
    add_check(
        checks,
        "Feature-policy holdout aligns with test",
        feature_holdout == split_dates[test_split],
        feature_holdout,
        split_dates[test_split],
        "One snapshot must remain reserved across both policies.",
    )

    observed_split_dates = dict(
        zip(
            primary_summary["primary_split"],
            primary_summary["snapshot_date"],
            strict=True,
        )
    )
    add_check(
        checks,
        "Every snapshot maps to one primary split",
        observed_split_dates == split_dates,
        observed_split_dates,
        str(split_dates),
        "Train, validation, and test each use one configured date.",
    )

    ordered_dates = [
        date.fromisoformat(split_dates[split]) for split in split_order
    ]
    chronological = all(
        earlier < later
        for earlier, later in zip(
            ordered_dates,
            ordered_dates[1:],
        )
    )
    add_check(
        checks,
        "Primary splits are chronological",
        chronological,
        [split_date.isoformat() for split_date in ordered_dates],
        "Strictly increasing dates",
        "The model learns from the past and is evaluated in the future.",
    )

    window_lookup = temporal_window_lookup(temporal_policy)
    window_gaps: list[int] = []
    for current_split, next_split in zip(
        split_order,
        split_order[1:],
    ):
        current_end = date.fromisoformat(
            window_lookup[split_dates[current_split]]
        )
        next_start = date.fromisoformat(split_dates[next_split])
        window_gaps.append(int((next_start - current_end).days))

    add_check(
        checks,
        "Prediction windows do not overlap",
        all(gap >= 0 for gap in window_gaps),
        window_gaps,
        "All gaps >= 0 days",
        "A termination cannot be counted in two primary periods.",
    )

    duplicate_keys = int(
        assignments.duplicated(
            ["employee_id", "snapshot_date"]
        ).sum()
    )
    add_check(
        checks,
        "Employee-snapshot assignment keys are unique",
        duplicate_keys == 0,
        duplicate_keys,
        "0 duplicates",
        "Every temporal row receives exactly one assignment.",
    )

    unassigned_rows = int(assignments["primary_split"].isna().sum())
    add_check(
        checks,
        "All eligible rows receive a primary split",
        unassigned_rows == 0,
        unassigned_rows,
        "0 unassigned rows",
        "No eligible temporal row is silently omitted.",
    )

    minimum_rows = int(
        validation_policy["minimum_rows_per_primary_split"]
    )
    minimum_observed_rows = int(primary_summary["rows"].min())
    add_check(
        checks,
        "Primary splits have sufficient rows",
        minimum_observed_rows >= minimum_rows,
        minimum_observed_rows,
        f">= {minimum_rows}",
        "Each chronological period must support model evaluation.",
    )

    development = population.loc[
        population["primary_split"].isin(development_splits)
    ]
    development_target = development[target_column]
    development_target_valid = (
        not development_target.isna().any()
        and set(development_target.astype(int)).issubset({0, 1})
    )
    add_check(
        checks,
        "Development targets are complete and binary",
        development_target_valid,
        sorted(development_target.dropna().astype(int).unique()),
        "[0, 1] with no missing values",
        "Only train and validation targets are inspected here.",
    )

    minimum_positive_cases = int(
        validation_policy["minimum_development_positive_cases"]
    )
    development_positive_counts = (
        development.groupby("primary_split")[target_column]
        .sum()
        .astype(int)
        .to_dict()
    )
    add_check(
        checks,
        "Development periods have enough positive cases",
        all(
            count >= minimum_positive_cases
            for count in development_positive_counts.values()
        ),
        development_positive_counts,
        f"Each >= {minimum_positive_cases}",
        "Train and validation need enough attrition examples.",
    )

    test_assignments = assignments.loc[
        assignments["primary_split"].eq(test_split)
    ]
    test_target_used = int(
        population.loc[
            population["primary_split"].eq(test_split),
            target_column,
        ].notna().sum()
    )
    test_fold_values = int(
        test_assignments["group_cv_fold"].notna().sum()
    )
    target_in_assignments = target_column in assignments.columns
    test_policy_passed = (
        test_target_used == 0
        and test_fold_values == 0
        and not target_in_assignments
    )
    add_check(
        checks,
        "Reserved test remains outside development",
        test_policy_passed,
        (
            f"{test_target_used} target values used; "
            f"{test_fold_values} fold assignments"
        ),
        "0 target values and 0 group-fold assignments",
        "Test targets are masked in memory and excluded from assignments.",
    )

    missing_development_folds = int(
        assignments.loc[
            assignments["primary_split"].isin(development_splits),
            "group_cv_fold",
        ].isna().sum()
    )
    add_check(
        checks,
        "Every development row receives a group fold",
        missing_development_folds == 0,
        missing_development_folds,
        "0 missing folds",
        "All train and validation rows support grouped robustness checks.",
    )

    fold_counts_per_employee = (
        assignments.loc[
            assignments["primary_split"].isin(development_splits)
        ]
        .groupby("employee_id")["group_cv_fold"]
        .nunique()
    )
    multi_fold_employees = int(
        fold_counts_per_employee.gt(1).sum()
    )
    add_check(
        checks,
        "Each development employee belongs to one fold",
        multi_fold_employees == 0,
        multi_fold_employees,
        "0 employees in multiple folds",
        "Repeated yearly rows stay together during grouped validation.",
    )

    maximum_overlap = int(fold_summary["employee_overlap"].max())
    add_check(
        checks,
        "Grouped folds are employee-disjoint",
        maximum_overlap == 0,
        maximum_overlap,
        "0 overlapping employees",
        "A fold never trains and validates on the same employee.",
    )

    expected_folds = int(grouped_policy["folds"])
    observed_folds = sorted(fold_summary["fold"].astype(int).tolist())
    add_check(
        checks,
        "Expected number of grouped folds exists",
        observed_folds == list(range(1, expected_folds + 1)),
        observed_folds,
        str(list(range(1, expected_folds + 1))),
        "Every configured robustness fold must be populated.",
    )

    minimum_fold_positives = int(
        validation_policy[
            "minimum_group_fold_validation_positive_cases"
        ]
    )
    observed_minimum_fold_positives = int(
        fold_summary["validation_positive_cases"].min()
    )
    add_check(
        checks,
        "Grouped folds have sufficient positive cases",
        observed_minimum_fold_positives >= minimum_fold_positives,
        observed_minimum_fold_positives,
        f">= {minimum_fold_positives}",
        "Every fold must contain enough attrition cases.",
    )

    positive_rate_spread = float(
        fold_summary["validation_positive_rate"].max()
        - fold_summary["validation_positive_rate"].min()
    )
    maximum_rate_spread = float(
        validation_policy[
            "maximum_group_fold_positive_rate_spread"
        ]
    )
    add_check(
        checks,
        "Grouped-fold positive rates are balanced",
        positive_rate_spread <= maximum_rate_spread,
        round(positive_rate_spread, 6),
        f"<= {maximum_rate_spread}",
        "Stratification keeps fold outcome rates reasonably similar.",
    )

    validation = pd.DataFrame(checks)
    failures = validation.loc[validation["status"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(
            "Model split validation failed:\n"
            + failures.to_string(index=False)
        )

    return validation


def save_outputs(
    assignments: pd.DataFrame,
    primary_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Save the four generated Checkpoint 40 artifacts."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(ASSIGNMENT_PATH, index=False)
    primary_summary.to_csv(PRIMARY_SUMMARY_PATH, index=False)
    fold_summary.to_csv(GROUP_SUMMARY_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)


def print_results(
    assignments: pd.DataFrame,
    primary_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Print beginner-readable Checkpoint 40 results."""

    print("\nPRIMARY TEMPORAL SPLITS")
    print(primary_summary.to_string(index=False))

    print("\nEMPLOYEE-GROUPED ROBUSTNESS FOLDS")
    display_folds = fold_summary.copy()
    display_folds["training_positive_rate"] = (
        display_folds["training_positive_rate"]
        .map(lambda value: f"{value:.2%}")
    )
    display_folds["validation_positive_rate"] = (
        display_folds["validation_positive_rate"]
        .map(lambda value: f"{value:.2%}")
    )
    print(display_folds.to_string(index=False))

    print("\nSPLIT-INTEGRITY VALIDATION")
    print(validation.to_string(index=False))

    print("\nASSIGNMENT SUMMARY")
    print(f"Total assigned rows: {len(assignments):,}")
    print(
        "Development rows with grouped folds: "
        f"{assignments['group_cv_fold'].notna().sum():,}"
    )
    print(
        "Reserved test rows without grouped folds: "
        f"{assignments['reserved_test'].sum():,}"
    )
    print(f"Saved assignments: {ASSIGNMENT_PATH}")
    print(
        "\nMODEL SPLIT ASSIGNMENTS COMPLETED SUCCESSFULLY"
    )


def main() -> None:
    """Run the Checkpoint 40 model-split workflow."""

    (
        dataset,
        strategy,
        temporal_policy,
        feature_policy,
    ) = load_inputs()
    population = create_primary_assignments(
        dataset,
        strategy,
        feature_policy,
    )
    population, fold_summary = assign_grouped_folds(
        population,
        strategy,
    )
    primary_summary = create_primary_summary(
        population,
        strategy,
    )
    assignments = create_assignment_output(
        population,
        strategy,
    )
    validation = validate_splits(
        population,
        assignments,
        primary_summary,
        fold_summary,
        strategy,
        temporal_policy,
        feature_policy,
    )
    save_outputs(
        assignments,
        primary_summary,
        fold_summary,
        validation,
    )
    print_results(
        assignments,
        primary_summary,
        fold_summary,
        validation,
    )


if __name__ == "__main__":
    main()
