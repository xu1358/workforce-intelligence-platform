"""Run an isolated benchmark on the fictional IBM attrition dataset."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from download_ibm_attrition_benchmark import (
    file_sha256,
    load_config,
    validate_source_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ibm_benchmark"
FIGURE_DIR = PROCESSED_DIR / "figures"

PROFILE_PATH = PROCESSED_DIR / "ibm_benchmark_data_profile.csv"
METRIC_PATH = PROCESSED_DIR / "ibm_benchmark_metrics.csv"
FOLD_PATH = PROCESSED_DIR / "ibm_benchmark_fold_summary.csv"
BOOTSTRAP_PATH = PROCESSED_DIR / "ibm_benchmark_bootstrap.csv"
INTERVAL_PATH = PROCESSED_DIR / "ibm_benchmark_intervals.csv"
FEATURE_PATH = PROCESSED_DIR / "ibm_benchmark_feature_stability.csv"
SUBGROUP_PATH = PROCESSED_DIR / "ibm_benchmark_subgroup_metrics.csv"
COMPARISON_PATH = PROCESSED_DIR / "ibm_primary_context_comparison.csv"
MAPPING_PATH = PROCESSED_DIR / "ibm_feature_mapping.csv"
VALIDATION_PATH = PROCESSED_DIR / "ibm_benchmark_validation.csv"


def prepare_benchmark_data(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return configured predictors, binary target, and stable identifiers."""

    feature_policy = config["feature_policy"]
    features = list(feature_policy["numerical_features"]) + list(
        feature_policy["categorical_features"]
    )
    target_column = str(config["schema"]["target_column"])
    positive_label = str(config["schema"]["positive_label"])
    identifier_column = str(config["schema"]["identifier_column"])

    predictors = frame[features].copy()
    target = frame[target_column].eq(positive_label).astype(int).to_numpy()
    identifiers = frame[identifier_column].to_numpy()

    return predictors, target, identifiers


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """Build the committed numeric and categorical preprocessing policy."""

    policy = config["feature_policy"]
    numerical = list(policy["numerical_features"])
    categorical = list(policy["categorical_features"])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def build_base_model(config: dict[str, Any]) -> Pipeline:
    """Build the fixed balanced Logistic Regression pipeline."""

    parameters = dict(config["model"]["parameters"])

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("classifier", LogisticRegression(**parameters)),
        ]
    )


def build_calibrated_model(config: dict[str, Any]) -> CalibratedClassifierCV:
    """Wrap the fixed model in leakage-safe inner-fold calibration."""

    return CalibratedClassifierCV(
        estimator=build_base_model(config),
        method=str(config["model"]["calibration_method"]),
        cv=int(config["model"]["inner_calibration_folds"]),
    )


def stable_top_fraction_mask(
    probabilities: np.ndarray,
    tie_breaker: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Select a stable top fraction, breaking probability ties explicitly."""

    count = int(math.ceil(len(probabilities) * fraction))
    order = np.lexsort((tie_breaker, -probabilities))
    selected = np.zeros(len(probabilities), dtype=bool)
    selected[order[:count]] = True
    return selected


def calculate_metrics(
    method: str,
    target: np.ndarray,
    probabilities: np.ndarray,
    tie_breaker: np.ndarray,
    top_fraction: float,
) -> dict[str, Any]:
    """Calculate probability and ranking metrics for one method."""

    selected = stable_top_fraction_mask(
        probabilities,
        tie_breaker,
        top_fraction,
    )
    positives = int(target.sum())
    selected_positives = int(target[selected].sum())
    prevalence = float(target.mean())
    precision = float(target[selected].mean())

    return {
        "method": method,
        "rows": len(target),
        "positive_cases": positives,
        "positive_rate": prevalence,
        "mean_predicted_probability": float(probabilities.mean()),
        "calibration_gap": float(probabilities.mean() - prevalence),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "pr_auc_lift_over_baseline": float(
            average_precision_score(target, probabilities) / prevalence
        ),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities)),
        "top_fraction": top_fraction,
        "selected_count": int(selected.sum()),
        "selected_positive_cases": selected_positives,
        "top_fraction_precision": precision,
        "top_fraction_capture": float(selected_positives / positives),
        "top_fraction_lift": float(precision / prevalence),
    }


def clean_feature_name(name: str) -> str:
    """Convert a scikit-learn encoded name into a readable label."""

    return name.replace("numeric__", "").replace("categorical__", "").replace("_", " ")


def evaluate_repeated_cross_validation(
    predictors: pd.DataFrame,
    target: np.ndarray,
    identifiers: np.ndarray,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """Generate repeated out-of-fold scores and coefficient diagnostics."""

    evaluation = config["evaluation"]
    folds = int(evaluation["outer_folds"])
    repeats = int(evaluation["repeats"])
    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=int(evaluation["random_seed"]),
    )

    uncalibrated = np.full((len(target), repeats), np.nan)
    calibrated = np.full((len(target), repeats), np.nan)
    coverage = np.zeros(len(target), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[pd.Series] = []

    for split_index, (train_index, test_index) in enumerate(
        splitter.split(predictors, target),
        start=1,
    ):
        repeat = ((split_index - 1) // folds) + 1
        fold = ((split_index - 1) % folds) + 1

        base_model = build_base_model(config)
        base_model.fit(
            predictors.iloc[train_index],
            target[train_index],
        )
        uncalibrated[test_index, repeat - 1] = base_model.predict_proba(
            predictors.iloc[test_index]
        )[:, 1]

        encoded_names = base_model.named_steps["preprocessor"].get_feature_names_out()
        coefficients = base_model.named_steps["classifier"].coef_[0]
        coefficient_rows.append(
            pd.Series(
                coefficients,
                index=[clean_feature_name(name) for name in encoded_names],
                name=split_index,
            )
        )

        calibrated_model = build_calibrated_model(config)
        calibrated_model.fit(
            predictors.iloc[train_index],
            target[train_index],
        )
        fold_probability = calibrated_model.predict_proba(predictors.iloc[test_index])[
            :, 1
        ]
        calibrated[test_index, repeat - 1] = fold_probability
        coverage[test_index] += 1

        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "training_rows": len(train_index),
                "validation_rows": len(test_index),
                "training_positive_cases": int(target[train_index].sum()),
                "validation_positive_cases": int(target[test_index].sum()),
                "employee_overlap": int(len(set(train_index).intersection(test_index))),
                "pr_auc": float(
                    average_precision_score(
                        target[test_index],
                        fold_probability,
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(
                        target[test_index],
                        fold_probability,
                    )
                ),
                "brier_score": float(
                    brier_score_loss(
                        target[test_index],
                        fold_probability,
                    )
                ),
            }
        )

        if split_index % 10 == 0:
            print(f"Completed IBM benchmark fold: {split_index}/{folds * repeats}")

    uncalibrated_probability = np.nanmean(uncalibrated, axis=1)
    calibrated_probability = np.nanmean(calibrated, axis=1)
    top_fraction = float(evaluation["top_fraction"])
    metric_table = pd.DataFrame(
        [
            calculate_metrics(
                "Uncalibrated Logistic Regression",
                target,
                uncalibrated_probability,
                identifiers,
                top_fraction,
            ),
            calculate_metrics(
                "Sigmoid-calibrated Logistic Regression",
                target,
                calibrated_probability,
                identifiers,
                top_fraction,
            ),
        ]
    )

    coefficient_matrix = pd.DataFrame(coefficient_rows)
    feature_table = coefficient_stability(coefficient_matrix)
    fold_table = pd.DataFrame(fold_rows)
    coverage_table = pd.DataFrame(
        {
            "minimum_oof_coverage": [int(coverage.min())],
            "maximum_oof_coverage": [int(coverage.max())],
            "expected_oof_coverage": [repeats],
        }
    )

    return (
        metric_table,
        fold_table,
        feature_table,
        uncalibrated_probability,
        calibrated_probability,
        coverage_table,
    )


def coefficient_stability(matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarize coefficient direction across outer training folds."""

    rows: list[dict[str, Any]] = []

    for feature in matrix.columns:
        values = matrix[feature].dropna().to_numpy()
        positive_share = float((values > 0).mean())
        negative_share = float((values < 0).mean())
        median = float(np.median(values))

        rows.append(
            {
                "encoded_feature": feature,
                "fold_estimates": len(values),
                "coefficient_median": median,
                "coefficient_lower_95": float(np.quantile(values, 0.025)),
                "coefficient_upper_95": float(np.quantile(values, 0.975)),
                "sign_stability": max(positive_share, negative_share),
                "direction": (
                    "Higher modeled attrition association"
                    if median > 0
                    else "Lower modeled attrition association"
                ),
                "absolute_median": abs(median),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["absolute_median", "encoded_feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def bootstrap_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap calibrated ranking metrics with stratified resampling."""

    evaluation = config["evaluation"]
    iterations = int(evaluation["bootstrap_iterations"])
    confidence = float(evaluation["confidence_level"])
    top_fraction = float(evaluation["top_fraction"])
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    positive_index = np.flatnonzero(target == 1)
    negative_index = np.flatnonzero(target == 0)
    rows: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        sampled_positive = rng.choice(
            positive_index,
            size=len(positive_index),
            replace=True,
        )
        sampled_negative = rng.choice(
            negative_index,
            size=len(negative_index),
            replace=True,
        )
        sampled = np.concatenate([sampled_positive, sampled_negative])
        rng.shuffle(sampled)

        metrics = calculate_metrics(
            "Sigmoid-calibrated Logistic Regression",
            target[sampled],
            probability[sampled],
            np.arange(len(sampled)),
            top_fraction,
        )
        metrics["iteration"] = iteration
        rows.append(metrics)

    bootstrap = pd.DataFrame(rows)
    alpha = (1.0 - confidence) / 2.0
    metric_columns = [
        "pr_auc",
        "roc_auc",
        "brier_score",
        "top_fraction_precision",
        "top_fraction_capture",
        "top_fraction_lift",
    ]
    intervals = pd.DataFrame(
        [
            {
                "metric": metric,
                "lower_95": float(bootstrap[metric].quantile(alpha)),
                "median": float(bootstrap[metric].median()),
                "upper_95": float(bootstrap[metric].quantile(1.0 - alpha)),
            }
            for metric in metric_columns
        ]
    )

    return bootstrap, intervals


def add_age_band(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """Create configured age bands for descriptive subgroup analysis."""

    result = pd.Series(index=frame.index, dtype="object")

    for band in config["subgroups"]["age_bands"]:
        mask = frame["Age"].between(
            int(band["minimum"]),
            int(band["maximum"]),
            inclusive="both",
        )
        result.loc[mask] = str(band["label"])

    if result.isna().any():
        raise ValueError("At least one IBM age falls outside configured bands.")

    return result


def subgroup_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
    identifiers: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate descriptive performance measures by configured subgroup."""

    working = frame.copy()
    working["AgeBand"] = add_age_band(frame, config)
    working["_target"] = target
    working["_probability"] = probability
    working["_selected"] = stable_top_fraction_mask(
        probability,
        identifiers,
        float(config["evaluation"]["top_fraction"]),
    )
    rows: list[dict[str, Any]] = []

    for attribute in config["subgroups"]["attributes"]:
        for group, values in working.groupby(attribute, observed=True):
            group_target = values["_target"].to_numpy()
            group_probability = values["_probability"].to_numpy()
            group_selected = values["_selected"].to_numpy(dtype=bool)
            positive_cases = int(group_target.sum())
            negative_cases = int(len(group_target) - positive_cases)
            selected_positive = int(group_target[group_selected].sum())
            selected_negative = int(group_selected.sum() - selected_positive)

            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "sample_size": len(values),
                    "positive_cases": positive_cases,
                    "observed_attrition_rate": float(group_target.mean()),
                    "mean_predicted_probability": float(group_probability.mean()),
                    "calibration_gap": float(
                        group_probability.mean() - group_target.mean()
                    ),
                    "selected_count": int(group_selected.sum()),
                    "selection_rate": float(group_selected.mean()),
                    "true_positive_rate": (
                        float(selected_positive / positive_cases)
                        if positive_cases
                        else np.nan
                    ),
                    "false_positive_rate": (
                        float(selected_negative / negative_cases)
                        if negative_cases
                        else np.nan
                    ),
                    "precision": (
                        float(selected_positive / group_selected.sum())
                        if group_selected.sum()
                        else np.nan
                    ),
                    "brier_score": float(
                        brier_score_loss(
                            group_target,
                            group_probability,
                        )
                    ),
                    "eligible_for_descriptive_comparison": bool(
                        len(values) >= 50 and positive_cases >= 10
                    ),
                }
            )

    return pd.DataFrame(rows)


def data_profile(
    frame: pd.DataFrame,
    target: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create a compact profile of the external benchmark."""

    policy = config["feature_policy"]

    return pd.DataFrame(
        [
            {"metric": "Rows", "value": len(frame)},
            {"metric": "Source columns", "value": len(frame.columns)},
            {"metric": "Positive cases", "value": int(target.sum())},
            {
                "metric": "Positive rate",
                "value": float(target.mean()),
            },
            {
                "metric": "Missing cells",
                "value": int(frame.isna().sum().sum()),
            },
            {
                "metric": "Selected raw model features",
                "value": (
                    len(policy["numerical_features"])
                    + len(policy["categorical_features"])
                ),
            },
            {
                "metric": "Excluded source features",
                "value": len(policy["excluded_features"]),
            },
            {
                "metric": "Evaluation folds",
                "value": (
                    int(config["evaluation"]["outer_folds"])
                    * int(config["evaluation"]["repeats"])
                ),
            },
        ]
    )


def context_comparison(
    metric_table: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compare context without claiming direct dataset transportability."""

    selected = metric_table.loc[
        metric_table["method"].eq("Sigmoid-calibrated Logistic Regression")
    ].iloc[0]
    primary = config["primary_reference"]

    rows = [
        {
            "metric": "Evaluation design",
            "primary_version_2": primary["evaluation_design"],
            "ibm_external_benchmark": config["evaluation"]["design"],
            "directly_comparable": False,
        },
        {
            "metric": "Rows",
            "primary_version_2": primary["rows"],
            "ibm_external_benchmark": int(selected["rows"]),
            "directly_comparable": False,
        },
        {
            "metric": "Positive rate",
            "primary_version_2": primary["positive_rate"],
            "ibm_external_benchmark": selected["positive_rate"],
            "directly_comparable": False,
        },
        {
            "metric": "PR-AUC",
            "primary_version_2": primary["pr_auc"],
            "ibm_external_benchmark": selected["pr_auc"],
            "directly_comparable": False,
        },
        {
            "metric": "ROC-AUC",
            "primary_version_2": primary["roc_auc"],
            "ibm_external_benchmark": selected["roc_auc"],
            "directly_comparable": False,
        },
        {
            "metric": "Brier score",
            "primary_version_2": primary["brier_score"],
            "ibm_external_benchmark": selected["brier_score"],
            "directly_comparable": False,
        },
        {
            "metric": "Top-decile lift",
            "primary_version_2": primary["top_decile_lift"],
            "ibm_external_benchmark": selected["top_fraction_lift"],
            "directly_comparable": False,
        },
        {
            "metric": "Target horizon",
            "primary_version_2": "Explicit next 12 months",
            "ibm_external_benchmark": "Not provided",
            "directly_comparable": False,
        },
    ]

    comparison = pd.DataFrame(rows)
    comparison["comparison_note"] = (
        "Context only: different synthetic populations, features, targets, "
        "and evaluation designs."
    )
    return comparison


def append_check(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    """Append one normalized validation result."""

    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def build_validation(
    frame: pd.DataFrame,
    target: np.ndarray,
    metric_table: pd.DataFrame,
    fold_table: pd.DataFrame,
    coverage_table: pd.DataFrame,
    bootstrap: pd.DataFrame,
    intervals: pd.DataFrame,
    subgroup: pd.DataFrame,
    config: dict[str, Any],
    source_path: Path,
) -> pd.DataFrame:
    """Validate source, evaluation, calibration, isolation, and governance."""

    checks: list[dict[str, Any]] = []
    schema = config["schema"]
    policy = config["feature_policy"]
    evaluation = config["evaluation"]
    selected = metric_table.loc[
        metric_table["method"].eq("Sigmoid-calibrated Logistic Regression")
    ].iloc[0]
    uncalibrated = metric_table.loc[
        metric_table["method"].eq("Uncalibrated Logistic Regression")
    ].iloc[0]

    append_check(
        checks,
        "Pinned source checksum matches",
        file_sha256(source_path) == config["source"]["sha256"],
        file_sha256(source_path),
        config["source"]["sha256"],
        "The external file cannot silently change.",
    )
    append_check(
        checks,
        "Dataset is explicitly fictional",
        bool(config["source"]["fictional_data"]),
        config["source"]["fictional_data"],
        True,
        "This is not real IBM employee information.",
    )
    append_check(
        checks,
        "Dataset license is documented",
        "ODbL" in config["source"]["license_name"]
        and "DbCL" in config["source"]["license_name"],
        config["source"]["license_name"],
        "ODbL and DbCL",
        "The external benchmark retains its source terms.",
    )
    append_check(
        checks,
        "Dataset shape matches pinned schema",
        frame.shape
        == (
            int(schema["expected_rows"]),
            int(schema["expected_columns"]),
        ),
        frame.shape,
        (
            int(schema["expected_rows"]),
            int(schema["expected_columns"]),
        ),
        "Source rows and columns must remain stable.",
    )
    append_check(
        checks,
        "Target counts match pinned schema",
        int(target.sum()) == int(schema["expected_positive_cases"]),
        int(target.sum()),
        int(schema["expected_positive_cases"]),
        "Attrition class prevalence must be reproducible.",
    )
    append_check(
        checks,
        "Source contains no missing values",
        int(frame.isna().sum().sum()) == 0,
        int(frame.isna().sum().sum()),
        0,
        "The original static dataset is complete.",
    )
    append_check(
        checks,
        "Employee identifier is unique",
        frame[schema["identifier_column"]].is_unique,
        int(frame[schema["identifier_column"]].duplicated().sum()),
        0,
        "Every external row represents one fictional employee.",
    )

    selected_features = set(policy["numerical_features"]) | set(
        policy["categorical_features"]
    )
    excluded_features = set(policy["excluded_features"])
    append_check(
        checks,
        "Selected and excluded feature policies do not overlap",
        not selected_features.intersection(excluded_features),
        sorted(selected_features.intersection(excluded_features)),
        [],
        "Identifiers, constants, opaque rates, and target stay excluded.",
    )
    append_check(
        checks,
        "Gender is excluded from model features",
        "Gender" not in selected_features and "Gender" in excluded_features,
        "Gender" in selected_features,
        False,
        "Gender is retained only for descriptive subgroup diagnostics.",
    )
    append_check(
        checks,
        "Static design is labeled without temporal claims",
        bool(evaluation["no_temporal_claim_permitted"])
        and not any("date" in column.lower() for column in frame.columns),
        evaluation["design"],
        "Repeated stratified cross-validation",
        "The external dataset has no dates or defined outcome horizon.",
    )

    expected_splits = int(evaluation["outer_folds"]) * int(evaluation["repeats"])
    append_check(
        checks,
        "Every repeated fold was evaluated",
        len(fold_table) == expected_splits,
        len(fold_table),
        expected_splits,
        "All configured out-of-fold evaluations must complete.",
    )
    append_check(
        checks,
        "Outer folds are row-disjoint",
        int(fold_table["employee_overlap"].max()) == 0,
        int(fold_table["employee_overlap"].max()),
        0,
        "No row may be fitted and scored in one outer fold.",
    )
    append_check(
        checks,
        "Every row has one score per repeat",
        (
            int(coverage_table["minimum_oof_coverage"].iloc[0])
            == int(evaluation["repeats"])
            == int(coverage_table["maximum_oof_coverage"].iloc[0])
        ),
        coverage_table.iloc[0].to_dict(),
        int(evaluation["repeats"]),
        "Averaged probabilities must be fully out of fold.",
    )

    probability_metrics_valid = bool(
        metric_table[
            [
                "mean_predicted_probability",
                "pr_auc",
                "roc_auc",
                "brier_score",
                "top_fraction_precision",
                "top_fraction_capture",
            ]
        ]
        .apply(lambda column: column.between(0.0, 1.0).all())
        .all()
    )
    append_check(
        checks,
        "All probability metrics are valid",
        probability_metrics_valid,
        probability_metrics_valid,
        True,
        "Probabilities and rate metrics must remain inside [0, 1].",
    )
    append_check(
        checks,
        "Sigmoid calibration improves Brier score",
        float(selected["brier_score"]) < float(uncalibrated["brier_score"]),
        (f"{uncalibrated['brier_score']:.6f} -> {selected['brier_score']:.6f}"),
        "Calibrated < uncalibrated",
        "Calibration should improve probability accuracy out of fold.",
    )
    append_check(
        checks,
        "PR-AUC exceeds no-skill baseline",
        float(selected["pr_auc_lift_over_baseline"])
        >= float(config["validation"]["minimum_pr_auc_lift_over_baseline"]),
        float(selected["pr_auc_lift_over_baseline"]),
        (f">={config['validation']['minimum_pr_auc_lift_over_baseline']}"),
        "External features must rank attrition better than random ordering.",
    )
    append_check(
        checks,
        "ROC-AUC exceeds minimum diagnostic value",
        float(selected["roc_auc"]) >= float(config["validation"]["minimum_roc_auc"]),
        float(selected["roc_auc"]),
        f">={config['validation']['minimum_roc_auc']}",
        "The benchmark must show nonrandom separation.",
    )
    append_check(
        checks,
        "Top-decile lift exceeds minimum diagnostic value",
        float(selected["top_fraction_lift"])
        >= float(config["validation"]["minimum_top_decile_lift"]),
        float(selected["top_fraction_lift"]),
        f">={config['validation']['minimum_top_decile_lift']}",
        "The highest score group should concentrate attrition cases.",
    )
    append_check(
        checks,
        "Bootstrap uncertainty is complete",
        len(bootstrap) == int(evaluation["bootstrap_iterations"])
        and len(intervals) == 6,
        f"{len(bootstrap)} iterations; {len(intervals)} intervals",
        (f"{evaluation['bootstrap_iterations']} iterations; 6 intervals"),
        "Reported benchmark metrics require uncertainty estimates.",
    )
    append_check(
        checks,
        "Configured subgroup diagnostics are complete",
        set(subgroup["attribute"]) == set(config["subgroups"]["attributes"]),
        sorted(subgroup["attribute"].unique()),
        sorted(config["subgroups"]["attributes"]),
        "Gender, age, and department are descriptive diagnostics only.",
    )
    append_check(
        checks,
        "Primary pipeline remains isolated",
        not any(bool(value) for value in config["isolation"].values()),
        config["isolation"],
        "All False",
        "No primary model, final test, policy, dashboard, or review list changes.",
    )

    return pd.DataFrame(checks)


def save_figures(
    target: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
    subgroup: pd.DataFrame,
) -> list[Path]:
    """Save aggregate external-benchmark figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    precision, recall, _ = precision_recall_curve(target, calibrated)
    fpr, tpr, _ = roc_curve(target, calibrated)
    prevalence = float(target.mean())

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(recall, precision, color="#2a6fbb", linewidth=2)
    axes[0].axhline(
        prevalence,
        color="#777777",
        linestyle="--",
        label=f"No skill ({prevalence:.1%})",
    )
    axes[0].set(
        xlabel="Recall",
        ylabel="Precision",
        title="IBM benchmark precision–recall",
    )
    axes[0].legend()

    axes[1].plot(fpr, tpr, color="#2a6fbb", linewidth=2)
    axes[1].plot([0, 1], [0, 1], color="#777777", linestyle="--")
    axes[1].set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="IBM benchmark ROC",
    )
    figure.tight_layout()
    path = FIGURE_DIR / "ibm_benchmark_ranking_curves.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved.append(path)

    figure = Figure(figsize=(6.5, 5))
    axis = figure.subplots()
    for label, probability, color in [
        ("Uncalibrated", uncalibrated, "#c44e52"),
        ("Sigmoid", calibrated, "#2a6fbb"),
    ]:
        observed, predicted = calibration_curve(
            target,
            probability,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed attrition rate",
        title="Repeated out-of-fold calibration",
    )
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "ibm_benchmark_calibration.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    saved.append(path)

    eligible = subgroup.loc[subgroup["eligible_for_descriptive_comparison"]].copy()
    eligible["label"] = (
        eligible["attribute"].astype(str) + ": " + eligible["group"].astype(str)
    )
    eligible = eligible.sort_values("selection_rate")
    figure = Figure(figsize=(8, 7))
    axis = figure.subplots()
    axis.barh(
        eligible["label"],
        eligible["selection_rate"],
        color="#4c78a8",
        alpha=0.85,
    )
    axis.set(
        xlabel="Top-decile selection rate",
        title="Descriptive subgroup selection rates",
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    figure.tight_layout()
    path = FIGURE_DIR / "ibm_benchmark_subgroup_selection.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    saved.append(path)

    return saved


def write_outputs(
    profile: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    intervals: pd.DataFrame,
    features: pd.DataFrame,
    subgroup: pd.DataFrame,
    comparison: pd.DataFrame,
    mapping: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Write aggregate benchmark artifacts."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(PROFILE_PATH, index=False)
    metrics.to_csv(METRIC_PATH, index=False)
    folds.to_csv(FOLD_PATH, index=False)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    intervals.to_csv(INTERVAL_PATH, index=False)
    features.to_csv(FEATURE_PATH, index=False)
    subgroup.to_csv(SUBGROUP_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    mapping.to_csv(MAPPING_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)


def main() -> None:
    """Run the isolated external benchmark and validate every boundary."""

    config = load_config()
    source_path = PROJECT_ROOT / config["source"]["local_path"]

    if not source_path.exists():
        raise FileNotFoundError(
            "IBM benchmark data is missing. Run "
            "src\\download_ibm_attrition_benchmark.py first."
        )

    frame = pd.read_csv(source_path)
    validate_source_frame(frame, config)
    predictors, target, identifiers = prepare_benchmark_data(frame, config)

    (
        metrics,
        folds,
        features,
        uncalibrated,
        calibrated,
        coverage,
    ) = evaluate_repeated_cross_validation(
        predictors,
        target,
        identifiers,
        config,
    )
    bootstrap, intervals = bootstrap_metrics(
        target,
        calibrated,
        config,
    )
    subgroup = subgroup_metrics(
        frame,
        target,
        calibrated,
        identifiers,
        config,
    )
    profile = data_profile(frame, target, config)
    comparison = context_comparison(metrics, config)
    mapping = pd.DataFrame(config["feature_mapping"])
    validation = build_validation(
        frame,
        target,
        metrics,
        folds,
        coverage,
        bootstrap,
        intervals,
        subgroup,
        config,
        source_path,
    )
    figures = save_figures(
        target,
        uncalibrated,
        calibrated,
        subgroup,
    )
    write_outputs(
        profile,
        metrics,
        folds,
        bootstrap,
        intervals,
        features,
        subgroup,
        comparison,
        mapping,
        validation,
    )

    print("\nIBM BENCHMARK DATA PROFILE")
    print(profile.to_string(index=False))

    print("\nIBM BENCHMARK MODEL METRICS")
    print(metrics.to_string(index=False))

    print("\nIBM BENCHMARK 95% INTERVALS")
    print(intervals.to_string(index=False))

    print("\nSTRONGEST STABLE IBM ASSOCIATIONS")
    print(features.head(12).to_string(index=False))

    print("\nIBM DESCRIPTIVE SUBGROUP METRICS")
    print(subgroup.to_string(index=False))

    print("\nPRIMARY V2 VERSUS IBM CONTEXT")
    print(comparison.to_string(index=False))

    print("\nIBM BENCHMARK VALIDATION")
    print(validation.to_string(index=False))

    failures = validation.loc[validation["status"].ne("PASS")]
    if not failures.empty:
        names = ", ".join(failures["check"].astype(str))
        raise ValueError(f"IBM benchmark validation failed: {names}")

    print(f"\nSaved benchmark outputs to: {PROCESSED_DIR}")
    print("Saved figures: " + ", ".join(path.name for path in figures))
    print(
        "Primary model, final test, policy, dashboard, and review list "
        "were not changed."
    )
    print("\nIBM EXTERNAL BENCHMARK COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
