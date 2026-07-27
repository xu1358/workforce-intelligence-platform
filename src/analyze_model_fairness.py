"""Audit retention-model performance across employee subgroups.

Checkpoint 44 uses only the 2024 validation population and the selected
sigmoid-calibrated Logistic Regression from Checkpoint 42. It reports
subgroup performance, calibration, descriptive disparity measures, and
an age-and-education exclusion sensitivity model. The reserved 2025 test
target remains outside every calculation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import NormalDist
from typing import Any
import os
import tempfile

import numpy as np
import pandas as pd
import yaml


MPL_CONFIG_DIR = (
    Path(tempfile.gettempdir())
    / "workforce_intelligence_matplotlib"
)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from calibrate_retention_model import (  # noqa: E402
    fit_calibration_methods,
)
from retention_feature_policy import (  # noqa: E402
    load_feature_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROCESSED_DIR / "fairness_figures"

CONFIG_PATH = PROJECT_ROOT / "config" / "retention_fairness.yaml"
MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "model_comparison_v2.yaml"
)
CALIBRATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "retention_calibration.yaml"
)
DATA_PATH = PROCESSED_DIR / "retention_multi_snapshot.csv"
ASSIGNMENT_PATH = PROCESSED_DIR / "model_split_assignments.csv"
CALIBRATION_SELECTION_PATH = (
    PROCESSED_DIR / "retention_calibration_selection.csv"
)
CALIBRATION_PREDICTION_PATH = (
    PROCESSED_DIR / "retention_calibration_predictions.csv"
)
RANKING_SUMMARY_PATH = (
    PROCESSED_DIR / "retention_ranking_summary.csv"
)

GROUP_METRIC_PATH = (
    PROCESSED_DIR / "retention_fairness_group_metrics.csv"
)
DISPARITY_PATH = (
    PROCESSED_DIR / "retention_fairness_disparities.csv"
)
MODEL_COMPARISON_PATH = (
    PROCESSED_DIR / "retention_fairness_model_comparison.csv"
)
PREDICTION_PATH = (
    PROCESSED_DIR / "retention_fairness_predictions.csv"
)
VALIDATION_PATH = (
    PROCESSED_DIR / "retention_fairness_validation.csv"
)

KEY_COLUMNS = ["employee_id", "snapshot_date"]
TARGET_COLUMN = "attrition_next_12m"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")

    return loaded


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    int,
    str,
    bool,
]:
    """Load train and validation data without joining test targets."""

    required_paths = [
        CONFIG_PATH,
        MODEL_CONFIG_PATH,
        CALIBRATION_CONFIG_PATH,
        DATA_PATH,
        ASSIGNMENT_PATH,
        CALIBRATION_SELECTION_PATH,
        CALIBRATION_PREDICTION_PATH,
        RANKING_SUMMARY_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 44 input files:\n"
            + "\n".join(missing)
        )

    fairness_policy = load_yaml(CONFIG_PATH)
    model_policy = load_yaml(MODEL_CONFIG_PATH)
    calibration_policy = load_yaml(CALIBRATION_CONFIG_PATH)
    feature_policy = load_feature_policy()

    assignments = pd.read_csv(ASSIGNMENT_PATH)
    if TARGET_COLUMN in assignments.columns:
        raise ValueError(
            "Split assignments must not contain the attrition target."
        )
    assignments["snapshot_date"] = assignments[
        "snapshot_date"
    ].astype(str)
    test_rows = int(assignments["primary_split"].eq("test").sum())

    development_assignments = assignments.loc[
        assignments["primary_split"].isin(["train", "validation"])
    ].copy()
    dataset = pd.read_csv(DATA_PATH)
    dataset["snapshot_date"] = dataset["snapshot_date"].astype(str)
    development = dataset.merge(
        development_assignments,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_assignment"),
    )
    train = development.loc[
        development["primary_split"].eq("train")
    ].copy()
    validation = development.loc[
        development["primary_split"].eq("validation")
    ].copy()

    selection = pd.read_csv(CALIBRATION_SELECTION_PATH)
    selected_rows = selection.loc[selection["selected_method"].eq(True)]
    if len(selected_rows) != 1:
        raise ValueError(
            "Checkpoint 42 must select exactly one calibration method."
        )
    selected_method = str(selected_rows.iloc[0]["method"])
    probability_display_ready = bool(
        selected_rows.iloc[0]["probability_display_ready"]
    )

    selected_predictions = pd.read_csv(
        CALIBRATION_PREDICTION_PATH
    )
    selected_predictions = selected_predictions.loc[
        selected_predictions["method"].eq(selected_method)
    ].copy()
    selected_predictions["snapshot_date"] = selected_predictions[
        "snapshot_date"
    ].astype(str)
    selected_predictions["actual_attrition"] = selected_predictions[
        "actual_attrition"
    ].astype(int)
    selected_predictions["attrition_probability"] = pd.to_numeric(
        selected_predictions["attrition_probability"],
        errors="raise",
    )

    ranking_summary = pd.read_csv(RANKING_SUMMARY_PATH)

    return (
        train,
        validation,
        selected_predictions,
        ranking_summary,
        fairness_policy,
        model_policy,
        calibration_policy,
        feature_policy,
        test_rows,
        selected_method,
        probability_display_ready,
    )


def add_derived_groups(
    dataset: pd.DataFrame,
    fairness_policy: dict[str, Any],
) -> pd.DataFrame:
    """Create configured derived groups and normalized labels."""

    grouped = dataset.copy()
    age_policy = fairness_policy["derived_groups"]["age_band"]
    grouped["age_band"] = pd.cut(
        pd.to_numeric(
            grouped[str(age_policy["source_column"])],
            errors="raise",
        ),
        bins=[float(value) for value in age_policy["bins"]],
        labels=[str(value) for value in age_policy["labels"]],
        right=bool(age_policy["right_closed"]),
        include_lowest=True,
    )

    for attribute in fairness_policy["group_attributes"]:
        grouped[str(attribute)] = (
            grouped[str(attribute)]
            .astype("string")
            .fillna("Missing")
        )

    grouped["job_level"] = grouped["job_level"].map(
        lambda value: f"Level {value}"
    )
    return grouped


def build_exclusion_policy(
    feature_policy: dict[str, Any],
    fairness_policy: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Remove configured direct demographic inputs for sensitivity."""

    excluded_policy = deepcopy(feature_policy)
    excluded_features = [
        str(value)
        for value in fairness_policy[
            "attribute_exclusion_sensitivity"
        ]["excluded_features"]
    ]

    for feature in excluded_features:
        if feature in excluded_policy["numerical_features"]:
            excluded_policy["numerical_features"].remove(feature)
        if feature in excluded_policy["categorical_features"]:
            excluded_policy["categorical_features"].remove(feature)
        excluded_policy["reference_categories"].pop(feature, None)

    return excluded_policy, excluded_features


def mark_top_fraction(
    probability: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Return a stable global top-fraction selection indicator."""

    selected_count = min(
        len(probability),
        max(1, int(np.ceil(len(probability) * fraction))),
    )
    selected = np.zeros(len(probability), dtype=int)
    ranked_positions = np.argsort(
        -probability,
        kind="mergesort",
    )
    selected[ranked_positions[:selected_count]] = 1
    return selected


def build_prediction_variants(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    fairness_policy: dict[str, Any],
    model_policy: dict[str, Any],
    calibration_policy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Build baseline and attribute-exclusion validation predictions."""

    baseline_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "baseline_name"
        ]
    )
    exclusion_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "variant_name"
        ]
    )
    reporting_fraction = float(
        fairness_policy["reporting_fraction"]
    )

    group_columns = [
        "employee_id",
        "snapshot_date",
        TARGET_COLUMN,
        "approx_age",
        "education_level",
        "region",
        "employment_type",
        "department_name",
        "job_level",
        "organizational_level",
    ]
    validation_groups = validation.loc[:, group_columns].copy()
    baseline = selected_predictions.merge(
        validation_groups,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    targets_reconcile = bool(
        baseline["actual_attrition"]
        .eq(baseline[TARGET_COLUMN].astype(int))
        .all()
    )
    if not targets_reconcile:
        raise ValueError(
            "Saved calibration targets do not match the temporal panel."
        )
    baseline["model_variant"] = baseline_name

    excluded_policy, excluded_features = build_exclusion_policy(
        feature_policy,
        fairness_policy,
    )
    print(
        "Fitting age-and-education exclusion sensitivity model"
    )
    exclusion_probabilities, _, _ = fit_calibration_methods(
        train,
        validation,
        model_policy,
        calibration_policy,
        excluded_policy,
        "Logistic Regression",
    )
    exclusion = validation_groups.copy()
    exclusion["actual_attrition"] = exclusion[
        TARGET_COLUMN
    ].astype(int)
    exclusion["method"] = "Sigmoid"
    exclusion["attrition_probability"] = np.asarray(
        exclusion_probabilities["Sigmoid"],
        dtype=float,
    )
    exclusion["model_variant"] = exclusion_name

    common_columns = [
        *group_columns,
        "actual_attrition",
        "method",
        "attrition_probability",
        "model_variant",
    ]
    predictions = pd.concat(
        [
            baseline.loc[:, common_columns],
            exclusion.loc[:, common_columns],
        ],
        ignore_index=True,
    )
    predictions = add_derived_groups(
        predictions,
        fairness_policy,
    )
    predictions["selected_top_fraction"] = 0

    for model_variant in predictions["model_variant"].unique():
        variant_mask = predictions["model_variant"].eq(model_variant)
        selected = mark_top_fraction(
            predictions.loc[
                variant_mask,
                "attrition_probability",
            ].to_numpy(dtype=float),
            reporting_fraction,
        )
        predictions.loc[
            variant_mask,
            "selected_top_fraction",
        ] = selected

    predictions["selected_top_fraction"] = predictions[
        "selected_top_fraction"
    ].astype(int)
    return predictions, excluded_features


def wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Calculate a Wilson score interval for one binomial rate."""

    if trials <= 0:
        return np.nan, np.nan

    z_score = NormalDist().inv_cdf(
        0.5 + confidence_level / 2.0
    )
    rate = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (
        rate + z_score**2 / (2.0 * trials)
    ) / denominator
    margin = (
        z_score
        * np.sqrt(
            rate * (1.0 - rate) / trials
            + z_score**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def safe_roc_auc(
    target: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Calculate ROC-AUC only when both target classes exist."""

    if len(np.unique(target)) < 2:
        return np.nan
    return float(roc_auc_score(target, probability))


def safe_pr_auc(
    target: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Calculate PR-AUC only when positive cases exist."""

    if int(target.sum()) == 0:
        return np.nan
    return float(average_precision_score(target, probability))


def group_metric_record(
    group: pd.DataFrame,
    model_variant: str,
    attribute: str,
    group_value: str,
    fairness_policy: dict[str, Any],
) -> dict[str, Any]:
    """Calculate classification, ranking, and calibration metrics."""

    target = group["actual_attrition"].to_numpy(dtype=int)
    probability = group["attrition_probability"].to_numpy(
        dtype=float
    )
    selected = group["selected_top_fraction"].to_numpy(dtype=int)
    positive_cases = int(target.sum())
    negative_cases = int(len(target) - positive_cases)
    selected_count = int(selected.sum())

    true_positive = int(((target == 1) & (selected == 1)).sum())
    false_positive = int(((target == 0) & (selected == 1)).sum())
    false_negative = int(((target == 1) & (selected == 0)).sum())
    true_negative = int(((target == 0) & (selected == 0)).sum())

    selection_rate = selected_count / len(group)
    precision = (
        true_positive / selected_count
        if selected_count > 0
        else np.nan
    )
    true_positive_rate = (
        true_positive / positive_cases
        if positive_cases > 0
        else np.nan
    )
    false_positive_rate = (
        false_positive / negative_cases
        if negative_cases > 0
        else np.nan
    )
    observed_rate = float(target.mean())
    mean_probability = float(probability.mean())

    confidence_level = float(
        fairness_policy["intervals"]["confidence_level"]
    )
    selection_lower, selection_upper = wilson_interval(
        selected_count,
        len(group),
        confidence_level,
    )
    precision_lower, precision_upper = wilson_interval(
        true_positive,
        selected_count,
        confidence_level,
    )
    recall_lower, recall_upper = wilson_interval(
        true_positive,
        positive_cases,
        confidence_level,
    )
    fpr_lower, fpr_upper = wilson_interval(
        false_positive,
        negative_cases,
        confidence_level,
    )

    eligibility = fairness_policy["comparison_eligibility"]
    eligible = (
        len(group) >= int(eligibility["minimum_group_rows"])
        and positive_cases
        >= int(eligibility["minimum_positive_cases"])
        and negative_cases
        >= int(eligibility["minimum_negative_cases"])
        and selected_count
        >= int(eligibility["minimum_selected_rows"])
    )

    return {
        "model_variant": model_variant,
        "attribute": attribute,
        "group": group_value,
        "sample_size": len(group),
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "observed_attrition_rate": observed_rate,
        "mean_predicted_probability": mean_probability,
        "calibration_gap": mean_probability - observed_rate,
        "absolute_calibration_gap": abs(
            mean_probability - observed_rate
        ),
        "brier_score": float(
            brier_score_loss(target, probability)
        ),
        "pr_auc": safe_pr_auc(target, probability),
        "roc_auc": safe_roc_auc(target, probability),
        "selected_count": selected_count,
        "selection_rate": selection_rate,
        "selection_rate_lower_95": selection_lower,
        "selection_rate_upper_95": selection_upper,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "precision_lower_95": precision_lower,
        "precision_upper_95": precision_upper,
        "true_positive_rate": true_positive_rate,
        "true_positive_rate_lower_95": recall_lower,
        "true_positive_rate_upper_95": recall_upper,
        "false_positive_rate": false_positive_rate,
        "false_positive_rate_lower_95": fpr_lower,
        "false_positive_rate_upper_95": fpr_upper,
        "eligible_for_disparity_comparison": eligible,
    }


def build_group_metrics(
    predictions: pd.DataFrame,
    fairness_policy: dict[str, Any],
) -> pd.DataFrame:
    """Calculate metrics for every configured group and model variant."""

    records: list[dict[str, Any]] = []
    attributes = [
        str(value)
        for value in fairness_policy["group_attributes"]
    ]

    for model_variant, variant in predictions.groupby(
        "model_variant",
        sort=False,
    ):
        for attribute in attributes:
            for group_value, group in variant.groupby(
                attribute,
                sort=False,
                observed=True,
            ):
                records.append(
                    group_metric_record(
                        group,
                        str(model_variant),
                        attribute,
                        str(group_value),
                        fairness_policy,
                    )
                )

    metrics = pd.DataFrame(records)
    metrics["group_label"] = (
        metrics["attribute"].str.replace("_", " ").str.title()
        + " — "
        + metrics["group"]
    )
    return metrics


def metric_extremes(
    metrics: pd.DataFrame,
    column: str,
) -> tuple[float, float, str, str]:
    """Return minimum, maximum, and their group labels."""

    valid = metrics.dropna(subset=[column])
    if valid.empty:
        return np.nan, np.nan, "", ""

    minimum_index = valid[column].idxmin()
    maximum_index = valid[column].idxmax()
    return (
        float(valid.loc[minimum_index, column]),
        float(valid.loc[maximum_index, column]),
        str(valid.loc[minimum_index, "group"]),
        str(valid.loc[maximum_index, "group"]),
    )


def build_disparity_summary(
    group_metrics: pd.DataFrame,
    fairness_policy: dict[str, Any],
) -> pd.DataFrame:
    """Summarize descriptive max-minus-min subgroup disparities."""

    records: list[dict[str, Any]] = []
    thresholds = fairness_policy["screening_thresholds"]

    for model_variant, variant_metrics in group_metrics.groupby(
        "model_variant",
        sort=False,
    ):
        for attribute, attribute_metrics in variant_metrics.groupby(
            "attribute",
            sort=False,
        ):
            eligible = attribute_metrics.loc[
                attribute_metrics[
                    "eligible_for_disparity_comparison"
                ].eq(True)
            ].copy()

            (
                selection_min,
                selection_max,
                selection_low,
                selection_high,
            ) = metric_extremes(eligible, "selection_rate")
            recall_min, recall_max, recall_low, recall_high = (
                metric_extremes(eligible, "true_positive_rate")
            )
            fpr_min, fpr_max, fpr_low, fpr_high = metric_extremes(
                eligible,
                "false_positive_rate",
            )
            precision_min, precision_max, _, _ = metric_extremes(
                eligible,
                "precision",
            )
            calibration_min, calibration_max, _, _ = (
                metric_extremes(
                    eligible,
                    "calibration_gap",
                )
            )

            demographic_parity_difference = (
                selection_max - selection_min
            )
            equal_opportunity_difference = recall_max - recall_min
            false_positive_rate_difference = fpr_max - fpr_min
            equalized_odds_max_difference = max(
                equal_opportunity_difference,
                false_positive_rate_difference,
            )
            maximum_absolute_calibration_gap = float(
                eligible["absolute_calibration_gap"].max()
            )
            selection_rate_ratio = (
                selection_min / selection_max
                if selection_max > 0
                else np.nan
            )

            review_flags = {
                "demographic_parity": (
                    demographic_parity_difference
                    > float(
                        thresholds[
                            "demographic_parity_difference"
                        ]
                    )
                ),
                "equal_opportunity": (
                    equal_opportunity_difference
                    > float(
                        thresholds[
                            "equal_opportunity_difference"
                        ]
                    )
                ),
                "false_positive_rate": (
                    false_positive_rate_difference
                    > float(
                        thresholds[
                            "false_positive_rate_difference"
                        ]
                    )
                ),
                "calibration": (
                    maximum_absolute_calibration_gap
                    > float(
                        thresholds[
                            "maximum_absolute_calibration_gap"
                        ]
                    )
                ),
            }

            records.append(
                {
                    "model_variant": model_variant,
                    "attribute": attribute,
                    "eligible_groups": len(eligible),
                    "total_groups": len(attribute_metrics),
                    "demographic_parity_difference": (
                        demographic_parity_difference
                    ),
                    "selection_rate_ratio": selection_rate_ratio,
                    "lowest_selection_group": selection_low,
                    "highest_selection_group": selection_high,
                    "equal_opportunity_difference": (
                        equal_opportunity_difference
                    ),
                    "lowest_recall_group": recall_low,
                    "highest_recall_group": recall_high,
                    "false_positive_rate_difference": (
                        false_positive_rate_difference
                    ),
                    "lowest_fpr_group": fpr_low,
                    "highest_fpr_group": fpr_high,
                    "equalized_odds_max_difference": (
                        equalized_odds_max_difference
                    ),
                    "precision_difference": (
                        precision_max - precision_min
                    ),
                    "calibration_gap_range": (
                        calibration_max - calibration_min
                    ),
                    "maximum_absolute_calibration_gap": (
                        maximum_absolute_calibration_gap
                    ),
                    "screening_review_flag": any(
                        review_flags.values()
                    ),
                    "review_trigger_count": sum(
                        review_flags.values()
                    ),
                    "review_triggers": ", ".join(
                        name
                        for name, triggered in review_flags.items()
                        if triggered
                    )
                    or "None",
                    "binary_fairness_verdict": "Not assigned",
                }
            )

    return pd.DataFrame(records)


def top_fraction_summary(
    target: np.ndarray,
    probability: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int]:
    """Return overall ranking metrics for one model variant."""

    selected_count = int(selected.sum())
    captured = int(((target == 1) & (selected == 1)).sum())
    positive_cases = int(target.sum())
    precision = captured / selected_count
    capture = captured / positive_cases
    baseline = float(target.mean())

    return {
        "validation_rows": len(target),
        "positive_cases": positive_cases,
        "observed_attrition_rate": baseline,
        "mean_predicted_probability": float(
            probability.mean()
        ),
        "mean_calibration_gap": float(
            probability.mean() - baseline
        ),
        "brier_score": float(
            brier_score_loss(target, probability)
        ),
        "pr_auc": float(
            average_precision_score(target, probability)
        ),
        "roc_auc": float(roc_auc_score(target, probability)),
        "selected_count": selected_count,
        "captured_positive_cases": captured,
        "top_fraction_precision": precision,
        "top_fraction_capture": capture,
        "top_fraction_lift": precision / baseline,
    }


def build_model_comparison(
    predictions: pd.DataFrame,
    fairness_policy: dict[str, Any],
) -> pd.DataFrame:
    """Compare overall validation evidence across model variants."""

    baseline_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "baseline_name"
        ]
    )
    records: list[dict[str, Any]] = []

    for model_variant, variant in predictions.groupby(
        "model_variant",
        sort=False,
    ):
        summary = top_fraction_summary(
            variant["actual_attrition"].to_numpy(dtype=int),
            variant["attrition_probability"].to_numpy(dtype=float),
            variant["selected_top_fraction"].to_numpy(dtype=int),
        )
        records.append(
            {
                "model_variant": model_variant,
                **summary,
                "selected_for_use": model_variant == baseline_name,
                "sensitivity_analysis_only": (
                    model_variant != baseline_name
                ),
            }
        )

    comparison = pd.DataFrame(records)
    baseline = comparison.loc[
        comparison["model_variant"].eq(baseline_name)
    ].iloc[0]
    for metric in [
        "brier_score",
        "pr_auc",
        "roc_auc",
        "top_fraction_precision",
        "top_fraction_capture",
        "top_fraction_lift",
    ]:
        comparison[f"{metric}_difference_vs_selected"] = (
            comparison[metric] - float(baseline[metric])
        )

    return comparison


def save_figure(
    figure: Figure,
    filename: str,
    fairness_policy: dict[str, Any],
) -> None:
    """Save one deterministic portfolio figure."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / filename,
        dpi=int(fairness_policy["figures"]["dpi"]),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib"},
    )
    plt.close(figure)


def plot_group_rate(
    group_metrics: pd.DataFrame,
    fairness_policy: dict[str, Any],
    metric: str,
    lower_column: str,
    upper_column: str,
    filename: str,
    title: str,
    reference_rate: float | None = None,
) -> None:
    """Create one readable horizontal subgroup rate chart."""

    baseline_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "baseline_name"
        ]
    )
    plotted = group_metrics.loc[
        group_metrics["model_variant"].eq(baseline_name)
        & group_metrics[
            "eligible_for_disparity_comparison"
        ].eq(True)
    ].copy()
    plotted = plotted.sort_values(
        ["attribute", metric],
        ascending=[True, True],
    )

    values = plotted[metric].to_numpy(dtype=float)
    lower = plotted[lower_column].to_numpy(dtype=float)
    upper = plotted[upper_column].to_numpy(dtype=float)
    errors = np.vstack([values - lower, upper - values])

    figure, axis = plt.subplots(
        figsize=(11, max(8, 0.34 * len(plotted))),
    )
    positions = np.arange(len(plotted))
    axis.barh(
        positions,
        values,
        color="#3B6E8F",
        alpha=0.88,
    )
    axis.errorbar(
        values,
        positions,
        xerr=errors,
        fmt="none",
        ecolor="#233746",
        capsize=2,
        linewidth=0.8,
    )
    if reference_rate is not None:
        axis.axvline(
            reference_rate,
            color="#C65D3B",
            linestyle="--",
            linewidth=1.5,
            label="Overall validation rate",
        )
        axis.legend(loc="lower right", frameon=False)

    axis.set_yticks(positions)
    axis.set_yticklabels(plotted["group_label"])
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Rate")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, filename, fairness_policy)


def plot_calibration_gap(
    group_metrics: pd.DataFrame,
    fairness_policy: dict[str, Any],
) -> None:
    """Create a diverging chart of subgroup mean calibration gaps."""

    baseline_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "baseline_name"
        ]
    )
    plotted = group_metrics.loc[
        group_metrics["model_variant"].eq(baseline_name)
        & group_metrics[
            "eligible_for_disparity_comparison"
        ].eq(True)
    ].copy()
    plotted = plotted.sort_values(
        ["attribute", "calibration_gap"],
        ascending=[True, True],
    )

    values = plotted["calibration_gap"].to_numpy(dtype=float)
    colors = np.where(
        values >= 0,
        "#C65D3B",
        "#3B6E8F",
    )
    positions = np.arange(len(plotted))
    figure, axis = plt.subplots(
        figsize=(11, max(8, 0.34 * len(plotted))),
    )
    axis.barh(positions, values, color=colors, alpha=0.88)
    axis.axvline(0.0, color="#202020", linewidth=1.0)
    axis.set_yticks(positions)
    axis.set_yticklabels(plotted["group_label"])
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel(
        "Mean predicted probability minus observed attrition rate"
    )
    axis.set_title(
        "Subgroup calibration gaps",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    save_figure(
        figure,
        "calibration_gap_by_group.png",
        fairness_policy,
    )


def plot_disparity_comparison(
    disparities: pd.DataFrame,
    fairness_policy: dict[str, Any],
) -> None:
    """Compare disparity gaps for both diagnostic model variants."""

    attributes = [
        str(value)
        for value in fairness_policy["group_attributes"]
    ]
    variants = list(disparities["model_variant"].unique())
    colors = ["#3B6E8F", "#C58B39"]
    positions = np.arange(len(attributes))
    bar_height = 0.34

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        sharey=True,
    )
    metrics = [
        (
            "demographic_parity_difference",
            "Selection-rate gap",
        ),
        (
            "equal_opportunity_difference",
            "Recall gap",
        ),
    ]

    for axis, (metric, title) in zip(axes, metrics):
        for variant_index, model_variant in enumerate(variants):
            indexed = (
                disparities.loc[
                    disparities["model_variant"].eq(model_variant)
                ]
                .set_index("attribute")
                .reindex(attributes)
            )
            offset = (
                variant_index - (len(variants) - 1) / 2
            ) * bar_height
            axis.barh(
                positions + offset,
                indexed[metric].to_numpy(dtype=float),
                height=bar_height,
                color=colors[variant_index],
                label=model_variant,
                alpha=0.9,
            )

        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("Maximum minus minimum eligible-group rate")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right", "left"]].set_visible(False)

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        [
            value.replace("_", " ").title()
            for value in attributes
        ]
    )
    axes[1].legend(loc="lower right", frameon=False)
    figure.suptitle(
        "Descriptive subgroup disparity comparison",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(
        figure,
        "disparity_comparison.png",
        fairness_policy,
    )


def create_figures(
    group_metrics: pd.DataFrame,
    disparities: pd.DataFrame,
    model_comparison: pd.DataFrame,
    fairness_policy: dict[str, Any],
) -> None:
    """Render all configured subgroup diagnostic figures."""

    baseline_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "baseline_name"
        ]
    )
    baseline = model_comparison.loc[
        model_comparison["model_variant"].eq(baseline_name)
    ].iloc[0]

    plot_group_rate(
        group_metrics,
        fairness_policy,
        "selection_rate",
        "selection_rate_lower_95",
        "selection_rate_upper_95",
        "selection_rate_by_group.png",
        "Global top-10% selection rate by subgroup",
        float(fairness_policy["reporting_fraction"]),
    )
    plot_group_rate(
        group_metrics,
        fairness_policy,
        "true_positive_rate",
        "true_positive_rate_lower_95",
        "true_positive_rate_upper_95",
        "recall_by_group.png",
        "Attrition-case capture rate by subgroup",
        float(baseline["top_fraction_capture"]),
    )
    plot_calibration_gap(group_metrics, fairness_policy)
    plot_disparity_comparison(disparities, fairness_policy)


def add_check(
    records: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> None:
    """Append one validation record."""

    records.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def validate_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    predictions: pd.DataFrame,
    group_metrics: pd.DataFrame,
    disparities: pd.DataFrame,
    model_comparison: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    fairness_policy: dict[str, Any],
    test_rows: int,
    selected_method: str,
    probability_display_ready: bool,
    excluded_features: list[str],
) -> pd.DataFrame:
    """Validate leakage protection, reconciliation, and output coverage."""

    records: list[dict[str, Any]] = []
    validation_policy = fairness_policy["validation"]
    baseline_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "baseline_name"
        ]
    )
    exclusion_name = str(
        fairness_policy["attribute_exclusion_sensitivity"][
            "variant_name"
        ]
    )
    expected_variants = {baseline_name, exclusion_name}

    add_check(
        records,
        "Configured calibration method remains selected",
        selected_method
        == str(fairness_policy["selected_calibration_method"]),
        selected_method,
        fairness_policy["selected_calibration_method"],
        "Checkpoint 44 must reuse Checkpoint 42's selected method.",
    )
    add_check(
        records,
        "Selected scores remain probability-display ready",
        probability_display_ready,
        probability_display_ready,
        True,
        "Subgroup calibration uses the approved probability scale.",
    )
    no_test_predictions = not bool(
        predictions["snapshot_date"].eq("2025-06-30").any()
    )
    add_check(
        records,
        "Reserved test target remains unused",
        test_rows > 0 and no_test_predictions,
        f"{test_rows} reserved rows; 0 test predictions analyzed",
        "Reserved rows > 0 and no 2025 prediction rows",
        "Only train data fits sensitivity models; 2024 validates them.",
    )

    expected_snapshot = str(
        fairness_policy["validation_snapshot"]
    )
    observed_snapshots = sorted(
        predictions["snapshot_date"].astype(str).unique()
    )
    add_check(
        records,
        "Only the configured validation snapshot is analyzed",
        observed_snapshots == [expected_snapshot],
        observed_snapshots,
        [expected_snapshot],
        "Fairness diagnostics use one chronological validation period.",
    )
    duplicate_keys = int(
        predictions.duplicated(
            [*KEY_COLUMNS, "model_variant"]
        ).sum()
    )
    add_check(
        records,
        "Employee-snapshot-variant keys are unique",
        duplicate_keys == 0,
        duplicate_keys,
        0,
        "Each employee receives one score per diagnostic variant.",
    )
    target_values = sorted(
        predictions["actual_attrition"].unique().tolist()
    )
    target_complete = not bool(
        predictions["actual_attrition"].isna().any()
    )
    add_check(
        records,
        "Validation target is complete and binary",
        target_values == [0, 1] and target_complete,
        target_values,
        [0, 1],
        "Subgroup rates require known validation outcomes.",
    )

    unavailable = fairness_policy["unavailable_attributes"]
    gender_absent = (
        "gender" not in train.columns
        and "gender" not in validation.columns
        and str(unavailable["gender"]["status"]) == "not_present"
    )
    add_check(
        records,
        "Unavailable gender attribute is not invented",
        gender_absent,
        "Absent and documented" if gender_absent else "Unexpected",
        "Absent and documented",
        "The synthetic schema is audited as it exists.",
    )

    expected_attributes = {
        str(value)
        for value in fairness_policy["group_attributes"]
    }
    observed_attributes = set(group_metrics["attribute"])
    add_check(
        records,
        "Every configured subgroup attribute is reported",
        observed_attributes == expected_attributes,
        sorted(observed_attributes),
        sorted(expected_attributes),
        "No configured subgroup dimension may be silently skipped.",
    )

    minimum_rows = int(
        validation_policy["minimum_validation_rows"]
    )
    minimum_positives = int(
        validation_policy["minimum_validation_positive_cases"]
    )
    add_check(
        records,
        "Validation sample is sufficiently large",
        len(validation) >= minimum_rows
        and int(validation[TARGET_COLUMN].sum())
        >= minimum_positives,
        (
            f"{len(validation)} rows; "
            f"{int(validation[TARGET_COLUMN].sum())} positives"
        ),
        (
            f">= {minimum_rows} rows; "
            f">= {minimum_positives} positives"
        ),
        "Subgroup diagnostics require enough validation evidence.",
    )

    observed_variants = set(predictions["model_variant"])
    add_check(
        records,
        "Both configured model variants are complete",
        observed_variants == expected_variants
        and predictions.groupby("model_variant").size().nunique() == 1,
        sorted(observed_variants),
        sorted(expected_variants),
        "The exclusion model is diagnostic and not silently omitted.",
    )
    add_check(
        records,
        "Configured direct attributes are excluded exactly",
        set(excluded_features)
        == set(
            fairness_policy[
                "attribute_exclusion_sensitivity"
            ]["excluded_features"]
        ),
        excluded_features,
        fairness_policy["attribute_exclusion_sensitivity"][
            "excluded_features"
        ],
        "The sensitivity model removes only age and education.",
    )

    probability = predictions["attrition_probability"].to_numpy(
        dtype=float
    )
    probability_valid = bool(
        np.isfinite(probability).all()
        and probability.min()
        >= float(validation_policy["probability_minimum"])
        and probability.max()
        <= float(validation_policy["probability_maximum"])
    )
    add_check(
        records,
        "All diagnostic probabilities are finite and bounded",
        probability_valid,
        f"min={probability.min():.6f}; max={probability.max():.6f}",
        "All finite and inside [0, 1]",
        "Every model variant must output valid probabilities.",
    )

    baseline_predictions = predictions.loc[
        predictions["model_variant"].eq(baseline_name)
    ]
    baseline_reconciliation = selected_predictions.merge(
        baseline_predictions[
            [
                *KEY_COLUMNS,
                "attrition_probability",
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_checkpoint42", "_checkpoint44"),
        validate="one_to_one",
    )
    maximum_probability_difference = float(
        (
            baseline_reconciliation[
                "attrition_probability_checkpoint42"
            ]
            - baseline_reconciliation[
                "attrition_probability_checkpoint44"
            ]
        )
        .abs()
        .max()
    )
    tolerance = float(
        validation_policy["metric_reconciliation_tolerance"]
    )
    add_check(
        records,
        "Selected-model scores reproduce Checkpoint 42",
        maximum_probability_difference <= tolerance,
        maximum_probability_difference,
        f"<= {tolerance}",
        "Fairness starts from the exact selected calibrated scores.",
    )

    baseline_summary = model_comparison.loc[
        model_comparison["model_variant"].eq(baseline_name)
    ].iloc[0]
    checkpoint_43 = ranking_summary.iloc[0]
    metric_differences = {
        "pr_auc": abs(
            float(baseline_summary["pr_auc"])
            - float(checkpoint_43["pr_auc"])
        ),
        "roc_auc": abs(
            float(baseline_summary["roc_auc"])
            - float(checkpoint_43["roc_auc"])
        ),
        "precision": abs(
            float(baseline_summary["top_fraction_precision"])
            - float(checkpoint_43["reporting_precision"])
        ),
        "capture": abs(
            float(baseline_summary["top_fraction_capture"])
            - float(checkpoint_43["reporting_capture_rate"])
        ),
    }
    add_check(
        records,
        "Selected-model metrics reproduce Checkpoint 43",
        max(metric_differences.values()) <= tolerance,
        metric_differences,
        f"Every difference <= {tolerance}",
        "Subgroup analysis must reconcile to the ranking checkpoint.",
    )

    rate_columns = [
        "selection_rate",
        "precision",
        "true_positive_rate",
        "false_positive_rate",
        "observed_attrition_rate",
        "mean_predicted_probability",
        "brier_score",
        "pr_auc",
        "roc_auc",
    ]
    metric_ranges_valid = all(
        group_metrics[column].dropna().between(0.0, 1.0).all()
        for column in rate_columns
    )
    add_check(
        records,
        "All subgroup metrics remain in valid ranges",
        metric_ranges_valid,
        metric_ranges_valid,
        True,
        "Rates, ranking metrics, and Brier scores use [0, 1].",
    )

    interval_columns = [
        column
        for column in group_metrics.columns
        if column.endswith("_lower_95")
        or column.endswith("_upper_95")
    ]
    intervals_valid = all(
        group_metrics[column].dropna().between(0.0, 1.0).all()
        for column in interval_columns
    )
    add_check(
        records,
        "All Wilson confidence intervals are valid",
        intervals_valid,
        intervals_valid,
        True,
        "Uncertain subgroup rates are reported with bounded intervals.",
    )

    eligible_counts = (
        group_metrics.loc[
            group_metrics[
                "eligible_for_disparity_comparison"
            ].eq(True)
        ]
        .groupby(["model_variant", "attribute"])
        .size()
    )
    eligible_count_values = eligible_counts.to_numpy(dtype=int)
    disparity_group_values = disparities[
        "eligible_groups"
    ].to_numpy(dtype=int)
    disparities_valid = bool(
        len(disparities)
        == len(expected_variants) * len(expected_attributes)
        and (eligible_count_values >= 2).all()
        and (disparity_group_values >= 2).all()
    )
    add_check(
        records,
        "Disparities use at least two eligible groups",
        disparities_valid,
        int(disparities["eligible_groups"].min()),
        ">= 2 per attribute and variant",
        "Small or sparse groups do not determine disparity ranges.",
    )

    screening_values = disparities[
        [
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "false_positive_rate_difference",
            "equalized_odds_max_difference",
            "maximum_absolute_calibration_gap",
        ]
    ]
    screening_array = screening_values.to_numpy(dtype=float)
    disparity_ranges_valid = bool(
        np.isfinite(screening_array).all()
        and (screening_array >= 0.0).all()
        and (screening_array <= 1.0).all()
    )
    add_check(
        records,
        "Descriptive disparity measures are valid",
        disparity_ranges_valid,
        disparity_ranges_valid,
        True,
        "Max-minus-min gaps must remain inside [0, 1].",
    )

    interpretation = fairness_policy["interpretation_policy"]
    no_verdict = bool(
        bool(interpretation["no_binary_fairness_verdict"])
        and bool(
            disparities["binary_fairness_verdict"]
            .eq("Not assigned")
            .all()
        )
    )
    add_check(
        records,
        "No binary fairness verdict is assigned",
        no_verdict,
        "Not assigned",
        "Not assigned",
        "Diagnostics trigger review; they do not prove fairness.",
    )
    no_threshold = (
        not bool(
            interpretation["final_operating_threshold_selected"]
        )
    )
    add_check(
        records,
        "No final operating threshold is selected",
        no_threshold,
        no_threshold,
        True,
        "Checkpoint 46 will make policy choices using explicit costs.",
    )

    expected_figures = {
        str(value)
        for value in fairness_policy["figures"]["expected_files"]
    }
    observed_figures = {
        path.name
        for path in FIGURE_DIR.glob("*.png")
        if path.name in expected_figures
    }
    add_check(
        records,
        "All subgroup figures were generated",
        observed_figures == expected_figures,
        sorted(observed_figures),
        sorted(expected_figures),
        "Every committed visualization specification must render.",
    )

    checks = pd.DataFrame(records)
    failures = checks.loc[checks["status"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(
            "Checkpoint 44 validation failed:\n"
            + failures.to_string(index=False)
        )

    return checks


def save_outputs(
    predictions: pd.DataFrame,
    group_metrics: pd.DataFrame,
    disparities: pd.DataFrame,
    model_comparison: pd.DataFrame,
    validation_checks: pd.DataFrame,
) -> None:
    """Save all reproducible Checkpoint 44 data products."""

    prediction_columns = [
        "employee_id",
        "snapshot_date",
        "actual_attrition",
        "method",
        "model_variant",
        "attrition_probability",
        "selected_top_fraction",
    ]
    predictions.loc[:, prediction_columns].to_csv(
        PREDICTION_PATH,
        index=False,
    )
    group_metrics.to_csv(GROUP_METRIC_PATH, index=False)
    disparities.to_csv(DISPARITY_PATH, index=False)
    model_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    validation_checks.to_csv(VALIDATION_PATH, index=False)


def print_results(
    group_metrics: pd.DataFrame,
    disparities: pd.DataFrame,
    model_comparison: pd.DataFrame,
    validation_checks: pd.DataFrame,
) -> None:
    """Print the most useful Checkpoint 44 evidence."""

    baseline_name = "Selected model"
    display_columns = [
        "attribute",
        "group",
        "sample_size",
        "positive_cases",
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
        "calibration_gap",
        "eligible_for_disparity_comparison",
    ]

    print("\nMODEL VARIANT COMPARISON")
    print(model_comparison.to_string(index=False))

    print("\nSELECTED-MODEL SUBGROUP METRICS")
    print(
        group_metrics.loc[
            group_metrics["model_variant"].eq(baseline_name),
            display_columns,
        ].to_string(index=False)
    )

    print("\nDESCRIPTIVE DISPARITY SUMMARY")
    disparity_columns = [
        "model_variant",
        "attribute",
        "eligible_groups",
        "demographic_parity_difference",
        "equal_opportunity_difference",
        "false_positive_rate_difference",
        "maximum_absolute_calibration_gap",
        "screening_review_flag",
        "review_triggers",
    ]
    print(disparities[disparity_columns].to_string(index=False))

    print("\nFAIRNESS AND SUBGROUP VALIDATION")
    print(validation_checks.to_string(index=False))

    print(f"\nSaved group metrics: {GROUP_METRIC_PATH}")
    print(f"Saved figures: {FIGURE_DIR}")


def main() -> None:
    """Run the complete Checkpoint 44 analysis."""

    (
        train,
        validation,
        selected_predictions,
        ranking_summary,
        fairness_policy,
        model_policy,
        calibration_policy,
        feature_policy,
        test_rows,
        selected_method,
        probability_display_ready,
    ) = load_inputs()

    predictions, excluded_features = build_prediction_variants(
        train,
        validation,
        selected_predictions,
        fairness_policy,
        model_policy,
        calibration_policy,
        feature_policy,
    )
    group_metrics = build_group_metrics(
        predictions,
        fairness_policy,
    )
    disparities = build_disparity_summary(
        group_metrics,
        fairness_policy,
    )
    model_comparison = build_model_comparison(
        predictions,
        fairness_policy,
    )
    create_figures(
        group_metrics,
        disparities,
        model_comparison,
        fairness_policy,
    )
    validation_checks = validate_outputs(
        train,
        validation,
        selected_predictions,
        predictions,
        group_metrics,
        disparities,
        model_comparison,
        ranking_summary,
        fairness_policy,
        test_rows,
        selected_method,
        probability_display_ready,
        excluded_features,
    )
    save_outputs(
        predictions,
        group_metrics,
        disparities,
        model_comparison,
        validation_checks,
    )
    print_results(
        group_metrics,
        disparities,
        model_comparison,
        validation_checks,
    )

    print(
        "\nRETENTION FAIRNESS AND SUBGROUP ANALYSIS "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
