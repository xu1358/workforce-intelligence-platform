"""Evaluate selected retention-model ranking without using test outcomes.

Checkpoint 43 measures top-k precision, capture, and lift for the
sigmoid-calibrated Logistic Regression on the 2024 validation snapshot.
It also builds PR, ROC, cumulative-gains, lift, distribution, and
reporting-only confusion-matrix figures. The 2025 test target stays locked.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

MATPLOTLIB_CACHE_DIR = (
    Path(tempfile.gettempdir())
    / "workforce_intelligence_matplotlib"
)
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(MATPLOTLIB_CACHE_DIR),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROCESSED_DIR / "ranking_figures"

CONFIG_PATH = (
    PROJECT_ROOT / "config" / "retention_ranking.yaml"
)
ASSIGNMENT_PATH = (
    PROCESSED_DIR / "model_split_assignments.csv"
)
CALIBRATION_SELECTION_PATH = (
    PROCESSED_DIR / "retention_calibration_selection.csv"
)
CALIBRATION_METRICS_PATH = (
    PROCESSED_DIR / "retention_calibration_metrics.csv"
)
CALIBRATION_PREDICTIONS_PATH = (
    PROCESSED_DIR / "retention_calibration_predictions.csv"
)

RANKING_METRICS_PATH = (
    PROCESSED_DIR / "retention_ranking_metrics.csv"
)
BOOTSTRAP_PATH = (
    PROCESSED_DIR / "retention_ranking_bootstrap.csv"
)
SUMMARY_PATH = (
    PROCESSED_DIR / "retention_ranking_summary.csv"
)
CURVE_PATH = (
    PROCESSED_DIR / "retention_curve_points.csv"
)
CONFUSION_PATH = (
    PROCESSED_DIR / "retention_top_decile_confusion_matrix.csv"
)
VALIDATION_PATH = (
    PROCESSED_DIR / "retention_ranking_validation.csv"
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML configuration."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")

    return loaded


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.Series,
    dict[str, Any],
    int,
    str,
    bool,
]:
    """Load selected validation scores while excluding test targets."""

    required_paths = [
        CONFIG_PATH,
        ASSIGNMENT_PATH,
        CALIBRATION_SELECTION_PATH,
        CALIBRATION_METRICS_PATH,
        CALIBRATION_PREDICTIONS_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 43 input files:\n"
            + "\n".join(missing)
        )

    ranking_policy = load_yaml(CONFIG_PATH)
    assignments = pd.read_csv(ASSIGNMENT_PATH)
    if "attrition_next_12m" in assignments.columns:
        raise ValueError(
            "Split assignments must not contain the attrition target."
        )
    test_rows = int(assignments["primary_split"].eq("test").sum())

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

    predictions = pd.read_csv(CALIBRATION_PREDICTIONS_PATH)
    predictions = predictions.loc[
        predictions["method"].eq(selected_method)
    ].copy()
    predictions["snapshot_date"] = predictions[
        "snapshot_date"
    ].astype(str)
    predictions["actual_attrition"] = predictions[
        "actual_attrition"
    ].astype(int)
    predictions["attrition_probability"] = pd.to_numeric(
        predictions["attrition_probability"],
        errors="raise",
    )

    calibration_metrics = pd.read_csv(CALIBRATION_METRICS_PATH)
    selected_metrics = calibration_metrics.loc[
        calibration_metrics["method"].eq(selected_method)
    ]
    if len(selected_metrics) != 1:
        raise ValueError(
            "Missing selected Checkpoint 42 calibration metrics."
        )

    return (
        predictions,
        selected_metrics.iloc[0],
        ranking_policy,
        test_rows,
        selected_method,
        probability_display_ready,
    )


def ranked_positions(probability: np.ndarray) -> np.ndarray:
    """Return descending score positions with stable tie handling."""

    return np.argsort(-probability, kind="mergesort")


def top_fraction_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    """Calculate ranking metrics for one selected fraction."""

    rows = len(target)
    selected_count = min(
        rows,
        max(1, int(np.ceil(rows * fraction))),
    )
    ordered = ranked_positions(probability)
    selected_positions = ordered[:selected_count]
    selected_target = target[selected_positions]
    positive_cases = int(target.sum())
    captured_positive_cases = int(selected_target.sum())
    baseline_rate = float(target.mean())
    precision = float(selected_target.mean())
    capture = (
        float(captured_positive_cases / positive_cases)
        if positive_cases > 0
        else 0.0
    )
    lift = (
        float(precision / baseline_rate)
        if baseline_rate > 0
        else 0.0
    )
    false_positives = selected_count - captured_positive_cases

    return {
        "requested_fraction": fraction,
        "selected_count": selected_count,
        "actual_selected_fraction": selected_count / rows,
        "captured_positive_cases": captured_positive_cases,
        "false_positive_cases": false_positives,
        "precision_at_k": precision,
        "capture_rate_at_k": capture,
        "lift_at_k": lift,
        "expected_random_positive_cases": (
            selected_count * baseline_rate
        ),
        "minimum_selected_probability": float(
            probability[selected_positions].min()
        ),
        "maximum_unselected_probability": (
            float(probability[ordered[selected_count]])
            if selected_count < rows
            else np.nan
        ),
        "reviews_per_captured_case": (
            selected_count / captured_positive_cases
            if captured_positive_cases > 0
            else np.nan
        ),
    }


def build_ranking_grid(
    target: np.ndarray,
    probability: np.ndarray,
    ranking_policy: dict[str, Any],
) -> pd.DataFrame:
    """Calculate observed top-k metrics on the configured grid."""

    records = [
        top_fraction_metrics(
            target,
            probability,
            float(fraction),
        )
        for fraction in ranking_policy["ranking_grid"]["fractions"]
    ]
    return pd.DataFrame(records)


def stratified_bootstrap_positions(
    target: np.ndarray,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample positive and negative validation rows separately."""

    negative_positions = np.flatnonzero(target == 0)
    positive_positions = np.flatnonzero(target == 1)
    sampled_negative = random_generator.choice(
        negative_positions,
        size=len(negative_positions),
        replace=True,
    )
    sampled_positive = random_generator.choice(
        positive_positions,
        size=len(positive_positions),
        replace=True,
    )
    combined = np.concatenate(
        [sampled_negative, sampled_positive]
    )
    return random_generator.permutation(combined)


def bootstrap_ranking_grid(
    target: np.ndarray,
    probability: np.ndarray,
    ranking_policy: dict[str, Any],
) -> pd.DataFrame:
    """Quantify top-k uncertainty with stratified bootstraps."""

    bootstrap_policy = ranking_policy["bootstrap"]
    iterations = int(bootstrap_policy["iterations"])
    random_generator = np.random.default_rng(
        int(bootstrap_policy["random_seed"])
    )
    fractions = [
        float(value)
        for value in ranking_policy["ranking_grid"]["fractions"]
    ]
    records: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        positions = stratified_bootstrap_positions(
            target,
            random_generator,
        )
        sampled_target = target[positions]
        sampled_probability = probability[positions]
        for fraction in fractions:
            metrics = top_fraction_metrics(
                sampled_target,
                sampled_probability,
                fraction,
            )
            records.append(
                {
                    "iteration": iteration,
                    "requested_fraction": fraction,
                    "selected_count": metrics["selected_count"],
                    "captured_positive_cases": (
                        metrics["captured_positive_cases"]
                    ),
                    "precision_at_k": metrics["precision_at_k"],
                    "capture_rate_at_k": (
                        metrics["capture_rate_at_k"]
                    ),
                    "lift_at_k": metrics["lift_at_k"],
                }
            )

        if iteration % 100 == 0:
            print(
                "Completed ranking bootstrap: "
                f"{iteration}/{iterations}"
            )

    return pd.DataFrame(records)


def add_bootstrap_intervals(
    ranking_grid: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ranking_policy: dict[str, Any],
) -> pd.DataFrame:
    """Add percentile intervals to every configured top-k row."""

    confidence_level = float(
        ranking_policy["bootstrap"]["confidence_level"]
    )
    tail = (1.0 - confidence_level) / 2.0
    lower_quantile = tail
    upper_quantile = 1.0 - tail
    records: list[dict[str, Any]] = []

    for fraction, group in bootstrap.groupby(
        "requested_fraction",
        sort=True,
    ):
        fraction_value = float(
            group["requested_fraction"].to_numpy(
                dtype=float
            )[0]
        )
        record: dict[str, Any] = {
            "requested_fraction": fraction_value
        }
        for metric in [
            "precision_at_k",
            "capture_rate_at_k",
            "lift_at_k",
            "captured_positive_cases",
        ]:
            record[f"{metric}_lower_95"] = float(
                group[metric].quantile(lower_quantile)
            )
            record[f"{metric}_upper_95"] = float(
                group[metric].quantile(upper_quantile)
            )
        records.append(record)

    return ranking_grid.merge(
        pd.DataFrame(records),
        on="requested_fraction",
        how="left",
        validate="one_to_one",
    )


def build_curve_points(
    target: np.ndarray,
    probability: np.ndarray,
    ranking_policy: dict[str, Any],
) -> pd.DataFrame:
    """Build long-form points for PR, ROC, gains, and lift curves."""

    curve_frames: list[pd.DataFrame] = []

    precision, recall, pr_thresholds = precision_recall_curve(
        target,
        probability,
    )
    pr_threshold_column = np.append(pr_thresholds, np.nan)
    curve_frames.append(
        pd.DataFrame(
            {
                "curve_type": "precision_recall",
                "point": np.arange(len(precision)),
                "x": recall,
                "y": precision,
                "threshold": pr_threshold_column,
                "x_label": "Recall",
                "y_label": "Precision",
            }
        )
    )

    false_positive_rate, true_positive_rate, roc_thresholds = (
        roc_curve(target, probability)
    )
    curve_frames.append(
        pd.DataFrame(
            {
                "curve_type": "roc",
                "point": np.arange(len(false_positive_rate)),
                "x": false_positive_rate,
                "y": true_positive_rate,
                "threshold": roc_thresholds,
                "x_label": "False-positive rate",
                "y_label": "True-positive rate",
            }
        )
    )

    number_of_points = int(
        ranking_policy["curves"]["gains_and_lift_points"]
    )
    fractions = np.linspace(
        1.0 / number_of_points,
        1.0,
        number_of_points,
    )
    gain_records: list[dict[str, Any]] = []
    lift_records: list[dict[str, Any]] = []
    for point, fraction in enumerate(fractions, start=1):
        metrics = top_fraction_metrics(
            target,
            probability,
            float(fraction),
        )
        gain_records.append(
            {
                "curve_type": "cumulative_gains",
                "point": point,
                "x": metrics["actual_selected_fraction"],
                "y": metrics["capture_rate_at_k"],
                "threshold": metrics[
                    "minimum_selected_probability"
                ],
                "x_label": "Workforce fraction reviewed",
                "y_label": "Attrition cases captured",
            }
        )
        lift_records.append(
            {
                "curve_type": "lift",
                "point": point,
                "x": metrics["actual_selected_fraction"],
                "y": metrics["lift_at_k"],
                "threshold": metrics[
                    "minimum_selected_probability"
                ],
                "x_label": "Workforce fraction reviewed",
                "y_label": "Lift",
            }
        )

    curve_frames.append(pd.DataFrame(gain_records))
    curve_frames.append(pd.DataFrame(lift_records))
    return pd.concat(curve_frames, ignore_index=True)


def reporting_confusion_matrix(
    target: np.ndarray,
    probability: np.ndarray,
    reporting_fraction: float,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Build a top-fraction reporting matrix without choosing a policy."""

    metrics = top_fraction_metrics(
        target,
        probability,
        reporting_fraction,
    )
    selected_count = int(metrics["selected_count"])
    selected = np.zeros(len(target), dtype=int)
    selected[ranked_positions(probability)[:selected_count]] = 1

    true_positive = int(((target == 1) & (selected == 1)).sum())
    false_positive = int(((target == 0) & (selected == 1)).sum())
    false_negative = int(((target == 1) & (selected == 0)).sum())
    true_negative = int(((target == 0) & (selected == 0)).sum())
    records = [
        {
            "actual_outcome": "No attrition",
            "selection_outcome": "Not selected",
            "count": true_negative,
        },
        {
            "actual_outcome": "No attrition",
            "selection_outcome": "Selected top 10%",
            "count": false_positive,
        },
        {
            "actual_outcome": "Attrition",
            "selection_outcome": "Not selected",
            "count": false_negative,
        },
        {
            "actual_outcome": "Attrition",
            "selection_outcome": "Selected top 10%",
            "count": true_positive,
        },
    ]
    counts = {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "selected_count": selected_count,
        "selected_probability_cutoff": float(
            metrics["minimum_selected_probability"]
        ),
    }
    return pd.DataFrame(records), counts


def build_summary(
    predictions: pd.DataFrame,
    ranking_grid: pd.DataFrame,
    confusion_counts: dict[str, int | float],
    ranking_policy: dict[str, Any],
    selected_method: str,
) -> pd.DataFrame:
    """Create one-row ranking summary for later consumers."""

    target = predictions["actual_attrition"].to_numpy(dtype=int)
    probability = predictions[
        "attrition_probability"
    ].to_numpy(dtype=float)
    prevalence = float(target.mean())
    reporting_fraction = float(
        ranking_policy["ranking_grid"]["reporting_fraction"]
    )
    reporting_row = ranking_grid.loc[
        np.isclose(
            ranking_grid["requested_fraction"],
            reporting_fraction,
        )
    ].iloc[0]

    return pd.DataFrame(
        [
            {
                "selected_calibration_method": selected_method,
                "snapshot_date": str(
                    predictions["snapshot_date"].iloc[0]
                ),
                "validation_rows": len(predictions),
                "positive_cases": int(target.sum()),
                "positive_rate": prevalence,
                "pr_auc": float(
                    average_precision_score(target, probability)
                ),
                "pr_auc_lift_over_no_skill": float(
                    average_precision_score(target, probability)
                    / prevalence
                ),
                "roc_auc": float(
                    roc_auc_score(target, probability)
                ),
                "reporting_fraction": reporting_fraction,
                "reporting_selected_count": int(
                    reporting_row["selected_count"]
                ),
                "reporting_precision": float(
                    reporting_row["precision_at_k"]
                ),
                "reporting_capture_rate": float(
                    reporting_row["capture_rate_at_k"]
                ),
                "reporting_lift": float(
                    reporting_row["lift_at_k"]
                ),
                "reporting_probability_cutoff": float(
                    confusion_counts[
                        "selected_probability_cutoff"
                    ]
                ),
                "true_negative": int(
                    confusion_counts["true_negative"]
                ),
                "false_positive": int(
                    confusion_counts["false_positive"]
                ),
                "false_negative": int(
                    confusion_counts["false_negative"]
                ),
                "true_positive": int(
                    confusion_counts["true_positive"]
                ),
                "final_operating_threshold_selected": bool(
                    ranking_policy["confusion_matrix"][
                        "final_operating_threshold_selected"
                    ]
                ),
            }
        ]
    )


def save_figure(
    figure: Figure,
    path: Path,
    ranking_policy: dict[str, Any],
) -> None:
    """Save one deterministic, presentation-ready figure."""

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=int(ranking_policy["figures"]["dpi"]),
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    plt.close(figure)


def create_figures(
    predictions: pd.DataFrame,
    curves: pd.DataFrame,
    confusion: pd.DataFrame,
    ranking_policy: dict[str, Any],
) -> None:
    """Create six ranking and classification figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = predictions["actual_attrition"].to_numpy(dtype=int)
    probability = predictions[
        "attrition_probability"
    ].to_numpy(dtype=float)
    baseline = float(target.mean())

    plt.style.use("seaborn-v0_8-whitegrid")

    pr = curves.loc[curves["curve_type"].eq("precision_recall")]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(pr["x"], pr["y"], linewidth=2, label="Selected model")
    axis.axhline(
        baseline,
        linestyle="--",
        color="#666666",
        label=f"No-skill baseline ({baseline:.1%})",
    )
    axis.set(
        title="Precision–Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, max(0.35, float(pr["y"].max()) * 1.05)),
    )
    axis.legend()
    save_figure(
        figure,
        FIGURE_DIR / "precision_recall_curve.png",
        ranking_policy,
    )

    roc = curves.loc[curves["curve_type"].eq("roc")]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(roc["x"], roc["y"], linewidth=2, label="Selected model")
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#666666",
        label="Random ordering",
    )
    axis.set(
        title="ROC Curve",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    save_figure(
        figure,
        FIGURE_DIR / "roc_curve.png",
        ranking_policy,
    )

    gains = curves.loc[curves["curve_type"].eq("cumulative_gains")]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        gains["x"],
        gains["y"],
        linewidth=2,
        label="Selected model",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#666666",
        label="Random selection",
    )
    axis.set(
        title="Cumulative Gains Curve",
        xlabel="Fraction of workforce reviewed",
        ylabel="Fraction of attrition cases captured",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    save_figure(
        figure,
        FIGURE_DIR / "cumulative_gains_curve.png",
        ranking_policy,
    )

    lift = curves.loc[curves["curve_type"].eq("lift")]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        lift["x"],
        lift["y"],
        linewidth=2,
        label="Selected model",
    )
    axis.axhline(
        1.0,
        linestyle="--",
        color="#666666",
        label="Random-selection lift",
    )
    axis.set(
        title="Lift Curve",
        ylabel="Lift over random selection",
        xlim=(0, 1),
    )
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_xlabel(
        "Fraction of workforce reviewed",
        labelpad=10,
    )
    axis.legend()
    save_figure(
        figure,
        FIGURE_DIR / "lift_curve.png",
        ranking_policy,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        probability[target == 0],
        bins=30,
        alpha=0.65,
        label="No attrition",
        density=True,
    )
    axis.hist(
        probability[target == 1],
        bins=30,
        alpha=0.65,
        label="Attrition",
        density=True,
    )
    axis.set(
        title="Calibrated Risk-Score Distribution",
        xlabel="Estimated 12-month attrition probability",
        ylabel="Density",
    )
    axis.legend()
    save_figure(
        figure,
        FIGURE_DIR / "score_distribution.png",
        ranking_policy,
    )

    matrix = (
        confusion.pivot(
            index="actual_outcome",
            columns="selection_outcome",
            values="count",
        )
        .reindex(
            index=["No attrition", "Attrition"],
            columns=["Not selected", "Selected top 10%"],
        )
        .to_numpy(dtype=int)
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > matrix.max() / 2
                    else "black"
                ),
                fontsize=12,
            )
    axis.set_xticks([0, 1], ["Not selected", "Selected top 10%"])
    axis.set_yticks([0, 1], ["No attrition", "Attrition"])
    axis.set_xlabel("Reporting selection")
    axis.set_ylabel("Actual outcome")
    axis.set_title("Top-Decile Reporting Confusion Matrix")
    figure.colorbar(image, ax=axis, label="Employees")
    save_figure(
        figure,
        FIGURE_DIR / "top_decile_confusion_matrix.png",
        ranking_policy,
    )


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> None:
    """Append one validation check."""

    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def validate_outputs(
    predictions: pd.DataFrame,
    selected_metrics: pd.Series,
    ranking_grid: pd.DataFrame,
    bootstrap: pd.DataFrame,
    curves: pd.DataFrame,
    confusion: pd.DataFrame,
    summary: pd.DataFrame,
    ranking_policy: dict[str, Any],
    test_rows: int,
    selected_method: str,
    probability_display_ready: bool,
) -> pd.DataFrame:
    """Validate ranking metrics, figures, and holdout boundaries."""

    checks: list[dict[str, Any]] = []
    validation_policy = ranking_policy["validation"]
    target = predictions["actual_attrition"].to_numpy(dtype=int)
    probability = predictions[
        "attrition_probability"
    ].to_numpy(dtype=float)

    add_check(
        checks,
        "Configured calibration method is selected",
        selected_method
        == str(ranking_policy["selected_calibration_method"]),
        selected_method,
        ranking_policy["selected_calibration_method"],
        "Ranking must use the method selected in Checkpoint 42.",
    )
    add_check(
        checks,
        "Selected scores are approved for probability display",
        probability_display_ready,
        probability_display_ready,
        True,
        "Checkpoint 42 must approve the selected probability scale.",
    )
    assignment_columns = pd.read_csv(
        ASSIGNMENT_PATH,
        nrows=1,
    ).columns
    add_check(
        checks,
        "Reserved test target remains unused",
        test_rows > 0
        and "attrition_next_12m" not in assignment_columns,
        f"{test_rows} reserved rows; 0 test targets used",
        "Reserved rows > 0 and target absent from assignments",
        "Only selected 2024 validation predictions are analyzed.",
    )
    observed_snapshots = sorted(
        predictions["snapshot_date"].unique().tolist()
    )
    add_check(
        checks,
        "Only the configured validation snapshot is present",
        observed_snapshots
        == [str(ranking_policy["validation_snapshot"])],
        observed_snapshots,
        [ranking_policy["validation_snapshot"]],
        "Ranking evidence comes from one chronological validation period.",
    )
    duplicate_keys = int(
        predictions.duplicated(
            ["employee_id", "snapshot_date"]
        ).sum()
    )
    add_check(
        checks,
        "Validation employee-snapshot keys are unique",
        duplicate_keys == 0,
        duplicate_keys,
        0,
        "Every employee receives one selected probability.",
    )
    add_check(
        checks,
        "Validation target is complete and binary",
        not bool(
            predictions["actual_attrition"].isna().to_numpy().any()
        )
        and set(np.unique(target)) == {0, 1},
        sorted(np.unique(target).tolist()),
        [0, 1],
        "Ranking metrics require known validation outcomes.",
    )
    finite_probability = bool(
        np.isfinite(probability).all()
        and probability.min()
        >= float(validation_policy["probability_minimum"])
        and probability.max()
        <= float(validation_policy["probability_maximum"])
    )
    add_check(
        checks,
        "Selected probabilities are finite and bounded",
        finite_probability,
        (
            f"min={probability.min():.6f}; "
            f"max={probability.max():.6f}"
        ),
        "All finite and inside [0, 1]",
        "The selected calibration must produce valid probabilities.",
    )
    minimum_rows = int(validation_policy["minimum_rows"])
    minimum_positives = int(
        validation_policy["minimum_positive_cases"]
    )
    add_check(
        checks,
        "Validation sample is sufficiently large",
        len(predictions) >= minimum_rows
        and int(target.sum()) >= minimum_positives,
        f"{len(predictions)} rows; {int(target.sum())} positives",
        f">= {minimum_rows} rows; >= {minimum_positives} positives",
        "Top-k and curve estimates need enough validation examples.",
    )
    fractions = ranking_grid["requested_fraction"].to_numpy(
        dtype=float
    )
    add_check(
        checks,
        "Ranking grid is complete and ordered",
        len(ranking_grid)
        == int(validation_policy["expected_grid_rows"])
        and bool(np.all(np.diff(fractions) > 0))
        and bool(np.isclose(fractions[-1], 1.0))
        and bool(
            np.isclose(
                fractions,
                float(
                    ranking_policy["ranking_grid"][
                        "reporting_fraction"
                    ]
                ),
            ).any()
        ),
        fractions.tolist(),
        "Configured increasing fractions including 10% and 100%",
        "The grid supports multiple intervention-budget scenarios.",
    )
    metric_values = ranking_grid[
        ["precision_at_k", "capture_rate_at_k"]
    ].to_numpy(dtype=float)
    valid_metric_ranges = bool(
        np.isfinite(metric_values).all()
        and (metric_values >= 0).all()
        and (metric_values <= 1).all()
        and (
            ranking_grid["lift_at_k"].to_numpy(dtype=float)
            >= 0
        ).all()
    )
    add_check(
        checks,
        "Top-k metrics remain in valid ranges",
        valid_metric_ranges,
        valid_metric_ranges,
        True,
        "Precision, capture, and lift cannot be negative.",
    )
    final_row = ranking_grid.iloc[-1]
    baseline_rate = float(target.mean())
    add_check(
        checks,
        "Full-population ranking returns the baseline",
        int(final_row["selected_count"]) == len(target)
        and np.isclose(final_row["capture_rate_at_k"], 1.0)
        and np.isclose(
            final_row["precision_at_k"],
            baseline_rate,
        )
        and np.isclose(final_row["lift_at_k"], 1.0),
        (
            f"count={int(final_row['selected_count'])}; "
            f"capture={float(final_row['capture_rate_at_k']):.6f}; "
            f"lift={float(final_row['lift_at_k']):.6f}"
        ),
        "All rows; capture=1; precision=baseline; lift=1",
        "Reviewing everyone must reproduce the population outcome rate.",
    )
    capture_values = ranking_grid[
        "capture_rate_at_k"
    ].to_numpy(dtype=float)
    add_check(
        checks,
        "Cumulative capture is monotonic",
        bool(np.all(np.diff(capture_values) >= 0)),
        bool(np.all(np.diff(capture_values) >= 0)),
        True,
        "Reviewing more employees cannot lose captured cases.",
    )
    expected_bootstrap_rows = (
        int(ranking_policy["bootstrap"]["iterations"])
        * len(ranking_grid)
    )
    add_check(
        checks,
        "Ranking bootstrap is complete",
        len(bootstrap) == expected_bootstrap_rows,
        len(bootstrap),
        expected_bootstrap_rows,
        "Every resample evaluates every configured top-k fraction.",
    )
    expected_curve_types = {
        "precision_recall",
        "roc",
        "cumulative_gains",
        "lift",
    }
    add_check(
        checks,
        "All configured curve datasets exist",
        set(curves["curve_type"]) == expected_curve_types
        and len(expected_curve_types)
        == int(validation_policy["expected_curve_types"]),
        sorted(curves["curve_type"].unique().tolist()),
        sorted(expected_curve_types),
        "Curve data must support all four ranking visualizations.",
    )
    pr_auc = float(
        average_precision_score(target, probability)
    )
    roc_auc = float(roc_auc_score(target, probability))
    tolerance = float(
        validation_policy["metric_reconciliation_tolerance"]
    )
    add_check(
        checks,
        "Ranking summary reconciles to selected predictions",
        abs(float(summary.iloc[0]["pr_auc"]) - pr_auc)
        <= tolerance
        and abs(float(summary.iloc[0]["roc_auc"]) - roc_auc)
        <= tolerance,
        (
            f"PR-AUC={pr_auc:.12f}; "
            f"ROC-AUC={roc_auc:.12f}"
        ),
        f"Difference <= {tolerance}",
        "Summary metrics must be recomputable from saved predictions.",
    )
    add_check(
        checks,
        "Checkpoint 42 ranking metrics reconcile",
        abs(float(selected_metrics["pr_auc"]) - pr_auc)
        <= tolerance
        and abs(float(selected_metrics["roc_auc"]) - roc_auc)
        <= tolerance,
        (
            f"PR difference="
            f"{abs(float(selected_metrics['pr_auc']) - pr_auc):.3g}; "
            f"ROC difference="
            f"{abs(float(selected_metrics['roc_auc']) - roc_auc):.3g}"
        ),
        f"Each <= {tolerance}",
        "Checkpoint 43 must use the exact selected calibrated scores.",
    )
    confusion_total = int(confusion["count"].sum())
    confusion_positive = int(
        confusion.loc[
            confusion["actual_outcome"].eq("Attrition"),
            "count",
        ].sum()
    )
    confusion_selected = int(
        confusion.loc[
            confusion["selection_outcome"].eq(
                "Selected top 10%"
            ),
            "count",
        ].sum()
    )
    add_check(
        checks,
        "Reporting confusion matrix reconciles",
        confusion_total == len(target)
        and confusion_positive == int(target.sum())
        and confusion_selected
        == int(summary.iloc[0]["reporting_selected_count"]),
        (
            f"total={confusion_total}; positives={confusion_positive}; "
            f"selected={confusion_selected}"
        ),
        (
            f"total={len(target)}; positives={int(target.sum())}; "
            f"selected={int(summary.iloc[0]['reporting_selected_count'])}"
        ),
        "The matrix must match the top-decile reporting row.",
    )
    add_check(
        checks,
        "No final operating threshold is selected",
        not bool(
            summary.iloc[0]["final_operating_threshold_selected"]
        ),
        bool(
            summary.iloc[0]["final_operating_threshold_selected"]
        ),
        False,
        "Checkpoint 46 will select policy using explicit business costs.",
    )
    expected_figure_names = set(
        ranking_policy["figures"]["expected_files"]
    )
    actual_figure_names = {
        path.name
        for path in FIGURE_DIR.glob("*.png")
        if path.stat().st_size > 0
    }
    add_check(
        checks,
        "All ranking figures were generated",
        expected_figure_names.issubset(actual_figure_names),
        sorted(actual_figure_names),
        sorted(expected_figure_names),
        "Every committed visualization specification must render.",
    )

    validation_table = pd.DataFrame(checks)
    failures = validation_table.loc[
        validation_table["status"].eq("FAIL")
    ]
    if not failures.empty:
        raise ValueError(
            "Ranking validation failed:\n"
            + failures.to_string(index=False)
        )
    return validation_table


def save_outputs(
    ranking_grid: pd.DataFrame,
    bootstrap: pd.DataFrame,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    confusion: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Save all Checkpoint 43 processed tables."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ranking_grid.to_csv(RANKING_METRICS_PATH, index=False)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    curves.to_csv(CURVE_PATH, index=False)
    confusion.to_csv(CONFUSION_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)


def print_results(
    ranking_grid: pd.DataFrame,
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Print the most important ranking evidence."""

    print("\nTOP-K RANKING METRICS")
    print(
        ranking_grid[
            [
                "requested_fraction",
                "selected_count",
                "captured_positive_cases",
                "precision_at_k",
                "capture_rate_at_k",
                "lift_at_k",
                "precision_at_k_lower_95",
                "precision_at_k_upper_95",
                "capture_rate_at_k_lower_95",
                "capture_rate_at_k_upper_95",
            ]
        ].to_string(index=False)
    )

    print("\nSELECTED MODEL RANKING SUMMARY")
    print(summary.to_string(index=False))

    print("\nTOP-DECILE REPORTING CONFUSION MATRIX")
    print(confusion.to_string(index=False))

    print("\nRANKING VALIDATION")
    print(validation.to_string(index=False))

    print(f"\nSaved ranking metrics: {RANKING_METRICS_PATH}")
    print(f"Saved figures: {FIGURE_DIR}")
    print("\nRETENTION RANKING ANALYSIS COMPLETED SUCCESSFULLY")


def main() -> None:
    """Run the complete Checkpoint 43 ranking analysis."""

    (
        predictions,
        selected_metrics,
        ranking_policy,
        test_rows,
        selected_method,
        probability_display_ready,
    ) = load_inputs()
    target = predictions["actual_attrition"].to_numpy(dtype=int)
    probability = predictions[
        "attrition_probability"
    ].to_numpy(dtype=float)

    ranking_grid = build_ranking_grid(
        target,
        probability,
        ranking_policy,
    )
    bootstrap = bootstrap_ranking_grid(
        target,
        probability,
        ranking_policy,
    )
    ranking_grid = add_bootstrap_intervals(
        ranking_grid,
        bootstrap,
        ranking_policy,
    )
    curves = build_curve_points(
        target,
        probability,
        ranking_policy,
    )
    reporting_fraction = float(
        ranking_policy["confusion_matrix"][
            "reporting_fraction"
        ]
    )
    confusion, confusion_counts = reporting_confusion_matrix(
        target,
        probability,
        reporting_fraction,
    )
    summary = build_summary(
        predictions,
        ranking_grid,
        confusion_counts,
        ranking_policy,
        selected_method,
    )
    create_figures(
        predictions,
        curves,
        confusion,
        ranking_policy,
    )
    validation = validate_outputs(
        predictions,
        selected_metrics,
        ranking_grid,
        bootstrap,
        curves,
        confusion,
        summary,
        ranking_policy,
        test_rows,
        selected_method,
        probability_display_ready,
    )
    save_outputs(
        ranking_grid,
        bootstrap,
        summary,
        curves,
        confusion,
        validation,
    )
    print_results(
        ranking_grid,
        summary,
        confusion,
        validation,
    )


if __name__ == "__main__":
    main()
