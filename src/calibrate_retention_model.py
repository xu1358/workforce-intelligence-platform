"""Calibrate the selected Version 2 retention model without test access.

The base Logistic Regression produces out-of-fold probabilities on the
2023 training period. Sigmoid and isotonic mappings learn only from those
honest training predictions. The 2024 validation period compares the
methods, while the reserved 2025 target remains untouched.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from compare_retention_models_v2 import (
    build_candidate_model,
    fit_candidate,
    top_fraction_metrics,
)
from retention_feature_policy import (
    load_feature_policy,
    normalize_feature_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "retention_multi_snapshot.csv"
ASSIGNMENT_PATH = PROCESSED_DIR / "model_split_assignments.csv"
MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "model_comparison_v2.yaml"
)
CALIBRATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "retention_calibration.yaml"
)
MODEL_SELECTION_PATH = (
    PROCESSED_DIR / "model_selection_decision_v2.csv"
)
CHECKPOINT_41_PREDICTION_PATH = (
    PROCESSED_DIR / "retention_validation_predictions_v2.csv"
)

METRICS_PATH = (
    PROCESSED_DIR / "retention_calibration_metrics.csv"
)
RELIABILITY_PATH = (
    PROCESSED_DIR / "retention_reliability_bins.csv"
)
BOOTSTRAP_PATH = (
    PROCESSED_DIR / "retention_calibration_bootstrap.csv"
)
DIFFERENCE_PATH = (
    PROCESSED_DIR / "retention_calibration_differences.csv"
)
SELECTION_PATH = (
    PROCESSED_DIR / "retention_calibration_selection.csv"
)
PREDICTION_PATH = (
    PROCESSED_DIR / "retention_calibration_predictions.csv"
)
FOLD_PATH = (
    PROCESSED_DIR / "retention_calibration_folds.csv"
)
VALIDATION_PATH = (
    PROCESSED_DIR / "retention_calibration_validation.csv"
)


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
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    int,
    str,
    pd.DataFrame,
]:
    """Load development rows while structurally excluding test targets."""

    required_paths = [
        DATA_PATH,
        ASSIGNMENT_PATH,
        MODEL_CONFIG_PATH,
        CALIBRATION_CONFIG_PATH,
        MODEL_SELECTION_PATH,
        CHECKPOINT_41_PREDICTION_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 42 input files:\n"
            + "\n".join(missing)
        )

    assignments = pd.read_csv(ASSIGNMENT_PATH)
    feature_policy = load_feature_policy()
    target_column = str(feature_policy["target_column"])
    if target_column in assignments.columns:
        raise ValueError(
            "Split assignments must not contain the attrition target."
        )

    test_rows = int(assignments["primary_split"].eq("test").sum())
    development_assignments = assignments.loc[
        assignments["primary_split"].isin(["train", "validation"])
    ].copy()

    dataset = pd.read_csv(DATA_PATH)
    development = dataset.merge(
        development_assignments,
        on=["employee_id", "snapshot_date"],
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

    selection = pd.read_csv(MODEL_SELECTION_PATH)
    selected_rows = selection.loc[selection["selected_model"].eq(True)]
    if len(selected_rows) != 1:
        raise ValueError(
            "Checkpoint 41 must select exactly one provisional model."
        )
    selected_model = str(selected_rows.iloc[0]["model"])

    checkpoint_41_predictions = pd.read_csv(
        CHECKPOINT_41_PREDICTION_PATH
    )
    checkpoint_41_predictions = checkpoint_41_predictions.loc[
        checkpoint_41_predictions["model"].eq(selected_model)
    ].copy()

    return (
        train,
        validation,
        load_yaml(MODEL_CONFIG_PATH),
        load_yaml(CALIBRATION_CONFIG_PATH),
        feature_policy,
        test_rows,
        selected_model,
        checkpoint_41_predictions,
    )


def clip_probability(
    probability: np.ndarray,
    calibration_policy: dict[str, Any],
) -> np.ndarray:
    """Clip probabilities only where logarithms require open bounds."""

    epsilon = float(
        calibration_policy["calibration_training"][
            "probability_clip"
        ]
    )
    return np.clip(probability, epsilon, 1.0 - epsilon)


def probability_logit(
    probability: np.ndarray,
    calibration_policy: dict[str, Any],
) -> np.ndarray:
    """Convert probability to one log-odds feature."""

    clipped = clip_probability(probability, calibration_policy)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_calibration_methods(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_policy: dict[str, Any],
    calibration_policy: dict[str, Any],
    feature_policy: dict[str, Any],
    selected_model: str,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    pd.DataFrame,
]:
    """Fit out-of-fold calibrators and score validation probabilities."""

    target_column = str(feature_policy["target_column"])
    train_features = normalize_feature_frame(train, feature_policy)
    validation_features = normalize_feature_frame(
        validation,
        feature_policy,
    )
    train_target = train[target_column].astype(int).to_numpy()

    training_policy = calibration_policy["calibration_training"]
    folds = int(training_policy["folds"])
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=bool(training_policy["shuffle"]),
        random_state=int(training_policy["random_seed"]),
    )
    oof_probability = np.full(len(train), np.nan, dtype=float)
    fold_records: list[dict[str, Any]] = []

    for fold, (fit_index, calibration_index) in enumerate(
        splitter.split(train_features, train_target),
        start=1,
    ):
        print(
            "Generating out-of-fold probabilities: "
            f"fold {fold}/{folds}"
        )
        model = build_candidate_model(
            selected_model,
            model_policy,
            feature_policy,
        )
        model = fit_candidate(
            model,
            selected_model,
            train_features.iloc[fit_index],
            train_target[fit_index],
            model_policy,
        )
        fold_probability = np.asarray(
            model.predict_proba(
                train_features.iloc[calibration_index]
            )[:, 1],
            dtype=float,
        )
        oof_probability[calibration_index] = fold_probability

        fit_employees = set(
            train.iloc[fit_index]["employee_id"].astype(int)
        )
        calibration_employees = set(
            train.iloc[calibration_index][
                "employee_id"
            ].astype(int)
        )
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": len(fit_index),
                "calibration_rows": len(calibration_index),
                "fit_positive_cases": int(
                    train_target[fit_index].sum()
                ),
                "calibration_positive_cases": int(
                    train_target[calibration_index].sum()
                ),
                "employee_overlap": len(
                    fit_employees & calibration_employees
                ),
                "minimum_probability": float(
                    fold_probability.min()
                ),
                "maximum_probability": float(
                    fold_probability.max()
                ),
            }
        )

    if np.isnan(oof_probability).any():
        raise ValueError(
            "Every training row must receive one out-of-fold probability."
        )

    sigmoid = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=int(training_policy["random_seed"]),
    )
    sigmoid.fit(
        probability_logit(
            oof_probability,
            calibration_policy,
        ),
        train_target,
    )

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    isotonic.fit(oof_probability, train_target)

    full_model = build_candidate_model(
        selected_model,
        model_policy,
        feature_policy,
    )
    full_model = fit_candidate(
        full_model,
        selected_model,
        train_features,
        train_target,
        model_policy,
    )
    raw_validation_probability = np.asarray(
        full_model.predict_proba(validation_features)[:, 1],
        dtype=float,
    )

    method_probabilities = {
        "Uncalibrated": raw_validation_probability,
        "Sigmoid": np.asarray(
            sigmoid.predict_proba(
                probability_logit(
                    raw_validation_probability,
                    calibration_policy,
                )
            )[:, 1],
            dtype=float,
        ),
        "Isotonic": np.asarray(
            isotonic.predict(raw_validation_probability),
            dtype=float,
        ),
    }

    return (
        method_probabilities,
        oof_probability,
        pd.DataFrame(fold_records),
    )


def reliability_bins(
    target: np.ndarray,
    probability: np.ndarray,
    method: str,
    number_of_bins: int,
) -> pd.DataFrame:
    """Create deterministic equal-frequency reliability bins."""

    ordered_positions = np.argsort(probability, kind="mergesort")
    position_groups = np.array_split(
        ordered_positions,
        number_of_bins,
    )
    records: list[dict[str, Any]] = []

    for bin_number, positions in enumerate(position_groups, start=1):
        if len(positions) == 0:
            continue
        bin_probability = probability[positions]
        bin_target = target[positions]
        mean_probability = float(bin_probability.mean())
        observed_rate = float(bin_target.mean())
        records.append(
            {
                "method": method,
                "bin": bin_number,
                "records": len(positions),
                "positive_cases": int(bin_target.sum()),
                "minimum_probability": float(
                    bin_probability.min()
                ),
                "maximum_probability": float(
                    bin_probability.max()
                ),
                "mean_predicted_probability": mean_probability,
                "observed_attrition_rate": observed_rate,
                "calibration_gap": (
                    mean_probability - observed_rate
                ),
                "absolute_calibration_gap": abs(
                    mean_probability - observed_rate
                ),
            }
        )

    return pd.DataFrame(records)


def calibration_error(
    reliability: pd.DataFrame,
) -> tuple[float, float]:
    """Return expected and maximum calibration error."""

    total = float(reliability["records"].sum())
    ece = float(
        (
            reliability["records"]
            * reliability["absolute_calibration_gap"]
        ).sum()
        / total
    )
    mce = float(reliability["absolute_calibration_gap"].max())
    return ece, mce


def calculate_calibration_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    method: str,
    calibration_policy: dict[str, Any],
    model_policy: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate probability, calibration, and ranking metrics."""

    number_of_bins = int(
        calibration_policy["reliability"]["bins"]
    )
    reliability = reliability_bins(
        target,
        probability,
        method,
        number_of_bins,
    )
    ece, mce = calibration_error(reliability)
    observed_rate = float(target.mean())
    mean_probability = float(probability.mean())
    null_probability = np.full(len(target), observed_rate)
    null_brier = float(
        brier_score_loss(target, null_probability)
    )
    brier = float(brier_score_loss(target, probability))
    ranking = top_fraction_metrics(
        target,
        probability,
        float(model_policy["evaluation"]["top_fraction"]),
    )

    metrics = {
        "method": method,
        "rows": len(target),
        "positive_cases": int(target.sum()),
        "observed_attrition_rate": observed_rate,
        "mean_predicted_probability": mean_probability,
        "mean_calibration_gap": (
            mean_probability - observed_rate
        ),
        "absolute_mean_calibration_gap": abs(
            mean_probability - observed_rate
        ),
        "brier_score": brier,
        "brier_skill_score": (
            1.0 - brier / null_brier
            if null_brier > 0
            else 0.0
        ),
        "log_loss": float(
            log_loss(
                target,
                clip_probability(
                    probability,
                    calibration_policy,
                ),
            )
        ),
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "pr_auc": float(
            average_precision_score(target, probability)
        ),
        "roc_auc": float(roc_auc_score(target, probability)),
        **ranking,
    }
    return metrics, reliability


def compare_methods(
    validation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    calibration_policy: dict[str, Any],
    model_policy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every calibration method on the 2024 validation period."""

    target_column = str(feature_policy["target_column"])
    target = validation[target_column].astype(int).to_numpy()
    metric_records: list[dict[str, Any]] = []
    reliability_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for method in calibration_policy["methods"]:
        probability = probabilities[str(method)]
        metrics, reliability = calculate_calibration_metrics(
            target,
            probability,
            str(method),
            calibration_policy,
            model_policy,
        )
        metric_records.append(metrics)
        reliability_frames.append(reliability)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "employee_id": validation[
                        "employee_id"
                    ].astype(int).to_numpy(),
                    "snapshot_date": validation[
                        "snapshot_date"
                    ].astype(str).to_numpy(),
                    "actual_attrition": target,
                    "method": str(method),
                    "attrition_probability": probability,
                }
            )
        )

    return (
        pd.DataFrame(metric_records),
        pd.concat(reliability_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def stratified_bootstrap_positions(
    target: np.ndarray,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample positives and negatives separately with replacement."""

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


def bootstrap_calibration_metrics(
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    calibration_policy: dict[str, Any],
    model_policy: dict[str, Any],
) -> pd.DataFrame:
    """Run paired stratified validation bootstrap samples."""

    bootstrap_policy = calibration_policy["bootstrap"]
    iterations = int(bootstrap_policy["iterations"])
    random_generator = np.random.default_rng(
        int(bootstrap_policy["random_seed"])
    )
    records: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        positions = stratified_bootstrap_positions(
            target,
            random_generator,
        )
        sampled_target = target[positions]
        for method in calibration_policy["methods"]:
            sampled_probability = probabilities[str(method)][
                positions
            ]
            metrics, _ = calculate_calibration_metrics(
                sampled_target,
                sampled_probability,
                str(method),
                calibration_policy,
                model_policy,
            )
            records.append(
                {
                    "iteration": iteration,
                    "method": str(method),
                    "brier_score": metrics["brier_score"],
                    "log_loss": metrics["log_loss"],
                    "expected_calibration_error": (
                        metrics["expected_calibration_error"]
                    ),
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                }
            )

        if iteration % 100 == 0:
            print(
                "Completed paired calibration bootstrap: "
                f"{iteration}/{iterations}"
            )

    return pd.DataFrame(records)


def confidence_bounds(
    confidence_level: float,
) -> tuple[float, float]:
    """Return lower and upper quantiles."""

    tail = (1.0 - confidence_level) / 2.0
    return tail, 1.0 - tail


def add_bootstrap_intervals(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    calibration_policy: dict[str, Any],
) -> pd.DataFrame:
    """Add method-level uncertainty intervals."""

    lower_quantile, upper_quantile = confidence_bounds(
        float(
            calibration_policy["bootstrap"][
                "confidence_level"
            ]
        )
    )
    interval_records: list[dict[str, Any]] = []

    for method, group in bootstrap.groupby("method", sort=False):
        record: dict[str, Any] = {"method": method}
        for metric in [
            "brier_score",
            "log_loss",
            "expected_calibration_error",
            "pr_auc",
            "roc_auc",
        ]:
            record[f"{metric}_lower_95"] = float(
                group[metric].quantile(lower_quantile)
            )
            record[f"{metric}_upper_95"] = float(
                group[metric].quantile(upper_quantile)
            )
        interval_records.append(record)

    return metrics.merge(
        pd.DataFrame(interval_records),
        on="method",
        how="left",
        validate="one_to_one",
    )


def pairwise_brier_differences(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    calibration_policy: dict[str, Any],
) -> pd.DataFrame:
    """Compare Brier scores with paired uncertainty intervals."""

    lower_quantile, upper_quantile = confidence_bounds(
        float(
            calibration_policy["bootstrap"][
                "confidence_level"
            ]
        )
    )
    observed_values = metrics["brier_score"].to_numpy(
        dtype=float
    )
    observed = dict(
        zip(
            metrics["method"].astype(str).tolist(),
            observed_values.tolist(),
        )
    )
    pivot = bootstrap.pivot(
        index="iteration",
        columns="method",
        values="brier_score",
    )
    records: list[dict[str, Any]] = []

    for method_a, method_b in combinations(
        calibration_policy["methods"],
        2,
    ):
        difference = pivot[str(method_a)] - pivot[str(method_b)]
        lower = float(difference.quantile(lower_quantile))
        upper = float(difference.quantile(upper_quantile))
        if lower > 0:
            conclusion = f"{method_b} has lower Brier score"
        elif upper < 0:
            conclusion = f"{method_a} has lower Brier score"
        else:
            conclusion = "Inconclusive"
        records.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "observed_brier_difference_a_minus_b": (
                    observed[str(method_a)]
                    - observed[str(method_b)]
                ),
                "bootstrap_mean_difference": float(
                    difference.mean()
                ),
                "difference_lower_95": lower,
                "difference_upper_95": upper,
                "interval_includes_zero": lower <= 0 <= upper,
                "interpretation": (
                    "Positive means method B has the lower "
                    "Brier score."
                ),
                "conclusion": conclusion,
            }
        )

    return pd.DataFrame(records)


def brier_difference_interval(
    method_a: str,
    method_b: str,
    differences: pd.DataFrame,
) -> tuple[float, float]:
    """Return the interval for method A minus method B."""

    if method_a == method_b:
        return 0.0, 0.0

    direct = differences.loc[
        differences["method_a"].eq(method_a)
        & differences["method_b"].eq(method_b)
    ]
    if not direct.empty:
        row = direct.iloc[0]
        return (
            float(row["difference_lower_95"]),
            float(row["difference_upper_95"]),
        )

    reverse = differences.loc[
        differences["method_a"].eq(method_b)
        & differences["method_b"].eq(method_a)
    ]
    if reverse.empty:
        raise ValueError("Missing paired Brier comparison.")
    row = reverse.iloc[0]
    return (
        -float(row["difference_upper_95"]),
        -float(row["difference_lower_95"]),
    )


def choose_calibration_method(
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    calibration_policy: dict[str, Any],
) -> pd.DataFrame:
    """Select one validation-stage calibration method."""

    selection_policy = calibration_policy["selection"]
    tolerance = float(
        selection_policy["practical_brier_tolerance"]
    )
    minimum_improvement = float(
        selection_policy["minimum_brier_improvement"]
    )
    indexed = metrics.set_index("method")
    uncalibrated_brier = float(
        indexed.loc["Uncalibrated", "brier_score"]
    )
    best_method = str(indexed["brier_score"].idxmin())
    best_brier = float(indexed.loc[best_method, "brier_score"])
    best_improvement = uncalibrated_brier - best_brier
    improvement_lower, improvement_upper = (
        brier_difference_interval(
            "Uncalibrated",
            best_method,
            differences,
        )
    )
    clear_calibration_gain = (
        best_method != "Uncalibrated"
        and best_improvement >= minimum_improvement
        and improvement_lower > 0
    )
    complexity_order = {
        method: rank
        for rank, method in enumerate(
            selection_policy["complexity_preference"]
        )
    }
    records: list[dict[str, Any]] = []

    for method in calibration_policy["methods"]:
        method_name = str(method)
        brier = float(indexed.loc[method_name, "brier_score"])
        gap = brier - best_brier
        improvement = uncalibrated_brier - brier
        lower, upper = brier_difference_interval(
            "Uncalibrated",
            method_name,
            differences,
        )
        within_tolerance = gap <= tolerance
        if clear_calibration_gain:
            eligible = (
                method_name != "Uncalibrated"
                and within_tolerance
            )
        else:
            eligible = method_name == "Uncalibrated"
        records.append(
            {
                "method": method_name,
                "validation_brier_score": brier,
                "best_observed_method": best_method,
                "gap_from_best_brier": gap,
                "brier_improvement_vs_uncalibrated": improvement,
                "improvement_lower_95": lower,
                "improvement_upper_95": upper,
                "clear_calibration_gain": clear_calibration_gain,
                "within_practical_tolerance": within_tolerance,
                "eligible_for_selection": eligible,
                "complexity_rank": complexity_order[method_name],
            }
        )

    decision = pd.DataFrame(records)
    eligible = decision.loc[
        decision["eligible_for_selection"]
    ].sort_values("complexity_rank")
    if eligible.empty:
        raise ValueError("No calibration method is eligible.")
    selected_method = str(eligible.iloc[0]["method"])
    decision["selected_method"] = decision["method"].eq(
        selected_method
    )

    selected_metrics = indexed.loc[selected_method]
    dashboard_policy = calibration_policy[
        "dashboard_probability_policy"
    ]
    probability_ready = (
        float(
            selected_metrics["expected_calibration_error"]
        )
        <= float(dashboard_policy["maximum_selected_ece"])
        and float(
            selected_metrics[
                "absolute_mean_calibration_gap"
            ]
        )
        <= float(
            dashboard_policy[
                "maximum_absolute_mean_calibration_gap"
            ]
        )
    )
    display_label = (
        str(dashboard_policy["probability_label"])
        if probability_ready
        else str(dashboard_policy["fallback_label"])
    )
    decision["probability_display_ready"] = np.where(
        decision["selected_method"],
        probability_ready,
        False,
    )
    decision["recommended_dashboard_label"] = np.where(
        decision["selected_method"],
        display_label,
        "Not selected.",
    )
    decision["selection_basis"] = np.where(
        decision["selected_method"],
        (
            "Selected using validation Brier score, paired "
            "uncertainty, practical tolerance, and method simplicity."
        ),
        "Not selected.",
    )
    return decision


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
    train: pd.DataFrame,
    validation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    oof_probability: np.ndarray,
    folds: pd.DataFrame,
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    differences: pd.DataFrame,
    selection: pd.DataFrame,
    predictions: pd.DataFrame,
    checkpoint_41_predictions: pd.DataFrame,
    calibration_policy: dict[str, Any],
    feature_policy: dict[str, Any],
    test_rows: int,
    selected_model: str,
) -> pd.DataFrame:
    """Validate leakage boundaries, calibration, and saved outputs."""

    checks: list[dict[str, Any]] = []
    validation_policy = calibration_policy["validation"]
    target_column = str(feature_policy["target_column"])
    methods = [str(method) for method in calibration_policy["methods"]]

    add_check(
        checks,
        "Checkpoint 41 selected the configured base model",
        selected_model
        == str(calibration_policy["selected_base_model"]),
        selected_model,
        calibration_policy["selected_base_model"],
        "Calibration must use the provisional model already selected.",
    )
    add_check(
        checks,
        "Reserved test target remains unused",
        test_rows > 0
        and target_column not in pd.read_csv(
            ASSIGNMENT_PATH,
            nrows=1,
        ).columns,
        f"{test_rows} reserved rows; 0 test targets used",
        "Reserved rows > 0 and target absent from assignments",
        "Only train and validation assignments were joined to targets.",
    )
    train_keys = set(
        zip(
            train["employee_id"],
            train["snapshot_date"].astype(str),
        )
    )
    validation_keys = set(
        zip(
            validation["employee_id"],
            validation["snapshot_date"].astype(str),
        )
    )
    add_check(
        checks,
        "Primary train and validation rows are separate",
        not (train_keys & validation_keys),
        len(train_keys & validation_keys),
        0,
        "Calibration training and evaluation use different snapshots.",
    )
    add_check(
        checks,
        "Training population is sufficiently large",
        len(train)
        >= int(validation_policy["minimum_training_rows"]),
        len(train),
        f">= {validation_policy['minimum_training_rows']}",
        "Out-of-fold calibration requires a useful training sample.",
    )
    validation_positives = int(
        validation[target_column].astype(int).sum()
    )
    add_check(
        checks,
        "Validation population and positives are sufficient",
        len(validation)
        >= int(validation_policy["minimum_validation_rows"])
        and validation_positives
        >= int(
            validation_policy[
                "minimum_validation_positive_cases"
            ]
        ),
        f"{len(validation)} rows; {validation_positives} positives",
        (
            f">= {validation_policy['minimum_validation_rows']} rows; "
            f">= {validation_policy['minimum_validation_positive_cases']} "
            "positives"
        ),
        "Brier and reliability estimates require enough observations.",
    )
    add_check(
        checks,
        "Every training row has one out-of-fold probability",
        bool(
            len(oof_probability) == len(train)
            and np.isfinite(oof_probability).all()
        ),
        len(oof_probability),
        len(train),
        "Calibration mappings learn from predictions made out of fold.",
    )
    expected_folds = int(
        calibration_policy["calibration_training"]["folds"]
    )
    add_check(
        checks,
        "Calibration folds are complete and employee-disjoint",
        len(folds) == expected_folds
        and int(folds["employee_overlap"].max()) == 0,
        (
            f"{len(folds)} folds; "
            f"{int(folds['employee_overlap'].max())} max overlap"
        ),
        f"{expected_folds} folds; 0 overlap",
        "No row may calibrate a prediction from a model that fitted it.",
    )
    minimum_fold_positives = int(
        calibration_policy["calibration_training"][
            "minimum_positive_cases_per_fold"
        ]
    )
    add_check(
        checks,
        "Every calibration fold has enough positive cases",
        int(folds["calibration_positive_cases"].min())
        >= minimum_fold_positives,
        int(folds["calibration_positive_cases"].min()),
        f">= {minimum_fold_positives}",
        "Sigmoid and isotonic mappings need positive examples.",
    )
    finite_probabilities = all(
        np.isfinite(probability).all()
        and float(probability.min())
        >= float(validation_policy["probability_minimum"])
        and float(probability.max())
        <= float(validation_policy["probability_maximum"])
        for probability in probabilities.values()
    )
    add_check(
        checks,
        "Validation probabilities are finite and bounded",
        finite_probabilities,
        [
            bool(np.isfinite(probability).all())
            for probability in probabilities.values()
        ],
        "All True and inside [0, 1]",
        "Every method must output a valid probability.",
    )
    checkpoint_41_sorted = checkpoint_41_predictions.sort_values(
        ["employee_id", "snapshot_date"]
    )
    current_uncalibrated = predictions.loc[
        predictions["method"].eq("Uncalibrated")
    ].sort_values(["employee_id", "snapshot_date"])
    reconciliation_difference = float(
        np.max(
            np.abs(
                checkpoint_41_sorted[
                    "attrition_probability"
                ].to_numpy()
                - current_uncalibrated[
                    "attrition_probability"
                ].to_numpy()
            )
        )
    )
    add_check(
        checks,
        "Uncalibrated scores reproduce Checkpoint 41",
        len(checkpoint_41_sorted) == len(current_uncalibrated)
        and reconciliation_difference <= 1e-12,
        reconciliation_difference,
        "<= 1e-12",
        "Calibration starts from the exact provisional model scores.",
    )
    add_check(
        checks,
        "Every configured method was evaluated",
        set(metrics["method"]) == set(methods)
        and len(metrics)
        == int(validation_policy["expected_methods"]),
        sorted(metrics["method"].tolist()),
        sorted(methods),
        "Uncalibrated, sigmoid, and isotonic must all be compared.",
    )
    expected_bins = (
        len(methods)
        * int(calibration_policy["reliability"]["bins"])
    )
    bin_counts = reliability.groupby("method")["records"].sum()
    add_check(
        checks,
        "Reliability bins cover every validation row",
        bool(
            len(reliability) == expected_bins
            and np.asarray(
                bin_counts == len(validation)
            ).all()
        ),
        (
            f"{len(reliability)} bins; "
            f"coverage={bin_counts.to_dict()}"
        ),
        f"{expected_bins} bins; {len(validation)} rows per method",
        "Equal-frequency bins support the reliability diagram.",
    )
    metric_values = metrics[
        [
            "brier_score",
            "expected_calibration_error",
            "pr_auc",
            "roc_auc",
        ]
    ].to_numpy(dtype=float)
    valid_metrics = bool(
        np.isfinite(metric_values).all()
        and (metric_values >= 0).all()
        and (metric_values <= 1).all()
    )
    add_check(
        checks,
        "Calibration and ranking metrics are valid",
        bool(valid_metrics),
        bool(valid_metrics),
        True,
        "Probability and ranking metrics must remain in valid ranges.",
    )
    expected_bootstrap_rows = (
        int(calibration_policy["bootstrap"]["iterations"])
        * len(methods)
    )
    add_check(
        checks,
        "Paired calibration bootstraps completed",
        len(bootstrap) == expected_bootstrap_rows,
        len(bootstrap),
        expected_bootstrap_rows,
        "Every resample evaluates all methods on the same employees.",
    )
    expected_pairs = len(list(combinations(methods, 2)))
    add_check(
        checks,
        "Every calibration pair has a Brier interval",
        len(differences) == expected_pairs,
        len(differences),
        expected_pairs,
        "Paired intervals quantify uncertainty in score differences.",
    )
    selected_count = int(selection["selected_method"].sum())
    add_check(
        checks,
        "Exactly one calibration method is selected",
        selected_count == 1,
        selected_count,
        1,
        "The decision follows committed validation-stage rules.",
    )
    metric_index = metrics.set_index("method")
    selected_method = str(
        selection.loc[
            selection["selected_method"],
            "method",
        ].iloc[0]
    )
    add_check(
        checks,
        "Selected calibration does not worsen Brier score",
        float(metric_index.loc[selected_method, "brier_score"])
        <= float(metric_index.loc["Uncalibrated", "brier_score"]),
        float(metric_index.loc[selected_method, "brier_score"]),
        (
            "<= "
            f"{float(metric_index.loc['Uncalibrated', 'brier_score']):.6f}"
        ),
        "The selected mapping must improve or preserve probability quality.",
    )
    ranking_change = float(
        (
            metric_index["pr_auc"]
            - float(metric_index.loc["Uncalibrated", "pr_auc"])
        )
        .abs()
        .max()
    )
    add_check(
        checks,
        "Calibration preserves ranking performance",
        ranking_change
        <= float(
            validation_policy["maximum_ranking_metric_change"]
        ),
        ranking_change,
        (
            "<= "
            f"{validation_policy['maximum_ranking_metric_change']}"
        ),
        "Calibration should adjust probabilities, not replace ranking.",
    )
    expected_prediction_rows = len(validation) * len(methods)
    add_check(
        checks,
        "Saved predictions have complete method coverage",
        len(predictions) == expected_prediction_rows,
        len(predictions),
        expected_prediction_rows,
        "Every validation employee needs one score per method.",
    )

    validation_table = pd.DataFrame(checks)
    failures = validation_table.loc[
        validation_table["status"].eq("FAIL")
    ]
    if not failures.empty:
        raise ValueError(
            "Calibration validation failed:\n"
            + failures.to_string(index=False)
        )
    return validation_table


def save_outputs(
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    differences: pd.DataFrame,
    selection: pd.DataFrame,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Save all Checkpoint 42 processed outputs."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    reliability.to_csv(RELIABILITY_PATH, index=False)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    differences.to_csv(DIFFERENCE_PATH, index=False)
    selection.to_csv(SELECTION_PATH, index=False)
    predictions.to_csv(PREDICTION_PATH, index=False)
    folds.to_csv(FOLD_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)


def print_results(
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    selection: pd.DataFrame,
    folds: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Print the most important calibration evidence."""

    print("\nOUT-OF-FOLD CALIBRATION FOLDS")
    print(
        folds[
            [
                "fold",
                "fit_rows",
                "calibration_rows",
                "calibration_positive_cases",
                "employee_overlap",
            ]
        ].to_string(index=False)
    )

    print("\nTEMPORAL VALIDATION CALIBRATION COMPARISON")
    display_columns = [
        "method",
        "brier_score",
        "brier_score_lower_95",
        "brier_score_upper_95",
        "log_loss",
        "expected_calibration_error",
        "mean_predicted_probability",
        "observed_attrition_rate",
        "pr_auc",
        "roc_auc",
        "top_decile_lift",
        "selected_method",
    ]
    print(metrics[display_columns].to_string(index=False))

    print("\nPAIRED BRIER SCORE DIFFERENCES")
    print(
        differences[
            [
                "method_a",
                "method_b",
                "observed_brier_difference_a_minus_b",
                "difference_lower_95",
                "difference_upper_95",
                "interval_includes_zero",
                "conclusion",
            ]
        ].to_string(index=False)
    )

    print("\nCALIBRATION METHOD SELECTION")
    print(
        selection[
            [
                "method",
                "validation_brier_score",
                "brier_improvement_vs_uncalibrated",
                "gap_from_best_brier",
                "eligible_for_selection",
                "selected_method",
                "probability_display_ready",
                "recommended_dashboard_label",
            ]
        ].to_string(index=False)
    )
    selected_method = str(
        selection.loc[
            selection["selected_method"],
            "method",
        ].iloc[0]
    )
    dashboard_label = str(
        selection.loc[
            selection["selected_method"],
            "recommended_dashboard_label",
        ].iloc[0]
    )
    print(f"\nSelected calibration method: {selected_method}")
    print(f"Recommended dashboard label: {dashboard_label}")

    print("\nCALIBRATION VALIDATION")
    print(validation.to_string(index=False))
    print(f"\nSaved calibration metrics: {METRICS_PATH}")
    print("\nRETENTION CALIBRATION COMPLETED SUCCESSFULLY")


def main() -> None:
    """Run the complete Checkpoint 42 calibration analysis."""

    (
        train,
        validation_rows,
        model_policy,
        calibration_policy,
        feature_policy,
        test_rows,
        selected_model,
        checkpoint_41_predictions,
    ) = load_inputs()
    (
        probabilities,
        oof_probability,
        folds,
    ) = fit_calibration_methods(
        train,
        validation_rows,
        model_policy,
        calibration_policy,
        feature_policy,
        selected_model,
    )
    metrics, reliability, predictions = compare_methods(
        validation_rows,
        probabilities,
        calibration_policy,
        model_policy,
        feature_policy,
    )
    target_column = str(feature_policy["target_column"])
    validation_target = (
        validation_rows[target_column].astype(int).to_numpy()
    )
    bootstrap = bootstrap_calibration_metrics(
        validation_target,
        probabilities,
        calibration_policy,
        model_policy,
    )
    metrics = add_bootstrap_intervals(
        metrics,
        bootstrap,
        calibration_policy,
    )
    differences = pairwise_brier_differences(
        metrics,
        bootstrap,
        calibration_policy,
    )
    selection = choose_calibration_method(
        metrics,
        differences,
        calibration_policy,
    )
    metrics = metrics.merge(
        selection[
            [
                "method",
                "selected_method",
                "probability_display_ready",
                "recommended_dashboard_label",
            ]
        ],
        on="method",
        how="left",
        validate="one_to_one",
    )
    validation = validate_outputs(
        train,
        validation_rows,
        probabilities,
        oof_probability,
        folds,
        metrics,
        reliability,
        bootstrap,
        differences,
        selection,
        predictions,
        checkpoint_41_predictions,
        calibration_policy,
        feature_policy,
        test_rows,
        selected_model,
    )
    save_outputs(
        metrics,
        reliability,
        bootstrap,
        differences,
        selection,
        predictions,
        folds,
        validation,
    )
    print_results(
        metrics,
        differences,
        selection,
        folds,
        validation,
    )


if __name__ == "__main__":
    main()
