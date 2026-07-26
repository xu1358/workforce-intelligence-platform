"""Compare Version 2 retention models without using the test target.

Primary model comparison fits on the 2023 snapshot and evaluates on the
2024 validation snapshot. Five employee-grouped development folds provide
a separate robustness check. Paired stratified bootstrap samples quantify
validation uncertainty.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from retention_feature_policy import (
    build_preprocessor,
    load_feature_policy,
    normalize_feature_frame,
    selected_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "retention_multi_snapshot.csv"
ASSIGNMENT_PATH = PROCESSED_DIR / "model_split_assignments.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "model_comparison_v2.yaml"

COMPARISON_PATH = PROCESSED_DIR / "model_comparison_v2.csv"
GROUP_METRICS_PATH = PROCESSED_DIR / "model_grouped_cv_metrics.csv"
BOOTSTRAP_PATH = PROCESSED_DIR / "model_metric_bootstrap.csv"
PAIRWISE_PATH = (
    PROCESSED_DIR / "model_pairwise_pr_auc_bootstrap.csv"
)
SELECTION_PATH = (
    PROCESSED_DIR / "model_selection_decision_v2.csv"
)
PREDICTION_PATH = (
    PROCESSED_DIR / "retention_validation_predictions_v2.csv"
)
VALIDATION_PATH = (
    PROCESSED_DIR / "model_comparison_validation.csv"
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


def load_development_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    int,
]:
    """Load train and validation rows while excluding test targets."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing retention_multi_snapshot.csv. "
            "Run Checkpoint 38 first."
        )
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing model_split_assignments.csv. "
            "Run Checkpoint 40 first."
        )

    dataset = pd.read_csv(DATA_PATH)
    assignments = pd.read_csv(ASSIGNMENT_PATH)
    comparison_policy = load_yaml(CONFIG_PATH)
    feature_policy = load_feature_policy()

    target_column = str(feature_policy["target_column"])
    if target_column in assignments.columns:
        raise ValueError(
            "Split assignments must not contain the attrition target."
        )

    key_columns = ["employee_id", "snapshot_date"]
    assignments["snapshot_date"] = assignments[
        "snapshot_date"
    ].astype(str)
    test_rows = int(assignments["primary_split"].eq("test").sum())

    development_assignments = assignments.loc[
        assignments["primary_split"].isin(["train", "validation"])
    ].copy()
    development = dataset.merge(
        development_assignments,
        on=key_columns,
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

    return (
        train,
        validation,
        comparison_policy,
        feature_policy,
        test_rows,
    )


def build_candidate_model(
    model_name: str,
    comparison_policy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> Pipeline:
    """Build one fixed candidate pipeline."""

    configuration = comparison_policy["candidate_models"][model_name]
    model_type = str(configuration["model_type"])
    parameters = dict(configuration["parameters"])

    if model_type == "logistic_regression":
        classifier = LogisticRegression(**parameters)
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(**parameters)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(**parameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(feature_policy),
            ),
            ("classifier", classifier),
        ]
    )


def fit_candidate(
    model: Pipeline,
    model_name: str,
    features: pd.DataFrame,
    target: np.ndarray,
    comparison_policy: dict[str, Any],
) -> Pipeline:
    """Fit one model, adding balanced weights when configured."""

    configuration = comparison_policy["candidate_models"][model_name]
    fit_parameters: dict[str, Any] = {}

    if bool(configuration.get("use_balanced_sample_weight", False)):
        fit_parameters["classifier__sample_weight"] = (
            compute_sample_weight(
                class_weight="balanced",
                y=target,
            )
        )

    model.fit(features, target, **fit_parameters)
    return model


def top_fraction_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    top_fraction: float,
) -> dict[str, float | int]:
    """Measure precision, capture, and lift in the highest scores."""

    top_count = max(1, int(np.ceil(len(target) * top_fraction)))
    ranked_positions = np.argsort(
        -probability,
        kind="mergesort",
    )[:top_count]
    selected_target = target[ranked_positions]
    baseline_rate = float(target.mean())
    top_precision = float(selected_target.mean())
    positive_cases = int(target.sum())
    capture = (
        float(selected_target.sum() / positive_cases)
        if positive_cases
        else 0.0
    )
    lift = (
        float(top_precision / baseline_rate)
        if baseline_rate > 0
        else 0.0
    )

    return {
        "top_count": top_count,
        "top_decile_precision": top_precision,
        "top_decile_capture": capture,
        "top_decile_lift": lift,
    }


def calculate_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    comparison_policy: dict[str, Any],
) -> dict[str, float | int]:
    """Calculate ranking, probability, and fixed-threshold metrics."""

    evaluation = comparison_policy["evaluation"]
    threshold = float(evaluation["reporting_threshold"])
    predicted = (probability >= threshold).astype(int)
    ranking = top_fraction_metrics(
        target,
        probability,
        float(evaluation["top_fraction"]),
    )

    return {
        "rows": len(target),
        "positive_cases": int(target.sum()),
        "positive_rate": float(target.mean()),
        "pr_auc": float(
            average_precision_score(target, probability)
        ),
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier_score": float(
            brier_score_loss(target, probability)
        ),
        "precision_at_0_50": float(
            precision_score(
                target,
                predicted,
            )
        ),
        "recall_at_0_50": float(
            recall_score(
                target,
                predicted,
            )
        ),
        "f1_at_0_50": float(
            f1_score(
                target,
                predicted,
            )
        ),
        **ranking,
    }


def compare_on_temporal_validation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    comparison_policy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """Fit on 2023 and compare candidate probabilities on 2024."""

    target_column = str(feature_policy["target_column"])
    train_features = normalize_feature_frame(
        train,
        feature_policy,
    )
    validation_features = normalize_feature_frame(
        validation,
        feature_policy,
    )
    train_target = train[target_column].astype(int).to_numpy()
    validation_target = (
        validation[target_column].astype(int).to_numpy()
    )

    metric_records: list[dict[str, Any]] = []
    prediction_records: list[pd.DataFrame] = []
    probabilities: dict[str, np.ndarray] = {}

    for model_name in comparison_policy["candidate_models"]:
        print(f"Fitting primary comparison: {model_name}")
        model = build_candidate_model(
            model_name,
            comparison_policy,
            feature_policy,
        )
        model = fit_candidate(
            model,
            model_name,
            train_features,
            train_target,
            comparison_policy,
        )
        probability = np.asarray(
            model.predict_proba(validation_features)[:, 1],
            dtype=float,
        )
        probabilities[model_name] = probability
        metric_records.append(
            {
                "model": model_name,
                "evaluation_period": "2024 validation",
                **calculate_metrics(
                    validation_target,
                    probability,
                    comparison_policy,
                ),
            }
        )
        prediction_records.append(
            pd.DataFrame(
                {
                    "employee_id": validation[
                        "employee_id"
                    ].to_numpy(),
                    "snapshot_date": validation[
                        "snapshot_date"
                    ].astype(str).to_numpy(),
                    "actual_attrition": validation_target,
                    "model": model_name,
                    "attrition_probability": probability,
                }
            )
        )

    return (
        pd.DataFrame(metric_records),
        probabilities,
        pd.concat(prediction_records, ignore_index=True),
    )


def grouped_robustness_metrics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    comparison_policy: dict[str, Any],
    feature_policy: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate candidates in employee-disjoint development folds."""

    development = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    target_column = str(feature_policy["target_column"])
    folds = sorted(
        development["group_cv_fold"].dropna().astype(int).unique()
    )
    records: list[dict[str, Any]] = []

    for fold in folds:
        fold_train = development.loc[
            development["group_cv_fold"].astype(int).ne(fold)
        ]
        fold_validation = development.loc[
            development["group_cv_fold"].astype(int).eq(fold)
        ]
        training_employees = set(fold_train["employee_id"])
        validation_employees = set(
            fold_validation["employee_id"]
        )
        overlap = len(training_employees & validation_employees)

        training_features = normalize_feature_frame(
            fold_train,
            feature_policy,
        )
        validation_features = normalize_feature_frame(
            fold_validation,
            feature_policy,
        )
        training_target = (
            fold_train[target_column].astype(int).to_numpy()
        )
        validation_target = (
            fold_validation[target_column].astype(int).to_numpy()
        )

        for model_name in comparison_policy["candidate_models"]:
            print(
                "Fitting grouped robustness fold "
                f"{fold}: {model_name}"
            )
            model = build_candidate_model(
                model_name,
                comparison_policy,
                feature_policy,
            )
            model = fit_candidate(
                model,
                model_name,
                training_features,
                training_target,
                comparison_policy,
            )
            probability = np.asarray(
                model.predict_proba(validation_features)[:, 1],
                dtype=float,
            )
            records.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "training_rows": len(fold_train),
                    "validation_rows": len(fold_validation),
                    "training_employees": len(training_employees),
                    "validation_employees": len(
                        validation_employees
                    ),
                    "employee_overlap": overlap,
                    **calculate_metrics(
                        validation_target,
                        probability,
                        comparison_policy,
                    ),
                }
            )

    return pd.DataFrame(records)


def paired_bootstrap_metrics(
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    comparison_policy: dict[str, Any],
) -> pd.DataFrame:
    """Create paired stratified validation bootstrap metrics."""

    bootstrap_policy = comparison_policy["bootstrap"]
    iterations = int(bootstrap_policy["iterations"])
    random_generator = np.random.default_rng(
        int(bootstrap_policy["random_seed"])
    )
    positive_positions = np.flatnonzero(target == 1)
    negative_positions = np.flatnonzero(target == 0)
    records: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        selected_positions = np.concatenate(
            [
                random_generator.choice(
                    positive_positions,
                    size=len(positive_positions),
                    replace=True,
                ),
                random_generator.choice(
                    negative_positions,
                    size=len(negative_positions),
                    replace=True,
                ),
            ]
        )
        sampled_target = target[selected_positions]

        for model_name, model_probability in probabilities.items():
            sampled_probability = model_probability[
                selected_positions
            ]
            metrics = calculate_metrics(
                sampled_target,
                sampled_probability,
                comparison_policy,
            )
            records.append(
                {
                    "iteration": iteration,
                    "model": model_name,
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "brier_score": metrics["brier_score"],
                    "top_decile_precision": metrics[
                        "top_decile_precision"
                    ],
                    "top_decile_capture": metrics[
                        "top_decile_capture"
                    ],
                    "top_decile_lift": metrics[
                        "top_decile_lift"
                    ],
                }
            )

        if iteration % 100 == 0:
            print(
                "Completed paired validation bootstrap: "
                f"{iteration}/{iterations}"
            )

    return pd.DataFrame(records)


def bootstrap_summary(
    bootstrap: pd.DataFrame,
    comparison_policy: dict[str, Any],
) -> pd.DataFrame:
    """Summarize bootstrap means and confidence intervals."""

    confidence_level = float(
        comparison_policy["bootstrap"]["confidence_level"]
    )
    tail = (1.0 - confidence_level) / 2.0
    metrics = [
        "pr_auc",
        "roc_auc",
        "brier_score",
        "top_decile_precision",
        "top_decile_capture",
        "top_decile_lift",
    ]
    records: list[dict[str, Any]] = []

    for model_name, model_rows in bootstrap.groupby("model"):
        record: dict[str, Any] = {"model": model_name}
        for metric in metrics:
            values = model_rows[metric].astype(float)
            record[f"{metric}_bootstrap_mean"] = float(
                values.mean()
            )
            record[f"{metric}_lower_95"] = float(
                values.quantile(tail)
            )
            record[f"{metric}_upper_95"] = float(
                values.quantile(1.0 - tail)
            )
        records.append(record)

    return pd.DataFrame(records)


def pairwise_pr_auc_comparison(
    temporal_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    comparison_policy: dict[str, Any],
) -> pd.DataFrame:
    """Estimate paired PR-AUC differences between every model pair."""

    confidence_level = float(
        comparison_policy["bootstrap"]["confidence_level"]
    )
    tail = (1.0 - confidence_level) / 2.0
    observed = temporal_metrics.set_index("model")["pr_auc"].to_dict()
    pivot = bootstrap.pivot(
        index="iteration",
        columns="model",
        values="pr_auc",
    )
    model_names = list(comparison_policy["candidate_models"])
    records: list[dict[str, Any]] = []

    for model_a, model_b in combinations(model_names, 2):
        differences = pivot[model_a] - pivot[model_b]
        lower = float(differences.quantile(tail))
        upper = float(differences.quantile(1.0 - tail))
        if lower > 0:
            conclusion = f"{model_a} higher"
        elif upper < 0:
            conclusion = f"{model_b} higher"
        else:
            conclusion = "Inconclusive"

        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "observed_pr_auc_difference_a_minus_b": (
                    float(observed[model_a] - observed[model_b])
                ),
                "bootstrap_mean_difference": float(
                    differences.mean()
                ),
                "difference_lower_95": lower,
                "difference_upper_95": upper,
                "interval_includes_zero": lower <= 0 <= upper,
                "conclusion": conclusion,
            }
        )

    return pd.DataFrame(records)


def difference_interval(
    best_model: str,
    candidate_model: str,
    pairwise: pd.DataFrame,
) -> tuple[float, float]:
    """Return the interval for best-model minus candidate PR-AUC."""

    if best_model == candidate_model:
        return 0.0, 0.0

    direct = pairwise.loc[
        pairwise["model_a"].eq(best_model)
        & pairwise["model_b"].eq(candidate_model)
    ]
    if not direct.empty:
        row = direct.iloc[0]
        return (
            float(row["difference_lower_95"]),
            float(row["difference_upper_95"]),
        )

    reverse = pairwise.loc[
        pairwise["model_a"].eq(candidate_model)
        & pairwise["model_b"].eq(best_model)
    ]
    if reverse.empty:
        raise ValueError("Missing paired model comparison.")
    row = reverse.iloc[0]
    return (
        -float(row["difference_upper_95"]),
        -float(row["difference_lower_95"]),
    )


def choose_model(
    temporal_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    comparison_policy: dict[str, Any],
) -> pd.DataFrame:
    """Choose a provisional model from validation evidence only."""

    selection_policy = comparison_policy["selection"]
    tolerance = float(
        selection_policy["practical_pr_auc_tolerance"]
    )
    metrics = temporal_metrics.set_index("model")
    best_model = str(metrics["pr_auc"].idxmax())
    best_pr_auc = float(metrics.loc[best_model, "pr_auc"])
    complexity_order = {
        model_name: rank
        for rank, model_name in enumerate(
            selection_policy["complexity_preference"],
            start=1,
        )
    }
    records: list[dict[str, Any]] = []

    for model_name in comparison_policy["candidate_models"]:
        model_pr_auc = float(metrics.loc[model_name, "pr_auc"])
        gap = best_pr_auc - model_pr_auc
        lower, upper = difference_interval(
            best_model,
            model_name,
            pairwise,
        )
        interval_includes_zero = lower <= 0 <= upper
        within_tolerance = gap <= tolerance
        eligible = within_tolerance or interval_includes_zero
        records.append(
            {
                "model": model_name,
                "validation_pr_auc": model_pr_auc,
                "best_observed_model": best_model,
                "gap_from_best_pr_auc": gap,
                "best_minus_model_lower_95": lower,
                "best_minus_model_upper_95": upper,
                "paired_interval_includes_zero": (
                    interval_includes_zero
                ),
                "within_practical_tolerance": within_tolerance,
                "eligible_for_simplicity_selection": eligible,
                "complexity_rank": complexity_order[model_name],
            }
        )

    decision = pd.DataFrame(records)
    eligible_rows = decision.loc[
        decision["eligible_for_simplicity_selection"]
    ].sort_values("complexity_rank")
    selected_model = str(eligible_rows.iloc[0]["model"])
    decision["selected_model"] = decision["model"].eq(
        selected_model
    )
    decision["selection_basis"] = np.where(
        decision["selected_model"],
        (
            "Selected using validation PR-AUC, paired uncertainty, "
            "practical tolerance, and model simplicity."
        ),
        "Not selected.",
    )

    return decision


def enrich_comparison(
    temporal_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selection: pd.DataFrame,
    comparison_policy: dict[str, Any],
) -> pd.DataFrame:
    """Combine primary, grouped, uncertainty, and selection results."""

    grouped = (
        group_metrics.groupby("model")
        .agg(
            grouped_cv_pr_auc_mean=("pr_auc", "mean"),
            grouped_cv_pr_auc_std=("pr_auc", "std"),
            grouped_cv_roc_auc_mean=("roc_auc", "mean"),
            grouped_cv_brier_mean=("brier_score", "mean"),
            grouped_cv_top_decile_lift_mean=(
                "top_decile_lift",
                "mean",
            ),
        )
        .reset_index()
    )
    intervals = bootstrap_summary(
        bootstrap,
        comparison_policy,
    )
    selection_columns = selection[
        [
            "model",
            "gap_from_best_pr_auc",
            "paired_interval_includes_zero",
            "within_practical_tolerance",
            "eligible_for_simplicity_selection",
            "selected_model",
            "selection_basis",
        ]
    ]
    comparison = (
        temporal_metrics.merge(grouped, on="model", how="left")
        .merge(intervals, on="model", how="left")
        .merge(selection_columns, on="model", how="left")
    )
    return comparison.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)


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


def validate_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    group_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pairwise: pd.DataFrame,
    selection: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison_policy: dict[str, Any],
    feature_policy: dict[str, Any],
    test_rows: int,
) -> pd.DataFrame:
    """Validate holdout protection and comparison completeness."""

    checks: list[dict[str, Any]] = []
    validation_policy = comparison_policy["validation"]
    expected_models = set(comparison_policy["candidate_models"])
    target_column = str(feature_policy["target_column"])

    add_check(
        checks,
        "Reserved test target remains unused",
        test_rows > 0 and "test" not in set(predictions.columns),
        f"{test_rows} reserved rows; 0 test targets used",
        "Reserved rows > 0 and target usage = 0",
        "Only train and validation assignments were joined to targets.",
    )

    train_keys = set(
        zip(
            train["employee_id"],
            train["snapshot_date"].astype(str),
            strict=True,
        )
    )
    validation_keys = set(
        zip(
            validation["employee_id"],
            validation["snapshot_date"].astype(str),
            strict=True,
        )
    )
    add_check(
        checks,
        "Primary train and validation rows are separate",
        not (train_keys & validation_keys),
        len(train_keys & validation_keys),
        "0 overlapping employee-snapshot keys",
        "The same employee may recur later, but one row has one role.",
    )

    add_check(
        checks,
        "Validation population is sufficiently large",
        len(validation)
        >= int(validation_policy["minimum_validation_rows"]),
        len(validation),
        f">= {validation_policy['minimum_validation_rows']}",
        "Primary model comparison requires a useful future population.",
    )

    validation_positives = int(
        validation[target_column].astype(int).sum()
    )
    add_check(
        checks,
        "Validation has sufficient positive cases",
        validation_positives
        >= int(
            validation_policy[
                "minimum_validation_positive_cases"
            ]
        ),
        validation_positives,
        (
            ">= "
            + str(
                validation_policy[
                    "minimum_validation_positive_cases"
                ]
            )
        ),
        "PR-AUC and ranking metrics require attrition examples.",
    )

    observed_models = set(temporal_metrics["model"])
    add_check(
        checks,
        "Every configured candidate was evaluated",
        observed_models == expected_models,
        sorted(observed_models),
        str(sorted(expected_models)),
        "No model may be silently skipped.",
    )

    probability_bounds = [
        bool(
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values <= 1).all()
        )
        for values in probabilities.values()
    ]
    add_check(
        checks,
        "Validation probabilities are finite and bounded",
        all(probability_bounds),
        probability_bounds,
        "All True",
        "Every candidate must output valid probabilities.",
    )

    metric_columns = [
        "pr_auc",
        "roc_auc",
        "brier_score",
        "precision_at_0_50",
        "recall_at_0_50",
        "f1_at_0_50",
        "top_decile_precision",
        "top_decile_capture",
    ]
    metrics_valid = all(
        temporal_metrics[column].between(0, 1).all()
        for column in metric_columns
    )
    add_check(
        checks,
        "Primary probability and classification metrics are valid",
        metrics_valid,
        metrics_valid,
        "True",
        "Metrics expressed as rates must remain between zero and one.",
    )

    expected_fold_rows = (
        int(validation_policy["expected_group_folds"])
        * len(expected_models)
    )
    add_check(
        checks,
        "All grouped model-fold evaluations completed",
        len(group_metrics) == expected_fold_rows,
        len(group_metrics),
        str(expected_fold_rows),
        "Every candidate must run in every employee-grouped fold.",
    )

    maximum_group_overlap = int(
        group_metrics["employee_overlap"].max()
    )
    add_check(
        checks,
        "Grouped robustness folds are employee-disjoint",
        maximum_group_overlap == 0,
        maximum_group_overlap,
        "0 overlapping employees",
        "No fold may train and validate on one employee.",
    )

    expected_bootstrap_rows = (
        int(comparison_policy["bootstrap"]["iterations"])
        * len(expected_models)
    )
    add_check(
        checks,
        "Paired validation bootstraps completed",
        len(bootstrap) == expected_bootstrap_rows,
        len(bootstrap),
        str(expected_bootstrap_rows),
        "Every iteration uses one shared resample across candidates.",
    )

    expected_pairwise_rows = (
        len(expected_models) * (len(expected_models) - 1) // 2
    )
    add_check(
        checks,
        "Every model pair has an uncertainty interval",
        len(pairwise) == expected_pairwise_rows,
        len(pairwise),
        str(expected_pairwise_rows),
        "Pairwise intervals prevent unsupported winner claims.",
    )

    selected_count = int(selection["selected_model"].sum())
    add_check(
        checks,
        "Exactly one provisional model is selected",
        selected_count == 1,
        selected_count,
        "1",
        "Selection uses validation evidence and explicit simplicity rules.",
    )

    selected_raw_features = selected_features(feature_policy)
    add_check(
        checks,
        "Checkpoint 39 feature policy is reused",
        len(selected_raw_features) == 21
        and target_column not in selected_raw_features,
        len(selected_raw_features),
        "21 target-free raw features",
        "No candidate receives dropped or outcome columns.",
    )

    failures = [
        check for check in checks if check["status"] == "FAIL"
    ]
    validation_table = pd.DataFrame(checks)
    if failures:
        raise ValueError(
            "Model comparison validation failed:\n"
            + pd.DataFrame(failures).to_string(index=False)
        )

    return validation_table


def save_outputs(
    comparison: pd.DataFrame,
    group_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pairwise: pd.DataFrame,
    selection: pd.DataFrame,
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Save all generated Checkpoint 41 artifacts."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    group_metrics.to_csv(GROUP_METRICS_PATH, index=False)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    pairwise.to_csv(PAIRWISE_PATH, index=False)
    selection.to_csv(SELECTION_PATH, index=False)
    predictions.to_csv(PREDICTION_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)


def print_results(
    comparison: pd.DataFrame,
    pairwise: pd.DataFrame,
    selection: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Print the main Checkpoint 41 evidence."""

    display_columns = [
        "model",
        "pr_auc",
        "pr_auc_lower_95",
        "pr_auc_upper_95",
        "roc_auc",
        "brier_score",
        "precision_at_0_50",
        "recall_at_0_50",
        "f1_at_0_50",
        "top_decile_lift",
        "top_decile_capture",
        "grouped_cv_pr_auc_mean",
        "selected_model",
    ]
    print("\nTEMPORAL VALIDATION MODEL COMPARISON")
    print(comparison[display_columns].to_string(index=False))

    print("\nPAIRED PR-AUC DIFFERENCES")
    print(pairwise.to_string(index=False))

    print("\nPROVISIONAL MODEL SELECTION")
    print(
        selection[
            [
                "model",
                "validation_pr_auc",
                "gap_from_best_pr_auc",
                "best_minus_model_lower_95",
                "best_minus_model_upper_95",
                "eligible_for_simplicity_selection",
                "complexity_rank",
                "selected_model",
            ]
        ].to_string(index=False)
    )

    selected_model = str(
        selection.loc[
            selection["selected_model"],
            "model",
        ].iloc[0]
    )
    print(f"\nSelected provisional model: {selected_model}")

    print("\nMODEL COMPARISON VALIDATION")
    print(validation.to_string(index=False))
    print(f"\nSaved comparison: {COMPARISON_PATH}")
    print(
        "\nVERSION 2 MODEL COMPARISON COMPLETED SUCCESSFULLY"
    )


def main() -> None:
    """Run the complete Checkpoint 41 model comparison."""

    (
        train,
        validation_rows,
        comparison_policy,
        feature_policy,
        test_rows,
    ) = load_development_data()
    (
        temporal_metrics,
        probabilities,
        predictions,
    ) = compare_on_temporal_validation(
        train,
        validation_rows,
        comparison_policy,
        feature_policy,
    )
    group_metrics = grouped_robustness_metrics(
        train,
        validation_rows,
        comparison_policy,
        feature_policy,
    )
    target_column = str(feature_policy["target_column"])
    validation_target = (
        validation_rows[target_column].astype(int).to_numpy()
    )
    bootstrap = paired_bootstrap_metrics(
        validation_target,
        probabilities,
        comparison_policy,
    )
    pairwise = pairwise_pr_auc_comparison(
        temporal_metrics,
        bootstrap,
        comparison_policy,
    )
    selection = choose_model(
        temporal_metrics,
        pairwise,
        comparison_policy,
    )
    comparison = enrich_comparison(
        temporal_metrics,
        group_metrics,
        bootstrap,
        selection,
        comparison_policy,
    )
    validation = validate_outputs(
        train,
        validation_rows,
        temporal_metrics,
        probabilities,
        group_metrics,
        bootstrap,
        pairwise,
        selection,
        predictions,
        comparison_policy,
        feature_policy,
        test_rows,
    )
    save_outputs(
        comparison,
        group_metrics,
        bootstrap,
        pairwise,
        selection,
        predictions,
        validation,
    )
    print_results(
        comparison,
        pairwise,
        selection,
        validation,
    )


if __name__ == "__main__":
    main()
