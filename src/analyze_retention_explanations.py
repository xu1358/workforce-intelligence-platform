"""Explain the selected retention model and test explanation stability.

Checkpoint 54 calculates exact interventional linear SHAP values for the
selected Logistic Regression on its native log-odds scale. Employee-level
values are used only in memory. Saved artifacts are aggregate summaries for
the current synthetic workforce and five employee-grouped stability refits.

These explanations describe model associations. They do not establish causes,
change the frozen policy, or authorize an employment decision.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

MATPLOTLIB_CACHE_DIR = Path(tempfile.gettempdir()) / "workforce_intelligence_matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedGroupKFold

from compare_retention_models_v2 import (
    build_candidate_model,
    fit_candidate,
)
from optimize_retention_policy import (
    CURRENT_PATH,
    CURRENT_SCORE_PATH,
    DECISION_PATH,
    FINAL_MODEL_METRICS_PATH,
    fit_grouped_sigmoid_model,
    load_inputs_before_test_access,
    load_labeled_history_after_freeze,
)
from retention_feature_policy import (
    normalize_feature_frame,
    selected_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "retention_explanations"
FIGURE_DIR = OUTPUT_DIR / "figures"

CONFIG_PATH = PROJECT_ROOT / "config" / "retention_explanations.yaml"

GLOBAL_IMPORTANCE_PATH = OUTPUT_DIR / "retention_explanation_global_importance.csv"
FOLD_STABILITY_PATH = OUTPUT_DIR / "retention_explanation_fold_stability.csv"
PAIRWISE_STABILITY_PATH = OUTPUT_DIR / "retention_explanation_pairwise_stability.csv"
STABILITY_SUMMARY_PATH = OUTPUT_DIR / "retention_explanation_stability_summary.csv"
QUARTILE_PATH = OUTPUT_DIR / "retention_explanation_probability_quartiles.csv"
DRIVER_FREQUENCY_PATH = OUTPUT_DIR / "retention_explanation_driver_frequency.csv"
VALIDATION_PATH = OUTPUT_DIR / "retention_explanation_validation.csv"


def load_explanation_policy(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Load the committed explanation policy."""

    if not path.exists():
        raise FileNotFoundError(f"Missing explanation policy: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")
    return loaded


def encoded_feature_to_raw(
    encoded_feature: str,
    raw_features: list[str],
    categorical_features: list[str],
) -> str:
    """Map an encoded feature name back to its policy raw feature."""

    if encoded_feature in raw_features:
        return encoded_feature

    matches = [
        feature
        for feature in categorical_features
        if encoded_feature.startswith(f"{feature}_")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not map encoded feature to one raw feature: {encoded_feature}"
        )
    return matches[0]


def linear_shap_values(
    model: Any,
    background_features: pd.DataFrame,
    scoring_features: pd.DataFrame,
) -> tuple[list[str], np.ndarray, float, np.ndarray]:
    """Calculate exact interventional linear SHAP values on log-odds.

    For a linear model ``f(x) = intercept + beta @ x`` and a transformed
    background mean ``E[X]``, each value is
    ``beta_j * (x_j - E[X_j])``.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    background = np.asarray(
        preprocessor.transform(background_features),
        dtype=float,
    )
    scoring = np.asarray(
        preprocessor.transform(scoring_features),
        dtype=float,
    )
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    background_mean = background.mean(axis=0)

    values = (scoring - background_mean) * coefficients
    expected_log_odds = float(
        classifier.intercept_[0] + np.dot(background_mean, coefficients)
    )
    decision = np.asarray(
        model.decision_function(scoring_features),
        dtype=float,
    )
    encoded_names = list(preprocessor.get_feature_names_out())

    if values.shape[1] != len(encoded_names):
        raise ValueError("Encoded feature names do not match the explanation matrix.")

    return encoded_names, values, expected_log_odds, decision


def group_encoded_contributions(
    encoded_names: list[str],
    encoded_values: np.ndarray,
    raw_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    """Sum encoded contributions into the committed raw-feature groups."""

    grouped = np.zeros(
        (encoded_values.shape[0], len(raw_features)),
        dtype=float,
    )
    raw_positions = {feature: position for position, feature in enumerate(raw_features)}

    for encoded_position, encoded_name in enumerate(encoded_names):
        raw_name = encoded_feature_to_raw(
            encoded_name,
            raw_features,
            categorical_features,
        )
        grouped[:, raw_positions[raw_name]] += encoded_values[
            :,
            encoded_position,
        ]

    return grouped


def stable_importance_order(
    values: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Order descending importance with feature name as a tie-break."""

    names = np.asarray(feature_names, dtype=str)
    return np.lexsort((names, -np.asarray(values, dtype=float)))


def importance_ranks(
    values: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Return one-based deterministic ranks for importance values."""

    order = stable_importance_order(values, feature_names)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def spearman_rank_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Calculate Spearman correlation from average ranks."""

    first_rank = pd.Series(first).rank(method="average").to_numpy()
    second_rank = pd.Series(second).rank(method="average").to_numpy()
    correlation = np.corrcoef(first_rank, second_rank)[0, 1]
    return float(correlation)


def top_k_jaccard(
    first: np.ndarray,
    second: np.ndarray,
    feature_names: list[str],
    top_k: int,
) -> float:
    """Calculate top-k feature-set overlap using Jaccard similarity."""

    first_set = set(stable_importance_order(first, feature_names)[:top_k])
    second_set = set(stable_importance_order(second, feature_names)[:top_k])
    union = first_set | second_set
    return float(len(first_set & second_set) / len(union))


def direction_label(value: float, tolerance: float = 1e-12) -> str:
    """Describe the signed model contribution without causal language."""

    if value > tolerance:
        return "Higher modeled log-odds"
    if value < -tolerance:
        return "Lower modeled log-odds"
    return "Near-zero mean contribution"


def probability_quartiles(
    probability: np.ndarray,
    labels: list[str],
) -> pd.Categorical:
    """Assign deterministic equal-count probability quartiles."""

    if len(labels) != 4:
        raise ValueError("Exactly four probability-quartile labels are required.")

    ranked = pd.Series(probability).rank(method="first")
    return pd.qcut(
        ranked,
        q=4,
        labels=labels,
    )


def build_global_importance(
    raw_features: list[str],
    grouped_values: np.ndarray,
) -> pd.DataFrame:
    """Summarize current aggregate raw-feature importance."""

    mean_absolute = np.abs(grouped_values).mean(axis=0)
    mean_signed = grouped_values.mean(axis=0)
    total_importance = float(mean_absolute.sum())
    ranks = importance_ranks(mean_absolute, raw_features)

    table = pd.DataFrame(
        {
            "raw_feature": raw_features,
            "mean_absolute_shap_log_odds": mean_absolute,
            "mean_signed_shap_log_odds": mean_signed,
            "importance_share": mean_absolute / total_importance,
            "importance_rank": ranks,
            "mean_direction": [direction_label(value) for value in mean_signed],
        }
    )
    return table.sort_values("importance_rank").reset_index(drop=True)


def build_quartile_summary(
    raw_features: list[str],
    grouped_values: np.ndarray,
    probability: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """Summarize signed and absolute contributions by score quartile."""

    quartile = probability_quartiles(probability, labels)
    records: list[dict[str, Any]] = []
    for label in labels:
        selected = np.asarray(quartile == label, dtype=bool)
        for position, feature in enumerate(raw_features):
            signed = float(grouped_values[selected, position].mean())
            records.append(
                {
                    "probability_quartile": label,
                    "raw_feature": feature,
                    "employees": int(selected.sum()),
                    "mean_probability": float(probability[selected].mean()),
                    "mean_signed_shap_log_odds": signed,
                    "mean_absolute_shap_log_odds": float(
                        np.abs(grouped_values[selected, position]).mean()
                    ),
                    "mean_direction": direction_label(signed),
                }
            )
    return pd.DataFrame(records)


def build_driver_frequency(
    raw_features: list[str],
    grouped_values: np.ndarray,
) -> pd.DataFrame:
    """Count each feature's strongest positive and negative appearances."""

    rows = len(grouped_values)
    positive_position = np.argmax(grouped_values, axis=1)
    negative_position = np.argmin(grouped_values, axis=1)
    records: list[dict[str, Any]] = []

    for driver_type, positions in [
        ("Strongest positive model driver", positive_position),
        ("Strongest negative model driver", negative_position),
    ]:
        counts = pd.Series(positions).value_counts()
        for position, feature in enumerate(raw_features):
            count = int(counts.get(position, 0))
            records.append(
                {
                    "driver_type": driver_type,
                    "raw_feature": feature,
                    "employee_appearances": count,
                    "appearance_share": float(count / rows),
                }
            )

    table = pd.DataFrame(records)
    return table.sort_values(
        ["driver_type", "employee_appearances", "raw_feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def fit_explanation_model(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    inputs: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Fit the committed model and return normalized feature frames."""

    feature_policy = inputs["feature_policy"]
    model_policy = inputs["model_policy"]
    model_name = inputs["selected_model"]
    target_column = str(feature_policy["target_column"])

    training_features = normalize_feature_frame(
        training,
        feature_policy,
    )
    scoring_features = normalize_feature_frame(
        scoring,
        feature_policy,
    )
    target = training[target_column].to_numpy(dtype=int)

    model = build_candidate_model(
        model_name,
        model_policy,
        feature_policy,
    )
    model = fit_candidate(
        model,
        model_name,
        training_features,
        target,
        model_policy,
    )
    return model, training_features, scoring_features


def calculate_fold_stability(
    history: pd.DataFrame,
    current: pd.DataFrame,
    current_probability: np.ndarray,
    inputs: dict[str, Any],
    explanation_policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit employee-grouped models and compare aggregate explanations."""

    feature_policy = inputs["feature_policy"]
    raw_features = selected_features(feature_policy)
    categorical_features = list(feature_policy["categorical_features"])
    target_column = str(feature_policy["target_column"])
    folds = int(explanation_policy["stability"]["employee_grouped_folds"])
    random_seed = int(explanation_policy["stability"]["random_seed"])
    top_k = int(explanation_policy["stability"]["ranking_top_k"])
    labels = list(explanation_policy["explanation"]["probability_quartiles"])

    all_features = normalize_feature_frame(
        history,
        feature_policy,
    )
    current_features = normalize_feature_frame(
        current,
        feature_policy,
    )
    target = history[target_column].to_numpy(dtype=int)
    groups = history["employee_id"].to_numpy(dtype=int)
    highest_quartile = np.asarray(
        probability_quartiles(current_probability, labels) == labels[-1],
        dtype=bool,
    )

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_seed,
    )
    fold_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    fold_importance: dict[int, np.ndarray] = {}

    for fold, (fit_index, holdout_index) in enumerate(
        splitter.split(all_features, target, groups=groups),
        start=1,
    ):
        print(
            f"Fitting employee-grouped explanation stability model: fold {fold}/{folds}"
        )
        model = build_candidate_model(
            inputs["selected_model"],
            inputs["model_policy"],
            feature_policy,
        )
        model = fit_candidate(
            model,
            inputs["selected_model"],
            all_features.iloc[fit_index],
            target[fit_index],
            inputs["model_policy"],
        )
        encoded_names, encoded_values, expected_value, decision = linear_shap_values(
            model,
            all_features.iloc[fit_index],
            current_features,
        )
        grouped = group_encoded_contributions(
            encoded_names,
            encoded_values,
            raw_features,
            categorical_features,
        )
        importance = np.abs(grouped).mean(axis=0)
        ranks = importance_ranks(importance, raw_features)
        signed_high = grouped[highest_quartile].mean(axis=0)
        fold_importance[fold] = importance

        fit_employees = set(groups[fit_index])
        holdout_employees = set(groups[holdout_index])
        additivity_error = float(
            np.max(np.abs(expected_value + encoded_values.sum(axis=1) - decision))
        )
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": len(fit_index),
                "holdout_rows": len(holdout_index),
                "fit_employees": len(fit_employees),
                "holdout_employees": len(holdout_employees),
                "employee_overlap": len(fit_employees & holdout_employees),
                "holdout_positive_cases": int(target[holdout_index].sum()),
                "maximum_additivity_error": additivity_error,
            }
        )
        for position, feature in enumerate(raw_features):
            feature_records.append(
                {
                    "fold": fold,
                    "raw_feature": feature,
                    "mean_absolute_shap_log_odds": float(importance[position]),
                    "importance_rank": int(ranks[position]),
                    "mean_signed_shap_highest_quartile": float(signed_high[position]),
                    "highest_quartile_direction": direction_label(
                        float(signed_high[position])
                    ),
                }
            )

    pair_records: list[dict[str, Any]] = []
    for first_fold in range(1, folds + 1):
        for second_fold in range(first_fold + 1, folds + 1):
            first = fold_importance[first_fold]
            second = fold_importance[second_fold]
            pair_records.append(
                {
                    "first_fold": first_fold,
                    "second_fold": second_fold,
                    "spearman_rank_correlation": (
                        spearman_rank_correlation(first, second)
                    ),
                    f"top_{top_k}_jaccard": top_k_jaccard(
                        first,
                        second,
                        raw_features,
                        top_k,
                    ),
                }
            )

    return (
        pd.DataFrame(feature_records),
        pd.DataFrame(pair_records),
        pd.DataFrame(fold_records),
    )


def summarize_stability(
    fold_features: pd.DataFrame,
    global_importance: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Create one raw-feature stability summary across grouped refits."""

    records: list[dict[str, Any]] = []
    global_rank = global_importance.set_index("raw_feature")["importance_rank"]

    for feature, group in fold_features.groupby(
        "raw_feature",
        sort=False,
    ):
        signed = group["mean_signed_shap_highest_quartile"].to_numpy(dtype=float)
        median_signed = float(np.median(signed))
        median_sign = np.sign(median_signed)
        if median_sign == 0:
            direction_stability = float(np.mean(np.abs(signed) <= 1e-12))
        else:
            direction_stability = float(np.mean(np.sign(signed) == median_sign))

        ranks = group["importance_rank"].to_numpy(dtype=int)
        importance = group["mean_absolute_shap_log_odds"].to_numpy(dtype=float)
        records.append(
            {
                "raw_feature": feature,
                "global_importance_rank": int(global_rank.loc[feature]),
                "median_fold_mean_absolute_shap_log_odds": float(np.median(importance)),
                "minimum_fold_mean_absolute_shap_log_odds": float(importance.min()),
                "maximum_fold_mean_absolute_shap_log_odds": float(importance.max()),
                "mean_fold_rank": float(ranks.mean()),
                "minimum_fold_rank": int(ranks.min()),
                "maximum_fold_rank": int(ranks.max()),
                "top_k_fold_count": int((ranks <= top_k).sum()),
                "median_signed_shap_highest_quartile": median_signed,
                "highest_quartile_direction": direction_label(median_signed),
                "direction_stability": direction_stability,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("global_importance_rank")
        .reset_index(drop=True)
    )


def build_figure(
    width: float = 10.0,
    height: float = 6.0,
) -> tuple[Figure, Any]:
    """Create one consistently styled Matplotlib figure."""

    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    return figure, axis


def save_global_importance_figure(
    global_importance: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Save the top aggregate current-workforce explanation chart."""

    selected = global_importance.head(15).sort_values(
        "importance_rank",
        ascending=False,
    )
    figure, axis = build_figure(10.5, 7.0)
    axis.barh(
        selected["raw_feature"],
        selected["mean_absolute_shap_log_odds"],
        color="#2F6B8A",
    )
    axis.set_title("Current Workforce: Aggregate Logistic-Model Drivers")
    axis.set_xlabel("Mean absolute SHAP value (base-model log-odds)")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_fold_stability_figure(
    fold_features: pd.DataFrame,
    global_importance: pd.DataFrame,
    top_k: int,
    path: Path,
    dpi: int,
) -> None:
    """Save a heatmap of fold ranks for the globally important features."""

    top_features = global_importance.head(top_k)["raw_feature"].tolist()
    pivot = (
        fold_features.loc[fold_features["raw_feature"].isin(top_features)]
        .pivot(
            index="raw_feature",
            columns="fold",
            values="importance_rank",
        )
        .reindex(top_features)
    )

    figure, axis = build_figure(8.5, 6.5)
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        cmap="Blues_r",
        aspect="auto",
        vmin=1,
        vmax=max(top_k + 4, int(pivot.to_numpy().max())),
    )
    axis.set_title("Employee-Grouped Refit Importance Ranks")
    axis.set_xlabel("Stability fold")
    axis.set_ylabel("")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns)
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)

    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            axis.text(
                column,
                row,
                str(int(pivot.iloc[row, column])),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label="Importance rank")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_quartile_figure(
    quartiles: pd.DataFrame,
    global_importance: pd.DataFrame,
    labels: list[str],
    top_k: int,
    path: Path,
    dpi: int,
) -> None:
    """Save signed mean contributions across probability quartiles."""

    top_features = global_importance.head(top_k)["raw_feature"].tolist()
    pivot = (
        quartiles.loc[quartiles["raw_feature"].isin(top_features)]
        .pivot(
            index="raw_feature",
            columns="probability_quartile",
            values="mean_signed_shap_log_odds",
        )
        .reindex(index=top_features, columns=labels)
    )
    limit = float(np.abs(pivot.to_numpy(dtype=float)).max())

    figure, axis = build_figure(11.0, 6.5)
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        cmap="RdBu_r",
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_title("Mean Signed Model Contributions by Probability Quartile")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(top_features)))
    axis.set_yticklabels(top_features)
    figure.colorbar(
        image,
        ax=axis,
        label="Mean SHAP value (base-model log-odds)",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def validation_record(
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> dict[str, Any]:
    """Create one validation record."""

    return {
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "requirement": requirement,
        "details": details,
    }


def validate_outputs(
    inputs: dict[str, Any],
    explanation_policy: dict[str, Any],
    history: pd.DataFrame,
    current: pd.DataFrame,
    current_probability: np.ndarray,
    probability_difference: float,
    encoded_names: list[str],
    encoded_values: np.ndarray,
    grouped_values: np.ndarray,
    expected_log_odds: float,
    decision: np.ndarray,
    global_importance: pd.DataFrame,
    fold_features: pd.DataFrame,
    pairwise: pd.DataFrame,
    fold_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    aggregate_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Validate mathematical, stability, privacy, and governance contracts."""

    contract = explanation_policy["model_contract"]
    stability = explanation_policy["stability"]
    governance = explanation_policy["governance"]
    tolerance = float(explanation_policy["validation"]["numeric_tolerance"])
    probability_tolerance = float(
        explanation_policy["validation"]["current_probability_reconciliation_tolerance"]
    )
    top_k = int(stability["ranking_top_k"])

    additivity_error = float(
        np.max(np.abs(expected_log_odds + encoded_values.sum(axis=1) - decision))
    )
    grouping_error = float(
        np.max(np.abs(encoded_values.sum(axis=1) - grouped_values.sum(axis=1)))
    )
    pairwise_spearman = pairwise["spearman_rank_correlation"]
    jaccard_column = f"top_{top_k}_jaccard"
    pairwise_jaccard = pairwise[jaccard_column]
    top_features = set(global_importance.head(top_k)["raw_feature"])
    top_stability = stability_summary.loc[
        stability_summary["raw_feature"].isin(top_features),
        "direction_stability",
    ]
    policy_decision = pd.read_csv(DECISION_PATH)
    final_metrics = pd.read_csv(FINAL_MODEL_METRICS_PATH)
    current_unknown = (
        pd.read_csv(
            CURRENT_PATH,
            usecols=["attrition_next_12m"],
        )["attrition_next_12m"]
        .notna()
        .sum()
    )

    forbidden_columns = {
        "employee_id",
        "attrition_next_12m",
        "actual_attrition",
        "selected_for_human_review",
    }
    exposed_columns = sorted(
        set().union(*(set(table.columns) for table in aggregate_tables))
        & forbidden_columns
    )
    expected_figures = set(explanation_policy["figures"]["expected_files"])
    actual_figures = {path.name for path in FIGURE_DIR.glob("*.png")}

    records = [
        validation_record(
            "Configured base model remains selected",
            inputs["selected_model"] == contract["selected_base_model"],
            inputs["selected_model"],
            contract["selected_base_model"],
            "Checkpoint 54 explains the model selected in Checkpoint 41.",
        ),
        validation_record(
            "Configured calibration method remains selected",
            inputs["selected_method"] == contract["selected_calibration_method"],
            inputs["selected_method"],
            contract["selected_calibration_method"],
            "Current probabilities retain Checkpoint 42's approved scale.",
        ),
        validation_record(
            "Historical explanation training scope is exact",
            len(history) == int(contract["expected_historical_rows"]),
            len(history),
            int(contract["expected_historical_rows"]),
            "The deployment refit uses all completed historical snapshots.",
        ),
        validation_record(
            "Current explanation population is exact",
            len(current) == int(contract["expected_current_rows"]),
            len(current),
            int(contract["expected_current_rows"]),
            "All and only model-eligible active employees are explained.",
        ),
        validation_record(
            "Current calibrated scores reproduce Checkpoint 46",
            probability_difference <= probability_tolerance,
            probability_difference,
            f"<= {probability_tolerance}",
            "Explanation analysis starts from the deployed score population.",
        ),
        validation_record(
            "Current future outcomes remain unknown",
            int(current_unknown) == 0,
            int(current_unknown),
            0,
            "Current explanations do not use an unobserved future target.",
        ),
        validation_record(
            "Encoded explanation width is exact",
            len(encoded_names) == int(contract["expected_encoded_features"]),
            len(encoded_names),
            int(contract["expected_encoded_features"]),
            "Every fitted coefficient receives one encoded contribution.",
        ),
        validation_record(
            "Raw explanation width is exact",
            grouped_values.shape[1] == int(contract["expected_raw_features"]),
            grouped_values.shape[1],
            int(contract["expected_raw_features"]),
            "One-hot columns are grouped into policy-level raw features.",
        ),
        validation_record(
            "Linear SHAP values exactly reconstruct model log-odds",
            additivity_error <= tolerance,
            additivity_error,
            f"<= {tolerance}",
            "Base value plus feature contributions equals decision_function.",
        ),
        validation_record(
            "Raw feature grouping preserves total contribution",
            grouping_error <= tolerance,
            grouping_error,
            f"<= {tolerance}",
            "Grouping changes presentation but not the explanation total.",
        ),
        validation_record(
            "Expected number of grouped stability folds completed",
            len(fold_summary) == int(stability["employee_grouped_folds"]),
            len(fold_summary),
            int(stability["employee_grouped_folds"]),
            "Every configured employee-grouped refit is represented.",
        ),
        validation_record(
            "Stability folds are employee-disjoint",
            int(fold_summary["employee_overlap"].max()) == 0,
            int(fold_summary["employee_overlap"].max()),
            0,
            "Repeated employee histories never cross fit and holdout groups.",
        ),
        validation_record(
            "Minimum pairwise importance-rank stability is acceptable",
            float(pairwise_spearman.min())
            >= float(stability["minimum_pairwise_spearman"]),
            float(pairwise_spearman.min()),
            f">= {stability['minimum_pairwise_spearman']}",
            "Important feature ordering is similar across grouped refits.",
        ),
        validation_record(
            "Median pairwise importance-rank stability is acceptable",
            float(pairwise_spearman.median())
            >= float(stability["minimum_median_pairwise_spearman"]),
            float(pairwise_spearman.median()),
            (f">= {stability['minimum_median_pairwise_spearman']}"),
            "The typical fold pair preserves the global importance order.",
        ),
        validation_record(
            "Minimum pairwise top-k overlap is acceptable",
            float(pairwise_jaccard.min())
            >= float(stability["minimum_pairwise_top_k_jaccard"]),
            float(pairwise_jaccard.min()),
            (f">= {stability['minimum_pairwise_top_k_jaccard']}"),
            "The most important feature set is not driven by one refit.",
        ),
        validation_record(
            "Top-feature direction stability is acceptable",
            float(top_stability.min())
            >= float(stability["minimum_top_feature_direction_stability"]),
            float(top_stability.min()),
            (f">= {stability['minimum_top_feature_direction_stability']}"),
            "Highest-quartile driver directions persist across refits.",
        ),
        validation_record(
            "Saved explanation tables are aggregate only",
            not exposed_columns,
            exposed_columns,
            [],
            "No employee ID, outcome, or review-selection field is exported.",
        ),
        validation_record(
            "Employee-level explanation export is prohibited",
            not bool(
                explanation_policy["explanation"]["save_employee_level_explanations"]
            )
            and not bool(governance["row_level_explanation_export_permitted"]),
            False,
            False,
            "Employee-level values exist only temporarily in memory.",
        ),
        validation_record(
            "Final test remains frozen and once-only",
            bool(policy_decision.iloc[0]["policy_frozen_before_test"])
            and bool(policy_decision.iloc[0]["test_evaluated_once"])
            and len(final_metrics) == 1,
            (
                "frozen="
                f"{policy_decision.iloc[0]['policy_frozen_before_test']}; "
                "test_once="
                f"{policy_decision.iloc[0]['test_evaluated_once']}"
            ),
            "frozen=True; test_once=True",
            "No test result or intervention rule is recalculated.",
        ),
        validation_record(
            "Explanation governance is human-review only",
            bool(governance["human_review_required"])
            and not bool(governance["automatic_employment_action_permitted"])
            and not bool(governance["policy_retuning_permitted"]),
            (
                "human_review="
                f"{governance['human_review_required']}; "
                "automatic_action="
                f"{governance['automatic_employment_action_permitted']}; "
                "retuning="
                f"{governance['policy_retuning_permitted']}"
            ),
            ("human_review=True; automatic_action=False; retuning=False"),
            "Explanations are diagnostic associations, not actions.",
        ),
        validation_record(
            "All explanation figures were generated",
            expected_figures == actual_figures,
            sorted(actual_figures),
            sorted(expected_figures),
            "Aggregate charts support reviewer-facing interpretation.",
        ),
    ]

    table = pd.DataFrame(records)
    failures = table.loc[table["status"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(
            "Retention explanation validation failed:\n"
            + failures.to_string(index=False)
        )
    return table


def main() -> None:
    """Run the complete aggregate explanation and stability analysis."""

    explanation_policy = load_explanation_policy()
    inputs = load_inputs_before_test_access()
    development, test, _ = load_labeled_history_after_freeze(inputs)
    history = pd.concat([development, test], ignore_index=True)

    eligible_levels = set(
        inputs["feature_policy"]["model_population"]["eligible_organizational_levels"]
    )
    current = (
        inputs["current_features"]
        .loc[inputs["current_features"]["organizational_level"].isin(eligible_levels)]
        .copy()
    )

    print("Reproducing current calibrated probabilities")
    recomputed_probability, _ = fit_grouped_sigmoid_model(
        history,
        current,
        inputs,
        use_committed_folds=False,
    )
    saved_scores = pd.read_csv(CURRENT_SCORE_PATH)[
        ["employee_id", "attrition_probability"]
    ]
    probability_check = (
        current[["employee_id"]]
        .assign(recomputed_probability=recomputed_probability)
        .merge(
            saved_scores,
            on="employee_id",
            how="inner",
            validate="one_to_one",
        )
    )
    probability_difference = float(
        np.max(
            np.abs(
                probability_check["recomputed_probability"]
                - probability_check["attrition_probability"]
            )
        )
    )
    current_probability = probability_check["attrition_probability"].to_numpy(
        dtype=float
    )

    print("Fitting final current-score explanation model")
    model, history_features, current_features = fit_explanation_model(
        history,
        current,
        inputs,
    )
    encoded_names, encoded_values, expected_log_odds, decision = linear_shap_values(
        model,
        history_features,
        current_features,
    )
    raw_features = selected_features(inputs["feature_policy"])
    grouped_values = group_encoded_contributions(
        encoded_names,
        encoded_values,
        raw_features,
        list(inputs["feature_policy"]["categorical_features"]),
    )

    global_importance = build_global_importance(
        raw_features,
        grouped_values,
    )
    labels = list(explanation_policy["explanation"]["probability_quartiles"])
    quartiles = build_quartile_summary(
        raw_features,
        grouped_values,
        current_probability,
        labels,
    )
    driver_frequency = build_driver_frequency(
        raw_features,
        grouped_values,
    )

    fold_features, pairwise, fold_summary = calculate_fold_stability(
        history,
        current,
        current_probability,
        inputs,
        explanation_policy,
    )
    top_k = int(explanation_policy["stability"]["ranking_top_k"])
    stability_summary = summarize_stability(
        fold_features,
        global_importance,
        top_k,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for existing in FIGURE_DIR.glob("*.png"):
        existing.unlink()

    dpi = int(explanation_policy["figures"]["dpi"])
    save_global_importance_figure(
        global_importance,
        FIGURE_DIR / "retention_global_shap_importance.png",
        dpi,
    )
    save_fold_stability_figure(
        fold_features,
        global_importance,
        top_k,
        FIGURE_DIR / "retention_shap_fold_stability.png",
        dpi,
    )
    save_quartile_figure(
        quartiles,
        global_importance,
        labels,
        top_k,
        FIGURE_DIR / "retention_shap_probability_quartiles.png",
        dpi,
    )

    aggregate_tables = [
        global_importance,
        fold_features,
        pairwise,
        fold_summary,
        stability_summary,
        quartiles,
        driver_frequency,
    ]
    validation = validate_outputs(
        inputs,
        explanation_policy,
        history,
        current,
        current_probability,
        probability_difference,
        encoded_names,
        encoded_values,
        grouped_values,
        expected_log_odds,
        decision,
        global_importance,
        fold_features,
        pairwise,
        fold_summary,
        stability_summary,
        aggregate_tables,
    )

    global_importance.to_csv(GLOBAL_IMPORTANCE_PATH, index=False)
    fold_features.to_csv(FOLD_STABILITY_PATH, index=False)
    pairwise.to_csv(PAIRWISE_STABILITY_PATH, index=False)
    stability_summary.to_csv(
        STABILITY_SUMMARY_PATH,
        index=False,
    )
    quartiles.to_csv(QUARTILE_PATH, index=False)
    driver_frequency.to_csv(
        DRIVER_FREQUENCY_PATH,
        index=False,
    )
    validation.to_csv(VALIDATION_PATH, index=False)

    print("\nCURRENT AGGREGATE MODEL DRIVERS")
    print(global_importance.head(15).to_string(index=False))

    print("\nEXPLANATION STABILITY")
    stability_table = pd.DataFrame(
        [
            {
                "metric": "Minimum pairwise Spearman rank correlation",
                "value": float(pairwise["spearman_rank_correlation"].min()),
            },
            {
                "metric": "Median pairwise Spearman rank correlation",
                "value": float(pairwise["spearman_rank_correlation"].median()),
            },
            {
                "metric": f"Minimum pairwise top-{top_k} Jaccard",
                "value": float(pairwise[f"top_{top_k}_jaccard"].min()),
            },
            {
                "metric": "Minimum top-feature direction stability",
                "value": float(
                    stability_summary.head(top_k)["direction_stability"].min()
                ),
            },
            {
                "metric": "Maximum exact-additivity error",
                "value": float(
                    np.max(
                        np.abs(
                            expected_log_odds + encoded_values.sum(axis=1) - decision
                        )
                    )
                ),
            },
        ]
    )
    print(stability_table.to_string(index=False))

    print("\nRETENTION EXPLANATION VALIDATION")
    print(validation.to_string(index=False))

    print(f"\nSaved explanation outputs to: {OUTPUT_DIR}")
    print("Employee-level explanations saved: 0; final-test or policy changes made: 0")
    print("\nRETENTION MODEL EXPLANATIONS AND STABILITY COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
