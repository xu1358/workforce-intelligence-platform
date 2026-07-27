"""Optimize and evaluate a retention-intervention policy.

Checkpoint 46 selects policy rules on the 2024 validation period, freezes
them, and then evaluates them once on the previously untouched 2025 test
period. The final selected policy is a 700-employee, budget-constrained
expected-value rule. After final evaluation, the model is refitted on all
historical periods to create an anonymized synthetic current-workforce plan.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

MATPLOTLIB_CACHE_DIR = (
    Path(tempfile.gettempdir())
    / "workforce_intelligence_matplotlib"
)
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from calibrate_retention_model import probability_logit
from compare_retention_models_v2 import (
    build_candidate_model,
    fit_candidate,
)
from retention_feature_policy import (
    load_feature_policy,
    normalize_feature_frame,
    selected_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROCESSED_DIR / "policy_figures"

POLICY_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "retention_policy.yaml"
)
COST_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "retention_cost_assumptions.yaml"
)
MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "model_comparison_v2.yaml"
)
CALIBRATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "retention_calibration.yaml"
)

DATA_PATH = PROCESSED_DIR / "retention_multi_snapshot.csv"
CURRENT_PATH = (
    PROCESSED_DIR / "current_active_scoring_population.csv"
)
ASSIGNMENT_PATH = (
    PROCESSED_DIR / "model_split_assignments.csv"
)
CALIBRATION_PREDICTION_PATH = (
    PROCESSED_DIR / "retention_calibration_predictions.csv"
)
CALIBRATION_SELECTION_PATH = (
    PROCESSED_DIR / "retention_calibration_selection.csv"
)
MODEL_SELECTION_PATH = (
    PROCESSED_DIR / "model_selection_decision_v2.csv"
)
RANKING_SUMMARY_PATH = (
    PROCESSED_DIR / "retention_ranking_summary.csv"
)
COST_SCENARIO_PATH = (
    PROCESSED_DIR / "retention_cost_scenarios.csv"
)

POLICY_COMPARISON_PATH = (
    PROCESSED_DIR / "retention_policy_comparison.csv"
)
THRESHOLD_PATH = (
    PROCESSED_DIR / "retention_policy_threshold_analysis.csv"
)
DECISION_PATH = (
    PROCESSED_DIR / "retention_policy_decision.csv"
)
FINAL_TEST_PREDICTION_PATH = (
    PROCESSED_DIR / "retention_final_test_predictions.csv"
)
FINAL_MODEL_METRICS_PATH = (
    PROCESSED_DIR / "retention_final_test_model_metrics.csv"
)
CALIBRATION_FOLD_PATH = (
    PROCESSED_DIR / "retention_final_calibration_folds.csv"
)
SENSITIVITY_PATH = (
    PROCESSED_DIR / "retention_policy_sensitivity.csv"
)
BOOTSTRAP_PATH = (
    PROCESSED_DIR / "retention_policy_test_bootstrap.csv"
)
CURRENT_SCORE_PATH = (
    PROCESSED_DIR / "current_retention_policy_scores.csv"
)
CURRENT_SUMMARY_PATH = (
    PROCESSED_DIR / "current_retention_policy_summary.csv"
)
CURRENT_GROUP_PATH = (
    PROCESSED_DIR / "current_retention_policy_group_summary.csv"
)
VALIDATION_PATH = (
    PROCESSED_DIR / "retention_policy_validation.csv"
)

POLICY_KEYS = [
    "no_intervention",
    "fixed_threshold",
    "cost_optimal_threshold",
    "top_fraction",
    "top_count",
    "budget_expected_value",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")

    return loaded


def require_files(paths: list[Path]) -> None:
    """Raise one readable error when an input is missing."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 46 input files:\n"
            + "\n".join(missing)
        )


def selected_row(
    path: Path,
    flag_column: str,
    value_column: str,
) -> str:
    """Read the one selected value from a prior decision file."""

    table = pd.read_csv(path)
    selected = table.loc[table[flag_column].eq(True)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one selected row in: {path}"
        )
    return str(selected.iloc[0][value_column])


def feature_columns(
    feature_policy: dict[str, Any],
) -> list[str]:
    """Return metadata, economics, and model features without target."""

    columns = [
        "employee_id",
        "snapshot_date",
        "organizational_level",
        "department_name",
        "employment_type",
        "job_level",
        "base_salary",
        *selected_features(feature_policy),
    ]
    return list(dict.fromkeys(columns))


def load_inputs_before_test_access() -> dict[str, Any]:
    """Load policy-selection inputs without the reserved test target."""

    required = [
        POLICY_CONFIG_PATH,
        COST_CONFIG_PATH,
        MODEL_CONFIG_PATH,
        CALIBRATION_CONFIG_PATH,
        DATA_PATH,
        CURRENT_PATH,
        ASSIGNMENT_PATH,
        CALIBRATION_PREDICTION_PATH,
        CALIBRATION_SELECTION_PATH,
        MODEL_SELECTION_PATH,
        RANKING_SUMMARY_PATH,
        COST_SCENARIO_PATH,
    ]
    require_files(required)

    policy = load_yaml(POLICY_CONFIG_PATH)
    cost_policy = load_yaml(COST_CONFIG_PATH)
    model_policy = load_yaml(MODEL_CONFIG_PATH)
    calibration_policy = load_yaml(CALIBRATION_CONFIG_PATH)
    feature_policy = load_feature_policy()

    selected_model = selected_row(
        MODEL_SELECTION_PATH,
        "selected_model",
        "model",
    )
    selected_method = selected_row(
        CALIBRATION_SELECTION_PATH,
        "selected_method",
        "method",
    )

    assignments = pd.read_csv(ASSIGNMENT_PATH)
    if "attrition_next_12m" in assignments.columns:
        raise ValueError(
            "Split assignments must not contain the target."
        )

    columns = feature_columns(feature_policy)
    features = pd.read_csv(DATA_PATH, usecols=columns)
    features["snapshot_date"] = features[
        "snapshot_date"
    ].astype(str)
    current = pd.read_csv(CURRENT_PATH, usecols=columns)
    current["snapshot_date"] = current["snapshot_date"].astype(str)

    assignment_features = features.merge(
        assignments,
        on=["employee_id", "snapshot_date"],
        how="inner",
        validate="one_to_one",
    )
    validation_features = assignment_features.loc[
        assignment_features["primary_split"].eq("validation")
    ].copy()
    test_features = assignment_features.loc[
        assignment_features["primary_split"].eq("test")
    ].copy()

    selected_predictions = pd.read_csv(
        CALIBRATION_PREDICTION_PATH
    )
    selected_predictions = selected_predictions.loc[
        selected_predictions["method"].eq(selected_method)
    ].copy()
    selected_predictions["snapshot_date"] = selected_predictions[
        "snapshot_date"
    ].astype(str)

    validation = validation_features.merge(
        selected_predictions,
        on=["employee_id", "snapshot_date"],
        how="inner",
        validate="one_to_one",
    )
    validation["base_salary"] = pd.to_numeric(
        validation["base_salary"],
        errors="raise",
    )
    validation["attrition_probability"] = pd.to_numeric(
        validation["attrition_probability"],
        errors="raise",
    )
    validation["actual_attrition"] = validation[
        "actual_attrition"
    ].astype(int)

    cost_scenarios = pd.read_csv(COST_SCENARIO_PATH)
    ranking_summary = pd.read_csv(RANKING_SUMMARY_PATH)

    return {
        "policy": policy,
        "cost_policy": cost_policy,
        "model_policy": model_policy,
        "calibration_policy": calibration_policy,
        "feature_policy": feature_policy,
        "selected_model": selected_model,
        "selected_method": selected_method,
        "assignments": assignments,
        "validation": validation,
        "test_features": test_features,
        "current_features": current,
        "cost_scenarios": cost_scenarios,
        "ranking_summary": ranking_summary,
    }


def reference_economics(
    policy: dict[str, Any],
    cost_policy: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the named Checkpoint 45 reference assumptions."""

    reference = policy["reference_economic_scenario"]

    replacement = next(
        item
        for item in cost_policy["replacement_impact_scenarios"]
        if item["name"] == reference["replacement_impact"]
    )
    intervention = next(
        item
        for item in cost_policy["intervention_cost_scenarios"]
        if item["name"] == reference["intervention_cost"]
    )
    effectiveness = next(
        item
        for item in cost_policy[
            "intervention_effectiveness_scenarios"
        ]
        if item["name"] == reference["intervention_effectiveness"]
    )

    return {
        "replacement_impact_scenario": replacement["name"],
        "replacement_cost_salary_multiplier": float(
            replacement["salary_multiplier"]
        ),
        "vacancy_days": int(replacement["vacancy_days"]),
        "time_to_productivity_days": int(
            replacement["time_to_productivity_days"]
        ),
        "training_hours_per_replacement": float(
            replacement["training_hours_per_replacement"]
        ),
        "coverage_hours_per_replacement": (
            float(replacement["vacancy_days"])
            * float(replacement["coverage_hours_per_vacancy_day"])
        ),
        "intervention_cost_scenario": intervention["name"],
        "intervention_cost_usd": float(
            intervention["cost_per_employee_usd"]
        ),
        "effectiveness_scenario": effectiveness["name"],
        "success_probability": float(
            effectiveness["success_probability"]
        ),
    }


def individual_economics(
    probability: np.ndarray,
    salary: np.ndarray,
    economics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return replacement cost, avoidable cost, and predicted net value."""

    replacement_cost = (
        salary
        * float(economics["replacement_cost_salary_multiplier"])
    )
    expected_avoidable_cost = (
        probability
        * float(economics["success_probability"])
        * replacement_cost
    )
    expected_net_value = (
        expected_avoidable_cost
        - float(economics["intervention_cost_usd"])
    )
    return (
        replacement_cost,
        expected_avoidable_cost,
        expected_net_value,
    )


def stable_descending_order(
    values: np.ndarray,
    employee_ids: np.ndarray,
) -> np.ndarray:
    """Rank descending values with employee ID as a stable tie-break."""

    return np.lexsort((employee_ids, -values))


def choose_cost_threshold(
    validation: pd.DataFrame,
    economics: dict[str, Any],
) -> tuple[float, int, pd.DataFrame]:
    """Choose a global threshold by predicted net value on validation."""

    probability = validation[
        "attrition_probability"
    ].to_numpy(dtype=float)
    salary = validation["base_salary"].to_numpy(dtype=float)
    employee_ids = validation["employee_id"].to_numpy(dtype=int)
    _, _, expected_net = individual_economics(
        probability,
        salary,
        economics,
    )

    order = stable_descending_order(probability, employee_ids)
    cumulative_net = np.cumsum(expected_net[order])
    cumulative_cost = (
        np.arange(1, len(order) + 1)
        * float(economics["intervention_cost_usd"])
    )
    table = pd.DataFrame(
        {
            "selected_count": np.arange(1, len(order) + 1),
            "probability_threshold": probability[order],
            "cumulative_predicted_net_value_usd": cumulative_net,
            "cumulative_intervention_cost_usd": cumulative_cost,
        }
    )

    best_position = int(np.argmax(cumulative_net))
    if float(cumulative_net[best_position]) <= 0:
        return 1.0, 0, table

    best_count = best_position + 1
    threshold = float(probability[order[best_position]])
    table["selected_threshold"] = False
    table.loc[best_position, "selected_threshold"] = True
    return threshold, best_count, table


def freeze_policy(
    validation: pd.DataFrame,
    policy: dict[str, Any],
    economics: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select and hash policy rules before the test target is accessed."""

    threshold, threshold_count, threshold_table = (
        choose_cost_threshold(validation, economics)
    )
    candidate = policy["candidate_policies"]
    selected_key = str(
        policy["policy_selection"]["selected_policy"]
    )

    frozen = {
        "selected_policy": selected_key,
        "selected_policy_display_name": candidate[selected_key][
            "display_name"
        ],
        "validation_snapshot": policy["model"][
            "validation_snapshot"
        ],
        "test_snapshot": policy["model"]["test_snapshot"],
        "fixed_threshold": float(
            candidate["fixed_threshold"][
                "probability_threshold"
            ]
        ),
        "cost_optimal_threshold": threshold,
        "cost_optimal_validation_selected_count": threshold_count,
        "top_fraction": float(
            candidate["top_fraction"]["fraction"]
        ),
        "top_count": int(candidate["top_count"]["count"]),
        "maximum_employees": int(
            candidate["budget_expected_value"][
                "maximum_employees"
            ]
        ),
        "budget_usd": float(
            candidate["budget_expected_value"]["budget_usd"]
        ),
        "require_positive_expected_net_value": bool(
            candidate["budget_expected_value"][
                "require_positive_expected_net_value"
            ]
        ),
        **economics,
        "selected_using": "2024 validation probabilities only",
        "test_target_accessed_during_selection": False,
        "post_test_changes_allowed": False,
    }
    payload = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
    )
    frozen["policy_sha256"] = sha256(
        payload.encode("utf-8")
    ).hexdigest()
    return frozen, threshold_table


def policy_flags(
    frame: pd.DataFrame,
    frozen: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Apply all frozen policy definitions to one population."""

    probability = frame["attrition_probability"].to_numpy(dtype=float)
    salary = frame["base_salary"].to_numpy(dtype=float)
    employee_ids = frame["employee_id"].to_numpy(dtype=int)
    _, _, expected_net = individual_economics(
        probability,
        salary,
        frozen,
    )
    rows = len(frame)

    flags: dict[str, np.ndarray] = {
        "no_intervention": np.zeros(rows, dtype=bool),
        "fixed_threshold": (
            probability >= float(frozen["fixed_threshold"])
        ),
        "cost_optimal_threshold": (
            probability
            >= float(frozen["cost_optimal_threshold"])
        ),
    }

    probability_order = stable_descending_order(
        probability,
        employee_ids,
    )
    top_fraction_count = max(
        1,
        int(np.ceil(rows * float(frozen["top_fraction"]))),
    )
    top_fraction = np.zeros(rows, dtype=bool)
    top_fraction[probability_order[:top_fraction_count]] = True
    flags["top_fraction"] = top_fraction

    top_count = np.zeros(rows, dtype=bool)
    top_count[
        probability_order[
            : min(int(frozen["top_count"]), rows)
        ]
    ] = True
    flags["top_count"] = top_count

    expected_order = stable_descending_order(
        expected_net,
        employee_ids,
    )
    budget_selection = np.zeros(rows, dtype=bool)
    spent = 0.0
    selected = 0
    cost = float(frozen["intervention_cost_usd"])
    for position in expected_order:
        if selected >= int(frozen["maximum_employees"]):
            break
        if spent + cost > float(frozen["budget_usd"]) + 1e-9:
            break
        if (
            bool(frozen["require_positive_expected_net_value"])
            and expected_net[position] <= 0
        ):
            break
        budget_selection[position] = True
        spent += cost
        selected += 1
    flags["budget_expected_value"] = budget_selection

    return flags


def calculate_policy_metrics(
    frame: pd.DataFrame,
    selected: np.ndarray,
    policy_key: str,
    display_name: str,
    period: str,
    economics: dict[str, Any],
) -> dict[str, Any]:
    """Calculate expected and outcome-aligned policy metrics."""

    probability = frame["attrition_probability"].to_numpy(dtype=float)
    salary = frame["base_salary"].to_numpy(dtype=float)
    target = frame["actual_attrition"].to_numpy(dtype=int)
    replacement_cost, expected_avoidable, _ = individual_economics(
        probability,
        salary,
        economics,
    )
    success = float(economics["success_probability"])
    intervention_cost = float(economics["intervention_cost_usd"])

    selected_count = int(selected.sum())
    positive_cases = int(target.sum())
    selected_positives = int(target[selected].sum())
    spend = selected_count * intervention_cost
    expected_avoided_cost = float(
        expected_avoidable[selected].sum()
    )
    expected_net = expected_avoided_cost - spend
    expected_preventions = float(
        (probability[selected] * success).sum()
    )

    outcome_avoided_cost = float(
        (
            target[selected]
            * success
            * replacement_cost[selected]
        ).sum()
    )
    outcome_net = outcome_avoided_cost - spend
    outcome_preventions = float(selected_positives * success)
    precision = (
        float(selected_positives / selected_count)
        if selected_count
        else 0.0
    )
    capture = (
        float(selected_positives / positive_cases)
        if positive_cases
        else 0.0
    )
    baseline = float(target.mean())

    return {
        "evaluation_period": period,
        "policy_key": policy_key,
        "policy": display_name,
        "population_rows": len(frame),
        "positive_cases": positive_cases,
        "observed_attrition_rate": baseline,
        "selected_count": selected_count,
        "selection_rate": float(selected_count / len(frame)),
        "selected_positive_cases": selected_positives,
        "precision": precision,
        "capture_rate": capture,
        "lift": (
            float(precision / baseline)
            if baseline > 0
            else 0.0
        ),
        "intervention_spend_usd": spend,
        "expected_prevented_departures": expected_preventions,
        "expected_avoided_replacement_cost_usd": (
            expected_avoided_cost
        ),
        "expected_net_value_usd": expected_net,
        "expected_roi": (
            float(expected_net / spend)
            if spend > 0
            else np.nan
        ),
        "outcome_aligned_expected_preventions": outcome_preventions,
        "outcome_aligned_avoided_replacement_cost_usd": (
            outcome_avoided_cost
        ),
        "outcome_aligned_net_value_usd": outcome_net,
        "outcome_aligned_roi": (
            float(outcome_net / spend)
            if spend > 0
            else np.nan
        ),
        "average_selected_probability": (
            float(probability[selected].mean())
            if selected_count
            else np.nan
        ),
        "average_selected_salary": (
            float(salary[selected].mean())
            if selected_count
            else np.nan
        ),
    }


def compare_policies(
    frame: pd.DataFrame,
    frozen: dict[str, Any],
    policy: dict[str, Any],
    period: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Apply and compare every candidate policy."""

    flags = policy_flags(frame, frozen)
    rows = []
    for key in POLICY_KEYS:
        rows.append(
            calculate_policy_metrics(
                frame,
                flags[key],
                key,
                policy["candidate_policies"][key]["display_name"],
                period,
                frozen,
            )
        )
    return pd.DataFrame(rows), flags


def load_labeled_history_after_freeze(
    inputs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load historical outcomes only after the policy hash exists."""

    feature_policy = inputs["feature_policy"]
    columns = [
        *feature_columns(feature_policy),
        str(feature_policy["target_column"]),
    ]
    columns = list(dict.fromkeys(columns))
    labeled = pd.read_csv(DATA_PATH, usecols=columns)
    labeled["snapshot_date"] = labeled["snapshot_date"].astype(str)
    labeled = labeled.merge(
        inputs["assignments"],
        on=["employee_id", "snapshot_date"],
        how="inner",
        validate="one_to_one",
    )
    labeled[str(feature_policy["target_column"])] = labeled[
        str(feature_policy["target_column"])
    ].astype(int)

    development = labeled.loc[
        labeled["primary_split"].isin(["train", "validation"])
    ].copy()
    test = labeled.loc[
        labeled["primary_split"].eq("test")
    ].copy()
    return development, test, 1


def fit_grouped_sigmoid_model(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    inputs: dict[str, Any],
    use_committed_folds: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit a grouped OOF sigmoid calibrator and score another period."""

    feature_policy = inputs["feature_policy"]
    model_policy = inputs["model_policy"]
    calibration_policy = inputs["calibration_policy"]
    selected_model = inputs["selected_model"]
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

    split_pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
    if use_committed_folds:
        fold_values = sorted(
            training["group_cv_fold"].dropna().astype(int).unique()
        )
        for fold in fold_values:
            validation_index = np.flatnonzero(
                training["group_cv_fold"].astype(int).eq(fold)
            )
            fit_index = np.flatnonzero(
                training["group_cv_fold"].astype(int).ne(fold)
            )
            split_pairs.append((fit_index, validation_index, fold))
    else:
        folds = int(
            inputs["policy"]["final_model"][
                "grouped_calibration_folds"
            ]
        )
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(
                inputs["policy"]["final_model"]["random_seed"]
            ),
        )
        for fold, (fit_index, validation_index) in enumerate(
            splitter.split(
                training_features,
                target,
                groups=training["employee_id"].to_numpy(),
            ),
            start=1,
        ):
            split_pairs.append((fit_index, validation_index, fold))

    oof_probability = np.full(len(training), np.nan, dtype=float)
    records: list[dict[str, Any]] = []

    for fit_index, calibration_index, fold in split_pairs:
        print(
            "Generating grouped final calibration probabilities: "
            f"fold {fold}/{len(split_pairs)}"
        )
        model = build_candidate_model(
            selected_model,
            model_policy,
            feature_policy,
        )
        model = fit_candidate(
            model,
            selected_model,
            training_features.iloc[fit_index],
            target[fit_index],
            model_policy,
        )
        fold_probability = np.asarray(
            model.predict_proba(
                training_features.iloc[calibration_index]
            )[:, 1],
            dtype=float,
        )
        oof_probability[calibration_index] = fold_probability
        fit_employees = set(
            training.iloc[fit_index]["employee_id"].astype(int)
        )
        calibration_employees = set(
            training.iloc[calibration_index][
                "employee_id"
            ].astype(int)
        )
        records.append(
            {
                "training_scope": (
                    "Train and validation"
                    if use_committed_folds
                    else "All historical periods"
                ),
                "fold": fold,
                "fit_rows": len(fit_index),
                "calibration_rows": len(calibration_index),
                "calibration_positive_cases": int(
                    target[calibration_index].sum()
                ),
                "employee_overlap": len(
                    fit_employees & calibration_employees
                ),
            }
        )

    if np.isnan(oof_probability).any():
        raise ValueError(
            "Every calibration-training row needs an OOF score."
        )

    sigmoid = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=int(
            inputs["policy"]["final_model"]["random_seed"]
        ),
    )
    sigmoid.fit(
        probability_logit(
            oof_probability,
            calibration_policy,
        ),
        target,
    )

    final_model = build_candidate_model(
        selected_model,
        model_policy,
        feature_policy,
    )
    final_model = fit_candidate(
        final_model,
        selected_model,
        training_features,
        target,
        model_policy,
    )
    raw_probability = np.asarray(
        final_model.predict_proba(scoring_features)[:, 1],
        dtype=float,
    )
    calibrated = np.asarray(
        sigmoid.predict_proba(
            probability_logit(
                raw_probability,
                calibration_policy,
            )
        )[:, 1],
        dtype=float,
    )
    return calibrated, pd.DataFrame(records)


def build_final_test_frame(
    test: pd.DataFrame,
    probability: np.ndarray,
    frozen: dict[str, Any],
) -> pd.DataFrame:
    """Create the once-only final test prediction table."""

    frame = test.copy()
    frame["base_salary"] = pd.to_numeric(
        frame["base_salary"],
        errors="raise",
    )
    frame["actual_attrition"] = frame[
        "attrition_next_12m"
    ].astype(int)
    frame["attrition_probability"] = probability
    replacement, avoidable, net = individual_economics(
        probability,
        frame["base_salary"].to_numpy(dtype=float),
        frozen,
    )
    frame["replacement_cost_usd"] = replacement
    frame["predicted_avoidable_cost_usd"] = avoidable
    frame["predicted_net_value_usd"] = net
    return frame


def final_model_metrics(
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Report final probability and ranking metrics once."""

    target = test["actual_attrition"].to_numpy(dtype=int)
    probability = test[
        "attrition_probability"
    ].to_numpy(dtype=float)
    order = np.argsort(-probability, kind="stable")
    top_count = max(1, int(np.ceil(len(test) * 0.10)))
    top_target = target[order[:top_count]]
    positive_rate = float(target.mean())
    precision = float(top_target.mean())

    return pd.DataFrame(
        [
            {
                "evaluation_period": "2025 final test",
                "rows": len(test),
                "positive_cases": int(target.sum()),
                "positive_rate": positive_rate,
                "mean_predicted_probability": float(
                    probability.mean()
                ),
                "mean_calibration_gap": float(
                    probability.mean() - positive_rate
                ),
                "brier_score": float(
                    brier_score_loss(target, probability)
                ),
                "pr_auc": float(
                    average_precision_score(target, probability)
                ),
                "roc_auc": float(
                    roc_auc_score(target, probability)
                ),
                "top_decile_count": top_count,
                "top_decile_precision": precision,
                "top_decile_capture": float(
                    top_target.sum() / target.sum()
                ),
                "top_decile_lift": float(
                    precision / positive_rate
                ),
            }
        ]
    )


def build_sensitivity(
    frames: dict[str, pd.DataFrame],
    flags_by_period: dict[str, dict[str, np.ndarray]],
    policy: dict[str, Any],
    cost_policy: dict[str, Any],
) -> pd.DataFrame:
    """Revalue every fixed selection under all 27 cost scenarios."""

    records: list[dict[str, Any]] = []
    for replacement in cost_policy[
        "replacement_impact_scenarios"
    ]:
        for intervention in cost_policy[
            "intervention_cost_scenarios"
        ]:
            for effectiveness in cost_policy[
                "intervention_effectiveness_scenarios"
            ]:
                economics = {
                    "replacement_cost_salary_multiplier": float(
                        replacement["salary_multiplier"]
                    ),
                    "intervention_cost_usd": float(
                        intervention["cost_per_employee_usd"]
                    ),
                    "success_probability": float(
                        effectiveness["success_probability"]
                    ),
                }
                scenario_id = "-".join(
                    [
                        str(replacement["name"]).lower(),
                        str(intervention["name"]).lower(),
                        str(effectiveness["name"]).lower(),
                    ]
                )
                for period, frame in frames.items():
                    for key in POLICY_KEYS:
                        metrics = calculate_policy_metrics(
                            frame,
                            flags_by_period[period][key],
                            key,
                            policy["candidate_policies"][key][
                                "display_name"
                            ],
                            period,
                            economics,
                        )
                        records.append(
                            {
                                "scenario_id": scenario_id,
                                "replacement_impact_scenario": (
                                    replacement["name"]
                                ),
                                "intervention_cost_scenario": (
                                    intervention["name"]
                                ),
                                "effectiveness_scenario": (
                                    effectiveness["name"]
                                ),
                                "replacement_cost_salary_multiplier": (
                                    economics[
                                        "replacement_cost_salary_multiplier"
                                    ]
                                ),
                                "intervention_cost_usd": economics[
                                    "intervention_cost_usd"
                                ],
                                "success_probability": economics[
                                    "success_probability"
                                ],
                                **metrics,
                            }
                        )
    return pd.DataFrame(records)


def bootstrap_selected_test_policy(
    test: pd.DataFrame,
    selected: np.ndarray,
    frozen: dict[str, Any],
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Bootstrap the frozen selected policy on the final test period."""

    target = test["actual_attrition"].to_numpy(dtype=int)
    positive_positions = np.flatnonzero(target == 1)
    negative_positions = np.flatnonzero(target == 0)
    iterations = int(
        policy["test_evaluation"]["bootstrap_iterations"]
    )
    rng = np.random.default_rng(
        int(policy["test_evaluation"]["random_seed"])
    )
    records = []

    for iteration in range(1, iterations + 1):
        positions = np.concatenate(
            [
                rng.choice(
                    positive_positions,
                    size=len(positive_positions),
                    replace=True,
                ),
                rng.choice(
                    negative_positions,
                    size=len(negative_positions),
                    replace=True,
                ),
            ]
        )
        sample = test.iloc[positions].reset_index(drop=True)
        sample_selected = selected[positions]
        metrics = calculate_policy_metrics(
            sample,
            sample_selected,
            str(frozen["selected_policy"]),
            str(frozen["selected_policy_display_name"]),
            "2025 final test bootstrap",
            frozen,
        )
        records.append(
            {
                "iteration": iteration,
                "selected_count": metrics["selected_count"],
                "selected_positive_cases": metrics[
                    "selected_positive_cases"
                ],
                "capture_rate": metrics["capture_rate"],
                "expected_net_value_usd": metrics[
                    "expected_net_value_usd"
                ],
                "outcome_aligned_net_value_usd": metrics[
                    "outcome_aligned_net_value_usd"
                ],
            }
        )
        if iteration % 100 == 0:
            print(
                "Completed final policy bootstrap: "
                f"{iteration}/{iterations}"
            )

    return pd.DataFrame(records)


def bootstrap_interval(
    values: pd.Series,
    confidence_level: float,
) -> tuple[float, float]:
    """Return an equal-tailed bootstrap interval."""

    alpha = 1.0 - confidence_level
    return (
        float(values.quantile(alpha / 2.0)),
        float(values.quantile(1.0 - alpha / 2.0)),
    )


def score_current_population(
    all_history: pd.DataFrame,
    current: pd.DataFrame,
    inputs: dict[str, Any],
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit on all history and create a human-review current plan."""

    current = current.copy()
    eligible_levels = set(
        inputs["feature_policy"]["model_population"][
            "eligible_organizational_levels"
        ]
    )
    current = current.loc[
        current["organizational_level"].isin(eligible_levels)
    ].copy()
    current["base_salary"] = pd.to_numeric(
        current["base_salary"],
        errors="raise",
    )

    probability, folds = fit_grouped_sigmoid_model(
        all_history,
        current,
        inputs,
        use_committed_folds=False,
    )
    current["attrition_probability"] = probability
    replacement, avoidable, net = individual_economics(
        probability,
        current["base_salary"].to_numpy(dtype=float),
        frozen,
    )
    current["replacement_cost_usd"] = replacement
    current["predicted_avoidable_cost_usd"] = avoidable
    current["predicted_net_value_usd"] = net
    selected = policy_flags(
        current,
        frozen,
    )[str(frozen["selected_policy"])]
    current["selected_for_human_review"] = selected

    order = stable_descending_order(
        current["predicted_net_value_usd"].to_numpy(dtype=float),
        current["employee_id"].to_numpy(dtype=int),
    )
    rank = np.empty(len(current), dtype=int)
    rank[order] = np.arange(1, len(current) + 1)
    current["expected_value_rank"] = rank

    output_columns = [
        "employee_id",
        "snapshot_date",
        "department_name",
        "organizational_level",
        "employment_type",
        "job_level",
        "base_salary",
        "attrition_probability",
        "replacement_cost_usd",
        "predicted_avoidable_cost_usd",
        "predicted_net_value_usd",
        "expected_value_rank",
        "selected_for_human_review",
    ]
    scores = current[output_columns].sort_values(
        "expected_value_rank"
    )

    chosen = scores["selected_for_human_review"].eq(True)
    summary = pd.DataFrame(
        [
            {
                "snapshot_date": str(
                    scores["snapshot_date"].iloc[0]
                ),
                "eligible_employees": len(scores),
                "selected_for_human_review": int(chosen.sum()),
                "intervention_budget_usd": float(
                    frozen["budget_usd"]
                ),
                "projected_intervention_spend_usd": float(
                    chosen.sum()
                    * float(frozen["intervention_cost_usd"])
                ),
                "projected_expected_prevented_departures": float(
                    (
                        scores.loc[
                            chosen,
                            "attrition_probability",
                        ]
                        * float(frozen["success_probability"])
                    ).sum()
                ),
                "projected_expected_avoided_cost_usd": float(
                    scores.loc[
                        chosen,
                        "predicted_avoidable_cost_usd",
                    ].sum()
                ),
                "projected_expected_net_value_usd": float(
                    scores.loc[
                        chosen,
                        "predicted_net_value_usd",
                    ].sum()
                ),
                "human_review_required": True,
                "automatic_employment_action_permitted": False,
            }
        ]
    )

    group = (
        scores.groupby("department_name", as_index=False)
        .agg(
            eligible_employees=("employee_id", "size"),
            selected_for_human_review=(
                "selected_for_human_review",
                "sum",
            ),
            average_probability=(
                "attrition_probability",
                "mean",
            ),
            average_expected_net_value_usd=(
                "predicted_net_value_usd",
                "mean",
            ),
        )
    )
    group["selection_rate"] = (
        group["selected_for_human_review"]
        / group["eligible_employees"]
    )
    return scores, summary, group, folds


def make_figures(
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    selected_policy: str,
    policy: dict[str, Any],
) -> list[str]:
    """Create three restrained policy-comparison figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dpi = int(policy["figures"]["dpi"])
    created: list[str] = []

    test = comparison.loc[
        comparison["evaluation_period"].eq("2025 final test")
    ].copy()

    figure: Figure
    figure, axis = plt.subplots(figsize=(10, 5.6))
    axis.barh(
        test["policy"],
        test["outcome_aligned_net_value_usd"],
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_title(
        "Final test policy value under reference assumptions"
    )
    axis.set_xlabel("Outcome-aligned net value (USD)")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = FIGURE_DIR / "policy_net_value_comparison.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    created.append(path.name)

    figure, axis = plt.subplots(figsize=(10, 5.6))
    axis.barh(test["policy"], test["capture_rate"] * 100.0)
    axis.set_title("Final test attrition capture by policy")
    axis.set_xlabel("Observed attrition cases captured (%)")
    axis.set_ylabel("")
    axis.set_xlim(left=0)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = FIGURE_DIR / "policy_capture_comparison.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    created.append(path.name)

    selected_sensitivity = sensitivity.loc[
        sensitivity["policy_key"].eq(selected_policy)
        & sensitivity["evaluation_period"].eq("2025 final test")
    ].copy()
    selected_sensitivity = selected_sensitivity.sort_values(
        "outcome_aligned_net_value_usd"
    ).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(10, 5.6))
    axis.scatter(
        np.arange(1, len(selected_sensitivity) + 1),
        selected_sensitivity[
            "outcome_aligned_net_value_usd"
        ],
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_title(
        "Selected policy sensitivity across 27 cost scenarios"
    )
    axis.set_xlabel("Scenarios ordered from lowest to highest value")
    axis.set_ylabel("Outcome-aligned net value (USD)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = FIGURE_DIR / "scenario_sensitivity.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    created.append(path.name)

    return created


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


def validate_results(
    inputs: dict[str, Any],
    frozen: dict[str, Any],
    comparison: pd.DataFrame,
    test: pd.DataFrame,
    test_target_access_count: int,
    folds: pd.DataFrame,
    sensitivity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    current_scores: pd.DataFrame,
    current_summary: pd.DataFrame,
    figures: list[str],
) -> pd.DataFrame:
    """Validate selection, final evaluation, economics, and ethics."""

    checks: list[dict[str, Any]] = []
    policy = inputs["policy"]
    validation_policy = policy["validation"]
    tolerance = float(validation_policy["numeric_tolerance"])

    add_check(
        checks,
        "Configured base model remains selected",
        inputs["selected_model"]
        == policy["model"]["selected_base_model"],
        inputs["selected_model"],
        policy["model"]["selected_base_model"],
        "Policy evaluation must reuse the selected model.",
    )
    add_check(
        checks,
        "Configured calibration method remains selected",
        inputs["selected_method"]
        == policy["model"]["selected_calibration_method"],
        inputs["selected_method"],
        policy["model"]["selected_calibration_method"],
        "Policy evaluation must reuse the approved probability scale.",
    )
    add_check(
        checks,
        "Policy was frozen before test-target access",
        (
            bool(frozen["policy_sha256"])
            and not bool(
                frozen["test_target_accessed_during_selection"]
            )
        ),
        (
            f"hash={str(frozen['policy_sha256'])[:12]}...; "
            "test access during selection=False"
        ),
        "Frozen hash exists; no test access during selection",
        "Test outcomes cannot influence the policy definition.",
    )
    add_check(
        checks,
        "Test target is accessed exactly once",
        test_target_access_count == 1,
        test_target_access_count,
        1,
        "The final test is a once-only evaluation.",
    )
    add_check(
        checks,
        "Final test snapshot is exact",
        test["snapshot_date"].astype(str).unique().tolist()
        == [policy["model"]["test_snapshot"]],
        test["snapshot_date"].astype(str).unique().tolist(),
        [policy["model"]["test_snapshot"]],
        "Only the reserved chronological period is evaluated.",
    )
    add_check(
        checks,
        "Final test population is sufficiently large",
        (
            len(test)
            >= int(validation_policy["minimum_test_rows"])
            and int(test["actual_attrition"].sum())
            >= int(
                validation_policy[
                    "minimum_test_positive_cases"
                ]
            )
        ),
        (
            f"{len(test)} rows; "
            f"{int(test['actual_attrition'].sum())} positives"
        ),
        (
            f">= {validation_policy['minimum_test_rows']} rows; "
            f">= {validation_policy['minimum_test_positive_cases']} "
            "positives"
        ),
        "Final metrics require a useful future population.",
    )
    probability = test[
        "attrition_probability"
    ].to_numpy(dtype=float)
    add_check(
        checks,
        "Final test probabilities are finite and bounded",
        (
            np.isfinite(probability).all()
            and probability.min()
            >= float(validation_policy["probability_minimum"])
            and probability.max()
            <= float(validation_policy["probability_maximum"])
        ),
        f"min={probability.min():.6f}; max={probability.max():.6f}",
        "All finite and inside [0, 1]",
        "The final calibrated model must output valid probabilities.",
    )
    candidate_counts = (
        comparison.groupby("evaluation_period")["policy_key"]
        .nunique()
        .to_dict()
    )
    expected_policies = int(
        validation_policy["expected_candidate_policies"]
    )
    add_check(
        checks,
        "Every candidate policy is compared in both periods",
        all(
            value == expected_policies
            for value in candidate_counts.values()
        ),
        candidate_counts,
        f"{expected_policies} per period",
        "Fixed threshold, top-k, and cost policies cannot be omitted.",
    )
    add_check(
        checks,
        "Old fixed 0.50 threshold remains a comparator",
        "fixed_threshold" in set(comparison["policy_key"]),
        sorted(comparison["policy_key"].unique()),
        "fixed_threshold included",
        "The former no-op rule is retained for honest comparison.",
    )
    add_check(
        checks,
        "Selected policy is the constrained expected-value rule",
        frozen["selected_policy"] == "budget_expected_value",
        frozen["selected_policy"],
        "budget_expected_value",
        "The choice follows the committed capacity question.",
    )
    selected_rows = comparison.loc[
        comparison["policy_key"].eq(frozen["selected_policy"])
    ]
    add_check(
        checks,
        "Selected policy respects the 700-employee limit",
        selected_rows["selected_count"].le(
            int(validation_policy["maximum_selected_employees"])
        ).all(),
        selected_rows[
            ["evaluation_period", "selected_count"]
        ].to_dict("records"),
        (
            f"<= "
            f"{validation_policy['maximum_selected_employees']}"
        ),
        "Both validation and test apply the frozen capacity.",
    )
    add_check(
        checks,
        "Selected policy stays within budget",
        selected_rows["intervention_spend_usd"].le(
            float(validation_policy["maximum_budget_usd"])
            + tolerance
        ).all(),
        selected_rows[
            ["evaluation_period", "intervention_spend_usd"]
        ].to_dict("records"),
        f"<= ${validation_policy['maximum_budget_usd']:,.0f}",
        "The explicit resource constraint is enforced.",
    )
    formula_difference = (
        comparison["expected_net_value_usd"]
        - (
            comparison["expected_avoided_replacement_cost_usd"]
            - comparison["intervention_spend_usd"]
        )
    ).abs().max()
    add_check(
        checks,
        "Expected-net-value formulas reconcile",
        float(formula_difference) <= tolerance,
        float(formula_difference),
        f"<= {tolerance}",
        "Expected net value equals expected avoided cost minus spend.",
    )
    add_check(
        checks,
        "Grouped calibration folds are employee-disjoint",
        int(folds["employee_overlap"].max()) == 0,
        int(folds["employee_overlap"].max()),
        0,
        "Repeated employee histories cannot cross OOF fit boundaries.",
    )
    scenario_counts = (
        sensitivity.groupby(
            ["evaluation_period", "policy_key"]
        )["scenario_id"]
        .nunique()
    )
    add_check(
        checks,
        "All cost scenarios are evaluated for every policy",
        scenario_counts.eq(
            int(validation_policy["expected_cost_scenarios"])
        ).all(),
        (
            f"minimum={int(scenario_counts.min())}; "
            f"maximum={int(scenario_counts.max())}"
        ),
        (
            f"{validation_policy['expected_cost_scenarios']} "
            "per policy and period"
        ),
        "Sensitivity includes unfavorable and favorable assumptions.",
    )
    add_check(
        checks,
        "Final policy bootstrap is complete",
        len(bootstrap)
        == int(
            policy["test_evaluation"]["bootstrap_iterations"]
        ),
        len(bootstrap),
        policy["test_evaluation"]["bootstrap_iterations"],
        "Uncertainty is measured without retuning the policy.",
    )
    current_selected = int(
        current_scores["selected_for_human_review"].sum()
    )
    add_check(
        checks,
        "Current plan respects the frozen employee limit",
        current_selected
        <= int(validation_policy["maximum_selected_employees"]),
        current_selected,
        f"<= {validation_policy['maximum_selected_employees']}",
        "Current scoring reuses the tested policy capacity.",
    )
    add_check(
        checks,
        "Current plan contains no known future outcome",
        "attrition_next_12m" not in current_scores.columns,
        "attrition_next_12m" in current_scores.columns,
        False,
        "Current employees have no observable future label.",
    )
    add_check(
        checks,
        "Current plan is human-review only",
        (
            bool(
                current_summary.iloc[0][
                    "human_review_required"
                ]
            )
            and not bool(
                current_summary.iloc[0][
                    "automatic_employment_action_permitted"
                ]
            )
        ),
        (
            "human_review=True; "
            "automatic_action=False"
        ),
        "Human review required; automatic action prohibited",
        "Risk scores are decision support, not employment decisions.",
    )
    expected_figures = sorted(
        policy["figures"]["expected_files"]
    )
    add_check(
        checks,
        "All policy figures were generated",
        sorted(figures) == expected_figures,
        sorted(figures),
        expected_figures,
        "Decision results need reproducible visual summaries.",
    )

    validation = pd.DataFrame(checks)
    failed = validation.loc[validation["status"].eq("FAIL")]
    if not failed.empty:
        raise ValueError(
            "Checkpoint 46 validation failed:\n"
            + failed.to_string(index=False)
        )
    return validation


def decision_table(
    frozen: dict[str, Any],
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Create the final frozen-policy decision record."""

    selected = comparison.loc[
        comparison["policy_key"].eq(frozen["selected_policy"])
    ]
    validation = selected.loc[
        selected["evaluation_period"].eq("2024 validation")
    ].iloc[0]
    test = selected.loc[
        selected["evaluation_period"].eq("2025 final test")
    ].iloc[0]
    confidence = float(
        policy["test_evaluation"]["confidence_level"]
    )
    net_lower, net_upper = bootstrap_interval(
        bootstrap["outcome_aligned_net_value_usd"],
        confidence,
    )
    capture_lower, capture_upper = bootstrap_interval(
        bootstrap["capture_rate"],
        confidence,
    )

    return pd.DataFrame(
        [
            {
                **frozen,
                "policy_frozen_before_test": True,
                "test_evaluated_once": True,
                "validation_selected_count": int(
                    validation["selected_count"]
                ),
                "validation_expected_net_value_usd": float(
                    validation["expected_net_value_usd"]
                ),
                "validation_outcome_aligned_net_value_usd": float(
                    validation[
                        "outcome_aligned_net_value_usd"
                    ]
                ),
                "test_selected_count": int(test["selected_count"]),
                "test_expected_net_value_usd": float(
                    test["expected_net_value_usd"]
                ),
                "test_outcome_aligned_net_value_usd": float(
                    test["outcome_aligned_net_value_usd"]
                ),
                "test_outcome_aligned_net_value_lower_95": net_lower,
                "test_outcome_aligned_net_value_upper_95": net_upper,
                "test_capture_rate": float(test["capture_rate"]),
                "test_capture_rate_lower_95": capture_lower,
                "test_capture_rate_upper_95": capture_upper,
                "decision_status": "Final test completed; no retuning",
            }
        ]
    )


def print_results(
    frozen: dict[str, Any],
    comparison: pd.DataFrame,
    final_metrics: pd.DataFrame,
    decision: pd.DataFrame,
    current_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Print the policy results in a readable order."""

    comparison_columns = [
        "evaluation_period",
        "policy",
        "selected_count",
        "selected_positive_cases",
        "precision",
        "capture_rate",
        "intervention_spend_usd",
        "expected_net_value_usd",
        "outcome_aligned_net_value_usd",
    ]
    sensitivity_selected = sensitivity.loc[
        sensitivity["policy_key"].eq(frozen["selected_policy"])
        & sensitivity["evaluation_period"].eq("2025 final test")
    ]
    positive_expected_share = float(
        sensitivity_selected["expected_net_value_usd"].gt(0).mean()
    )
    positive_outcome_share = float(
        sensitivity_selected[
            "outcome_aligned_net_value_usd"
        ].gt(0).mean()
    )

    print("\nFROZEN POLICY")
    print(
        pd.DataFrame(
            [
                {
                    "selected_policy": frozen[
                        "selected_policy_display_name"
                    ],
                    "maximum_employees": frozen[
                        "maximum_employees"
                    ],
                    "budget_usd": frozen["budget_usd"],
                    "reference_intervention_cost_usd": frozen[
                        "intervention_cost_usd"
                    ],
                    "reference_success_probability": frozen[
                        "success_probability"
                    ],
                    "policy_hash": (
                        str(frozen["policy_sha256"])[:16] + "..."
                    ),
                }
            ]
        ).to_string(index=False)
    )

    print("\nVALIDATION AND FINAL TEST POLICY COMPARISON")
    print(
        comparison[comparison_columns].to_string(index=False)
    )

    print("\nFINAL TEST MODEL METRICS")
    print(final_metrics.to_string(index=False))

    print("\nFINAL SELECTED-POLICY DECISION")
    decision_columns = [
        "selected_policy_display_name",
        "validation_selected_count",
        "validation_expected_net_value_usd",
        "test_selected_count",
        "test_expected_net_value_usd",
        "test_outcome_aligned_net_value_usd",
        "test_outcome_aligned_net_value_lower_95",
        "test_outcome_aligned_net_value_upper_95",
        "test_capture_rate",
        "test_capture_rate_lower_95",
        "test_capture_rate_upper_95",
    ]
    print(decision[decision_columns].to_string(index=False))

    print("\nCOST-SCENARIO ROBUSTNESS")
    print(
        pd.DataFrame(
            [
                {
                    "test_scenarios": len(sensitivity_selected),
                    "share_positive_predicted_value": (
                        positive_expected_share
                    ),
                    "share_positive_outcome_aligned_value": (
                        positive_outcome_share
                    ),
                }
            ]
        ).to_string(index=False)
    )

    print("\nCURRENT SYNTHETIC WORKFORCE PLAN")
    print(current_summary.to_string(index=False))

    print("\nRETENTION POLICY VALIDATION")
    print(validation.to_string(index=False))

    print(f"\nSaved policy comparison: {POLICY_COMPARISON_PATH}")
    print(
        "The final test was evaluated once after policy freeze; "
        "no post-test retuning was performed."
    )
    print(
        "\nRETENTION POLICY OPTIMIZATION COMPLETED SUCCESSFULLY"
    )


def main() -> None:
    """Run policy selection, final evaluation, and current scoring."""

    inputs = load_inputs_before_test_access()
    policy = inputs["policy"]
    economics = reference_economics(
        policy,
        inputs["cost_policy"],
    )

    # Policy selection uses the 2024 validation predictions only.
    frozen, threshold_table = freeze_policy(
        inputs["validation"],
        policy,
        economics,
    )
    validation_comparison, validation_flags = compare_policies(
        inputs["validation"],
        frozen,
        policy,
        "2024 validation",
    )

    # The policy is now hashed and frozen. Only now are historical targets
    # loaded for final model fitting and once-only test evaluation.
    development, test_labeled, test_target_access_count = (
        load_labeled_history_after_freeze(inputs)
    )
    test_probability, final_folds = fit_grouped_sigmoid_model(
        development,
        test_labeled,
        inputs,
        use_committed_folds=True,
    )
    test = build_final_test_frame(
        test_labeled,
        test_probability,
        frozen,
    )
    test_comparison, test_flags = compare_policies(
        test,
        frozen,
        policy,
        "2025 final test",
    )
    comparison = pd.concat(
        [validation_comparison, test_comparison],
        ignore_index=True,
    )
    final_metrics = final_model_metrics(test)

    frames = {
        "2024 validation": inputs["validation"],
        "2025 final test": test,
    }
    flags_by_period = {
        "2024 validation": validation_flags,
        "2025 final test": test_flags,
    }
    sensitivity = build_sensitivity(
        frames,
        flags_by_period,
        policy,
        inputs["cost_policy"],
    )

    selected_test_flag = test_flags[str(frozen["selected_policy"])]
    bootstrap = bootstrap_selected_test_policy(
        test,
        selected_test_flag,
        frozen,
        policy,
    )
    decision = decision_table(
        frozen,
        comparison,
        bootstrap,
        policy,
    )

    all_history = pd.concat(
        [development, test_labeled],
        ignore_index=True,
    )
    (
        current_scores,
        current_summary,
        current_group,
        deployment_folds,
    ) = score_current_population(
        all_history,
        inputs["current_features"],
        inputs,
        frozen,
    )
    folds = pd.concat(
        [final_folds, deployment_folds],
        ignore_index=True,
    )

    figures = make_figures(
        comparison,
        sensitivity,
        str(frozen["selected_policy"]),
        policy,
    )
    validation = validate_results(
        inputs,
        frozen,
        comparison,
        test,
        test_target_access_count,
        folds,
        sensitivity,
        bootstrap,
        current_scores,
        current_summary,
        figures,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(THRESHOLD_PATH, index=False)
    comparison.to_csv(POLICY_COMPARISON_PATH, index=False)
    decision.to_csv(DECISION_PATH, index=False)
    final_metrics.to_csv(FINAL_MODEL_METRICS_PATH, index=False)
    folds.to_csv(CALIBRATION_FOLD_PATH, index=False)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    current_scores.to_csv(CURRENT_SCORE_PATH, index=False)
    current_summary.to_csv(CURRENT_SUMMARY_PATH, index=False)
    current_group.to_csv(CURRENT_GROUP_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)

    test_output_columns = [
        "employee_id",
        "snapshot_date",
        "department_name",
        "organizational_level",
        "employment_type",
        "job_level",
        "base_salary",
        "actual_attrition",
        "attrition_probability",
        "replacement_cost_usd",
        "predicted_avoidable_cost_usd",
        "predicted_net_value_usd",
    ]
    test[test_output_columns].to_csv(
        FINAL_TEST_PREDICTION_PATH,
        index=False,
    )

    print_results(
        frozen,
        comparison,
        final_metrics,
        decision,
        current_summary,
        sensitivity,
        validation,
    )


if __name__ == "__main__":
    main()
