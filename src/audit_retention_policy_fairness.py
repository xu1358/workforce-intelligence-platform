"""Audit subgroup allocation under the exact frozen retention policy.

Checkpoint 64 separates the probability-only top-10% diagnostic from the
deployed budget-constrained expected-value policy. It imports the policy
implementation selected in Checkpoint 46, applies the saved frozen decision
to the 2024 validation population, and reports aggregate subgroup evidence.
The reserved final-test target is never read.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_model_fairness import add_derived_groups  # noqa: E402
from optimize_retention_policy import (  # noqa: E402
    individual_economics,
    policy_flags,
    stable_descending_order,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "config" / "retention_policy_fairness.yaml"
FAIRNESS_CONFIG_PATH = PROJECT_ROOT / "config" / "retention_fairness.yaml"
DATA_PATH = PROCESSED_DIR / "retention_multi_snapshot.csv"
ASSIGNMENT_PATH = PROCESSED_DIR / "model_split_assignments.csv"
CALIBRATION_SELECTION_PATH = (
    PROCESSED_DIR / "retention_calibration_selection.csv"
)
CALIBRATION_PREDICTION_PATH = (
    PROCESSED_DIR / "retention_calibration_predictions.csv"
)
POLICY_DECISION_PATH = PROCESSED_DIR / "retention_policy_decision.csv"
POLICY_COMPARISON_PATH = PROCESSED_DIR / "retention_policy_comparison.csv"

OUTPUT_DIR = PROCESSED_DIR / "retention_policy_fairness"
GROUP_METRIC_PATH = OUTPUT_DIR / "deployed_policy_group_metrics.csv"
DISPARITY_PATH = OUTPUT_DIR / "deployed_policy_disparities.csv"
RULE_COMPARISON_PATH = OUTPUT_DIR / "selection_rule_comparison.csv"
OVERALL_PATH = OUTPUT_DIR / "selection_rule_overall.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation_checks.csv"

KEY_COLUMNS = ["employee_id", "snapshot_date"]
BASE_GROUP_COLUMNS = [
    "approx_age",
    "education_level",
    "region",
    "employment_type",
    "department_name",
    "job_level",
    "organizational_level",
    "base_salary",
]
TARGET_COLUMN = "actual_attrition"
PROBABILITY_COLUMN = "attrition_probability"
POLICY_FLAG_COLUMN = "selected_by_deployed_policy"
PROXY_FLAG_COLUMN = "selected_by_probability_proxy"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML contract."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_files(paths: list[Path]) -> None:
    """Require every upstream artifact used by the audit."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing deployed-policy fairness inputs:\n" + "\n".join(missing)
        )


def normalize_frozen_decision(row: pd.Series) -> dict[str, Any]:
    """Convert one saved policy-decision row into policy_flags inputs."""

    frozen = {
        key: (None if pd.isna(value) else value)
        for key, value in row.to_dict().items()
    }
    boolean_fields = [
        "require_positive_expected_net_value",
        "test_target_accessed_during_selection",
        "post_test_changes_allowed",
        "policy_frozen_before_test",
        "test_evaluated_once",
    ]
    for field in boolean_fields:
        value = frozen.get(field)
        if isinstance(value, str):
            frozen[field] = value.strip().lower() == "true"
        elif value is not None:
            frozen[field] = bool(value)
    return frozen


def load_validation_population(
    config: dict[str, Any],
    fairness_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Load validation scores, group attributes, and the frozen policy."""

    require_files(
        [
            DATA_PATH,
            ASSIGNMENT_PATH,
            CALIBRATION_SELECTION_PATH,
            CALIBRATION_PREDICTION_PATH,
            POLICY_DECISION_PATH,
            POLICY_COMPARISON_PATH,
        ]
    )

    decision = pd.read_csv(POLICY_DECISION_PATH)
    if len(decision) != 1:
        raise ValueError("Expected exactly one frozen policy-decision row.")
    frozen = normalize_frozen_decision(decision.iloc[0])

    method_selection = pd.read_csv(CALIBRATION_SELECTION_PATH)
    selected_method = method_selection.loc[
        method_selection["selected_method"].eq(True), "method"
    ]
    if len(selected_method) != 1:
        raise ValueError("Expected exactly one selected calibration method.")

    assignments = pd.read_csv(ASSIGNMENT_PATH)
    assignments["snapshot_date"] = assignments["snapshot_date"].astype(str)
    validation_keys = assignments.loc[
        assignments["primary_split"].eq("validation"),
        [*KEY_COLUMNS, "primary_split"],
    ].copy()

    source = pd.read_csv(
        DATA_PATH,
        usecols=[*KEY_COLUMNS, *BASE_GROUP_COLUMNS],
    )
    source["snapshot_date"] = source["snapshot_date"].astype(str)
    validation = validation_keys.merge(
        source,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    predictions = pd.read_csv(
        CALIBRATION_PREDICTION_PATH,
        usecols=[
            *KEY_COLUMNS,
            "method",
            TARGET_COLUMN,
            PROBABILITY_COLUMN,
        ],
    )
    predictions["snapshot_date"] = predictions["snapshot_date"].astype(str)
    predictions = predictions.loc[
        predictions["method"].eq(str(selected_method.iloc[0])),
        [*KEY_COLUMNS, TARGET_COLUMN, PROBABILITY_COLUMN],
    ]
    validation = validation.merge(
        predictions,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    validation = add_derived_groups(validation, fairness_config)

    configured_snapshot = str(config["source_policy"]["validation_snapshot"])
    validation = validation.loc[
        validation["snapshot_date"].eq(configured_snapshot)
    ].copy()
    comparison = pd.read_csv(POLICY_COMPARISON_PATH)
    return validation, frozen, comparison


def apply_selection_rules(
    population: pd.DataFrame,
    frozen: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply the exact deployed rule and the labeled probability proxy."""

    audited = population.copy()
    policy_key = str(config["source_policy"]["selected_policy_key"])
    flags = policy_flags(audited, frozen)
    if policy_key not in flags:
        raise ValueError(f"Frozen policy implementation is missing: {policy_key}")
    audited[POLICY_FLAG_COLUMN] = flags[policy_key].astype(bool)

    probability = audited[PROBABILITY_COLUMN].to_numpy(dtype=float)
    employee_ids = audited["employee_id"].to_numpy(dtype=int)
    order = stable_descending_order(probability, employee_ids)
    selected_count = max(
        1,
        int(
            np.ceil(
                len(audited)
                * float(config["probability_proxy"]["fraction"])
            )
        ),
    )
    proxy = np.zeros(len(audited), dtype=bool)
    proxy[order[:selected_count]] = True
    audited[PROXY_FLAG_COLUMN] = proxy

    replacement, avoidable, net_value = individual_economics(
        probability,
        audited["base_salary"].to_numpy(dtype=float),
        frozen,
    )
    audited["replacement_cost_usd"] = replacement
    audited["predicted_avoidable_cost_usd"] = avoidable
    audited["predicted_net_value_usd"] = net_value
    return audited


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a rate or NaN when the denominator is zero."""

    return numerator / denominator if denominator else np.nan


def group_record(
    group: pd.DataFrame,
    attribute: str,
    group_value: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate deployed-policy metrics for one subgroup."""

    target = group[TARGET_COLUMN].to_numpy(dtype=int)
    selected = group[POLICY_FLAG_COLUMN].to_numpy(dtype=bool)
    positives = int(target.sum())
    negatives = int(len(group) - positives)
    selected_count = int(selected.sum())
    true_positive = int(((target == 1) & selected).sum())
    false_positive = int(((target == 0) & selected).sum())
    eligibility = config["comparison_eligibility"]
    eligible = (
        len(group) >= int(eligibility["minimum_group_rows"])
        and positives >= int(eligibility["minimum_positive_cases"])
        and negatives >= int(eligibility["minimum_negative_cases"])
        and selected_count >= int(eligibility["minimum_selected_rows"])
    )

    return {
        "attribute": attribute,
        "group": group_value,
        "group_label": f"{attribute}: {group_value}",
        "sample_size": len(group),
        "positive_cases": positives,
        "negative_cases": negatives,
        "selected_count": selected_count,
        "selection_rate": selected_count / len(group),
        "selected_positive_cases": true_positive,
        "true_positive_rate": safe_rate(true_positive, positives),
        "false_positive_rate": safe_rate(false_positive, negatives),
        "precision": safe_rate(true_positive, selected_count),
        "mean_probability": float(group[PROBABILITY_COLUMN].mean()),
        "mean_salary_usd": float(group["base_salary"].mean()),
        "selected_mean_salary_usd": (
            float(group.loc[selected, "base_salary"].mean())
            if selected_count
            else np.nan
        ),
        "selected_expected_net_value_usd": float(
            group.loc[selected, "predicted_net_value_usd"].sum()
        ),
        "eligible_for_disparity_comparison": eligible,
    }


def build_group_metrics(
    population: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build aggregate metrics for every configured subgroup."""

    records: list[dict[str, Any]] = []
    for attribute in config["group_attributes"]:
        grouped = population.groupby(str(attribute), dropna=False, sort=True)
        for group_value, group in grouped:
            records.append(
                group_record(
                    group,
                    str(attribute),
                    str(group_value),
                    config,
                )
            )
    return pd.DataFrame(records)


def rule_group_record(
    group: pd.DataFrame,
    attribute: str,
    group_value: str,
) -> dict[str, Any]:
    """Compare probability-proxy and deployed-policy selection in one group."""

    deployed = group[POLICY_FLAG_COLUMN].to_numpy(dtype=bool)
    proxy = group[PROXY_FLAG_COLUMN].to_numpy(dtype=bool)
    return {
        "attribute": attribute,
        "group": group_value,
        "sample_size": len(group),
        "deployed_selected_count": int(deployed.sum()),
        "deployed_selection_rate": float(deployed.mean()),
        "probability_proxy_selected_count": int(proxy.sum()),
        "probability_proxy_selection_rate": float(proxy.mean()),
        "selection_rate_difference_deployed_minus_proxy": float(
            deployed.mean() - proxy.mean()
        ),
        "selection_overlap_count": int((deployed & proxy).sum()),
    }


def build_rule_comparison(
    population: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compare both rules for every configured subgroup."""

    records: list[dict[str, Any]] = []
    for attribute in config["group_attributes"]:
        grouped = population.groupby(str(attribute), dropna=False, sort=True)
        for group_value, group in grouped:
            records.append(
                rule_group_record(group, str(attribute), str(group_value))
            )
    return pd.DataFrame(records)


def selection_summary(
    population: pd.DataFrame,
    flag_column: str,
    label: str,
) -> dict[str, Any]:
    """Summarize one selection rule over the full validation population."""

    selected = population[flag_column].to_numpy(dtype=bool)
    target = population[TARGET_COLUMN].to_numpy(dtype=int)
    positives = int(target.sum())
    selected_count = int(selected.sum())
    selected_positives = int(target[selected].sum())
    return {
        "selection_rule": label,
        "selected_count": selected_count,
        "selection_rate": selected_count / len(population),
        "selected_positive_cases": selected_positives,
        "precision": safe_rate(selected_positives, selected_count),
        "capture_rate": safe_rate(selected_positives, positives),
        "mean_selected_probability": float(
            population.loc[selected, PROBABILITY_COLUMN].mean()
        ),
        "mean_selected_salary_usd": float(
            population.loc[selected, "base_salary"].mean()
        ),
        "predicted_net_value_usd": float(
            population.loc[selected, "predicted_net_value_usd"].sum()
        ),
    }


def build_overall_summary(
    population: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build overall proxy-versus-policy evidence."""

    deployed = population[POLICY_FLAG_COLUMN].to_numpy(dtype=bool)
    proxy = population[PROXY_FLAG_COLUMN].to_numpy(dtype=bool)
    overlap = int((deployed & proxy).sum())
    union = int((deployed | proxy).sum())
    rows = [
        selection_summary(
            population,
            POLICY_FLAG_COLUMN,
            "Frozen budget-constrained expected value",
        ),
        selection_summary(
            population,
            PROXY_FLAG_COLUMN,
            str(config["probability_proxy"]["label"]),
        ),
    ]
    for row in rows:
        row["cross_rule_overlap_count"] = overlap
        row["cross_rule_jaccard"] = safe_rate(overlap, union)
    return pd.DataFrame(rows)


def metric_range(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[float, float, str, str]:
    """Return min/max values and their group labels."""

    valid = frame.dropna(subset=[metric]).copy()
    minimum_index = valid[metric].idxmin()
    maximum_index = valid[metric].idxmax()
    return (
        float(valid.loc[minimum_index, metric]),
        float(valid.loc[maximum_index, metric]),
        str(valid.loc[minimum_index, "group"]),
        str(valid.loc[maximum_index, "group"]),
    )


def build_disparities(
    group_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build max-minus-min deployed-policy disparity summaries."""

    thresholds = config["screening_thresholds"]
    records: list[dict[str, Any]] = []
    for attribute in config["group_attributes"]:
        eligible = group_metrics.loc[
            group_metrics["attribute"].eq(attribute)
            & group_metrics["eligible_for_disparity_comparison"].eq(True)
        ].copy()
        if len(eligible) < 2:
            continue
        selection_min, selection_max, selection_low, selection_high = (
            metric_range(eligible, "selection_rate")
        )
        tpr_min, tpr_max, tpr_low, tpr_high = metric_range(
            eligible,
            "true_positive_rate",
        )
        fpr_min, fpr_max, fpr_low, fpr_high = metric_range(
            eligible,
            "false_positive_rate",
        )
        selection_gap = selection_max - selection_min
        tpr_gap = tpr_max - tpr_min
        fpr_gap = fpr_max - fpr_min
        triggers = []
        if selection_gap > float(thresholds["selection_rate_difference"]):
            triggers.append("selection_rate")
        if tpr_gap > float(thresholds["true_positive_rate_difference"]):
            triggers.append("true_positive_rate")
        if fpr_gap > float(thresholds["false_positive_rate_difference"]):
            triggers.append("false_positive_rate")
        records.append(
            {
                "attribute": attribute,
                "eligible_groups": len(eligible),
                "selection_rate_difference": selection_gap,
                "lowest_selection_group": selection_low,
                "highest_selection_group": selection_high,
                "true_positive_rate_difference": tpr_gap,
                "lowest_true_positive_rate_group": tpr_low,
                "highest_true_positive_rate_group": tpr_high,
                "false_positive_rate_difference": fpr_gap,
                "lowest_false_positive_rate_group": fpr_low,
                "highest_false_positive_rate_group": fpr_high,
                "screening_review_flag": bool(triggers),
                "review_triggers": ", ".join(triggers) if triggers else "None",
            }
        )
    return pd.DataFrame(records)


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
    population: pd.DataFrame,
    frozen: dict[str, Any],
    policy_comparison: pd.DataFrame,
    group_metrics: pd.DataFrame,
    overall: pd.DataFrame,
    figures: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate exact-policy use, reconciliation, scope, and governance."""

    checks: list[dict[str, Any]] = []
    expected = config["expected"]
    tolerance = float(config["validation"]["numeric_tolerance"])
    policy_key = str(config["source_policy"]["selected_policy_key"])
    deployed = population[POLICY_FLAG_COLUMN].to_numpy(dtype=bool)
    proxy = population[PROXY_FLAG_COLUMN].to_numpy(dtype=bool)

    add_check(
        checks,
        "Saved frozen policy is the configured expected-value rule",
        str(frozen["selected_policy"]) == policy_key,
        frozen["selected_policy"],
        policy_key,
        "The audit cannot substitute a probability-only selection rule.",
    )
    add_check(
        checks,
        "Validation population and target contract are exact",
        len(population) == int(expected["validation_rows"])
        and int(population[TARGET_COLUMN].sum())
        == int(expected["validation_positive_cases"]),
        {
            "rows": len(population),
            "positive_cases": int(population[TARGET_COLUMN].sum()),
        },
        {
            "rows": expected["validation_rows"],
            "positive_cases": expected["validation_positive_cases"],
        },
        "The policy audit uses the same development-stage population.",
    )
    add_check(
        checks,
        "Deployed and proxy capacities are independently exact",
        int(deployed.sum()) == int(expected["deployed_selected_count"])
        and int(proxy.sum())
        == int(expected["probability_proxy_selected_count"]),
        {
            "deployed": int(deployed.sum()),
            "probability_proxy": int(proxy.sum()),
        },
        {
            "deployed": expected["deployed_selected_count"],
            "probability_proxy": expected[
                "probability_proxy_selected_count"
            ],
        },
        "The 700-person policy is not mislabeled as a 10% diagnostic.",
    )
    overlap = int((deployed & proxy).sum())
    add_check(
        checks,
        "Different selection rules are explicitly demonstrated",
        overlap == int(expected["deployed_proxy_overlap"])
        and not np.array_equal(deployed, proxy),
        overlap,
        expected["deployed_proxy_overlap"],
        "Low overlap proves the original subgroup rates cannot describe the policy.",
    )

    comparison_row = policy_comparison.loc[
        policy_comparison["evaluation_period"].eq(
            config["source_policy"]["validation_period"]
        )
        & policy_comparison["policy_key"].eq(policy_key)
    ]
    if len(comparison_row) != 1:
        raise ValueError("Expected one saved validation-policy comparison row.")
    comparison_row = comparison_row.iloc[0]
    selected_mean_salary = float(
        population.loc[deployed, "base_salary"].mean()
    )
    add_check(
        checks,
        "Reconstructed selection reproduces Checkpoint 46",
        int(comparison_row["selected_count"]) == int(deployed.sum())
        and abs(
            float(comparison_row["average_selected_salary"])
            - selected_mean_salary
        )
        <= tolerance,
        {
            "selected_count": int(deployed.sum()),
            "selected_mean_salary_usd": selected_mean_salary,
        },
        {
            "selected_count": int(comparison_row["selected_count"]),
            "selected_mean_salary_usd": float(
                comparison_row["average_selected_salary"]
            ),
        },
        "The shared policy_flags implementation reproduces the frozen result.",
    )
    add_check(
        checks,
        "Expected selected salary is reproduced",
        abs(
            selected_mean_salary
            - float(expected["validation_selected_mean_salary_usd"])
        )
        <= tolerance,
        selected_mean_salary,
        expected["validation_selected_mean_salary_usd"],
        "Salary-dependent replacement cost is present in the audited rule.",
    )

    headline_observed: dict[str, dict[str, int]] = {}
    headline_ok = True
    for attribute, expected_groups in expected[
        "headline_selected_counts"
    ].items():
        observed_groups: dict[str, int] = {}
        for group_name, expected_count in expected_groups.items():
            row = group_metrics.loc[
                group_metrics["attribute"].eq(attribute)
                & group_metrics["group"].eq(group_name)
            ]
            actual_count = (
                int(row.iloc[0]["selected_count"]) if len(row) == 1 else -1
            )
            observed_groups[group_name] = actual_count
            headline_ok &= actual_count == int(expected_count)
        headline_observed[attribute] = observed_groups
    add_check(
        checks,
        "Reported headline subgroup selections use deployed flags",
        headline_ok,
        headline_observed,
        expected["headline_selected_counts"],
        "The corrected results replace the old probability-only examples.",
    )

    add_check(
        checks,
        "Every configured policy subgroup is reported",
        sorted(group_metrics["attribute"].unique().tolist())
        == sorted(config["group_attributes"]),
        sorted(group_metrics["attribute"].unique().tolist()),
        sorted(config["group_attributes"]),
        "No configured dimension is silently omitted.",
    )
    rate_columns = [
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
    ]
    rates = group_metrics[rate_columns].to_numpy(dtype=float)
    valid_rates = rates[~np.isnan(rates)]
    add_check(
        checks,
        "All policy subgroup rates are valid",
        bool(((valid_rates >= 0) & (valid_rates <= 1)).all()),
        {
            "minimum": float(valid_rates.min()),
            "maximum": float(valid_rates.max()),
        },
        "All finite rates inside [0, 1]",
        "Allocation and outcome rates must be mathematically valid.",
    )
    add_check(
        checks,
        "Reserved final-test target remains unread",
        bool(config["governance"]["accesses_reserved_test_target"]) is False
        and set(population["snapshot_date"].unique())
        == {str(config["source_policy"]["validation_snapshot"])},
        {
            "accesses_reserved_test_target": False,
            "snapshots": sorted(population["snapshot_date"].unique()),
        },
        {
            "accesses_reserved_test_target": False,
            "snapshot": config["source_policy"]["validation_snapshot"],
        },
        "The audit does not reopen the once-only final test.",
    )
    exported_columns = set(group_metrics.columns) | set(overall.columns)
    prohibited = {"employee_id", "employee_name", "actual_attrition"}
    add_check(
        checks,
        "Saved policy-fairness evidence is aggregate only",
        not bool(exported_columns & prohibited)
        and bool(config["governance"]["export_employee_level_rows"]) is False,
        sorted(exported_columns & prohibited),
        [],
        "No employee-level review list or outcome row is exported.",
    )
    add_check(
        checks,
        "Frozen policy and analytical results remain unchanged",
        not any(
            bool(config["governance"][field])
            for field in [
                "changes_model_results",
                "changes_frozen_policy",
                "changes_dashboard_data",
            ]
        ),
        {
            field: config["governance"][field]
            for field in [
                "changes_model_results",
                "changes_frozen_policy",
                "changes_dashboard_data",
            ]
        },
        "All False",
        "This checkpoint corrects the audit layer, not the decision rule.",
    )
    add_check(
        checks,
        "All deployed-policy fairness figures were generated",
        sorted(figures) == sorted(config["figures"]["expected_files"]),
        sorted(figures),
        sorted(config["figures"]["expected_files"]),
        "Reviewer-facing evidence must be reproducible.",
    )

    validation = pd.DataFrame(checks)
    failed = validation.loc[validation["status"].eq("FAIL")]
    if not failed.empty:
        raise ValueError(
            "Deployed-policy fairness validation failed:\n"
            + failed.to_string(index=False)
        )
    return validation


def make_figures(
    group_metrics: pd.DataFrame,
    rule_comparison: pd.DataFrame,
    disparities: pd.DataFrame,
    config: dict[str, Any],
) -> list[str]:
    """Create aggregate reviewer-facing policy-fairness figures."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dpi = int(config["figures"]["dpi"])
    created: list[str] = []

    eligible = group_metrics.loc[
        group_metrics["eligible_for_disparity_comparison"].eq(True)
    ].sort_values("selection_rate")
    figure, axis = plt.subplots(figsize=(10, 10))
    axis.barh(eligible["group_label"], eligible["selection_rate"])
    axis.set_xlabel("Frozen expected-value policy selection rate")
    axis.set_title("Deployed Policy Selection by Eligible Subgroup")
    axis.xaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1.0)
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = OUTPUT_DIR / "deployed_policy_selection_by_group.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    created.append(path.name)

    headline = rule_comparison.loc[
        rule_comparison["attribute"].isin(
            ["department_name", "employment_type", "job_level"]
        )
    ].copy()
    headline["group_label"] = (
        headline["attribute"] + ": " + headline["group"]
    )
    headline = headline.sort_values("deployed_selection_rate")
    positions = np.arange(len(headline))
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.barh(
        positions - 0.2,
        headline["probability_proxy_selection_rate"],
        height=0.4,
        label="Probability-only top 10% diagnostic",
    )
    axis.barh(
        positions + 0.2,
        headline["deployed_selection_rate"],
        height=0.4,
        label="Frozen expected-value policy",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(headline["group_label"])
    axis.set_xlabel("Selection rate")
    axis.set_title("Probability Proxy Versus Deployed Policy")
    axis.xaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1.0)
    )
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = OUTPUT_DIR / "proxy_vs_deployed_selection.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    created.append(path.name)

    ordered = disparities.sort_values("selection_rate_difference")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(
        ordered["attribute"],
        ordered["selection_rate_difference"],
    )
    axis.axvline(
        float(config["screening_thresholds"]["selection_rate_difference"]),
        color="tab:orange",
        linestyle="--",
        label="Descriptive review trigger",
    )
    axis.set_xlabel("Maximum minus minimum selection rate")
    axis.set_title("Deployed-Policy Subgroup Selection Gaps")
    axis.xaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1.0)
    )
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = OUTPUT_DIR / "deployed_policy_disparity_summary.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    created.append(path.name)
    return created


def print_results(
    population: pd.DataFrame,
    overall: pd.DataFrame,
    rule_comparison: pd.DataFrame,
    group_metrics: pd.DataFrame,
    disparities: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Print the corrected policy-fairness evidence."""

    print("\nSELECTION-RULE OVERALL COMPARISON")
    print(overall.to_string(index=False))

    headline = rule_comparison.loc[
        (
            rule_comparison["attribute"].eq("department_name")
            & rule_comparison["group"].isin(
                ["Customer Support", "Engineering"]
            )
        )
        | (
            rule_comparison["attribute"].eq("employment_type")
            & rule_comparison["group"].isin(["Hourly", "Salaried"])
        )
        | (
            rule_comparison["attribute"].eq("job_level")
            & rule_comparison["group"].isin(["Level 1", "Level 3"])
        )
    ]
    print("\nCORRECTED HEADLINE SELECTION RATES")
    print(
        headline[
            [
                "attribute",
                "group",
                "sample_size",
                "probability_proxy_selection_rate",
                "deployed_selection_rate",
                "selection_rate_difference_deployed_minus_proxy",
            ]
        ].to_string(index=False)
    )

    print("\nDEPLOYED-POLICY SUBGROUP METRICS")
    print(
        group_metrics[
            [
                "attribute",
                "group",
                "sample_size",
                "selected_count",
                "selection_rate",
                "true_positive_rate",
                "false_positive_rate",
                "precision",
                "selected_mean_salary_usd",
                "eligible_for_disparity_comparison",
            ]
        ].to_string(index=False)
    )
    print("\nDEPLOYED-POLICY DISPARITY SUMMARY")
    print(disparities.to_string(index=False))
    print("\nDEPLOYED-POLICY FAIRNESS VALIDATION")
    print(validation.to_string(index=False))
    print(f"\nSaved aggregate outputs to: {OUTPUT_DIR}")
    print(
        "Employee-level rows saved: 0; final-test target rows accessed: 0; "
        "policy changes made: 0"
    )
    print(
        "\nDEPLOYED RETENTION POLICY FAIRNESS AUDIT "
        "COMPLETED SUCCESSFULLY"
    )


def main() -> None:
    """Run the deployed-policy subgroup audit."""

    config = load_yaml(CONFIG_PATH)
    fairness_config = load_yaml(FAIRNESS_CONFIG_PATH)
    population, frozen, policy_comparison = load_validation_population(
        config,
        fairness_config,
    )
    population = apply_selection_rules(population, frozen, config)
    group_metrics = build_group_metrics(population, config)
    rule_comparison = build_rule_comparison(population, config)
    overall = build_overall_summary(population, config)
    disparities = build_disparities(group_metrics, config)
    figures = make_figures(
        group_metrics,
        rule_comparison,
        disparities,
        config,
    )
    validation = validate_results(
        population,
        frozen,
        policy_comparison,
        group_metrics,
        overall,
        figures,
        config,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    group_metrics.to_csv(GROUP_METRIC_PATH, index=False)
    disparities.to_csv(DISPARITY_PATH, index=False)
    rule_comparison.to_csv(RULE_COMPARISON_PATH, index=False)
    overall.to_csv(OVERALL_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    print_results(
        population,
        overall,
        rule_comparison,
        group_metrics,
        disparities,
        validation,
    )


if __name__ == "__main__":
    main()
