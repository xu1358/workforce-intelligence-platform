"""Audit salary-driven allocation in the frozen retention policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retention_policy_equity.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "policy_equity"
FIGURE_DIR = OUTPUT_DIR / "figures"

MULTI_SNAPSHOT_PATH = PROCESSED_DIR / "retention_multi_snapshot.csv"
CALIBRATION_PATH = PROCESSED_DIR / "retention_calibration_predictions.csv"
FINAL_TEST_PATH = PROCESSED_DIR / "retention_final_test_predictions.csv"
CURRENT_PATH = PROCESSED_DIR / "current_retention_policy_scores.csv"
POLICY_COMPARISON_PATH = PROCESSED_DIR / "retention_policy_comparison.csv"

POPULATION_SUMMARY_PATH = OUTPUT_DIR / "policy_equity_population_summary.csv"
SALARY_BAND_PATH = OUTPUT_DIR / "policy_equity_salary_band_metrics.csv"
POLICY_COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "policy_equity_policy_comparison.csv"
RISK_SALARY_PATH = OUTPUT_DIR / "policy_equity_risk_salary_intersections.csv"
DEPARTMENT_PATH = OUTPUT_DIR / "policy_equity_department_summary.csv"
THRESHOLD_PATH = OUTPUT_DIR / "policy_equity_frozen_thresholds.csv"
VALIDATION_PATH = OUTPUT_DIR / "policy_equity_validation.csv"

PERIOD_ORDER = ["2024 validation", "2025 final test", "2026 current"]
POLICY_ORDER = [
    "frozen_expected_value",
    "probability_only",
    "salary_capped_expected_value",
    "constant_cost",
]
FORBIDDEN_OUTCOME_COLUMNS = {
    "actual_attrition",
    "attrition_next_12m",
    "termination_date",
    "termination_type",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")

    return config


def stable_descending_order(
    values: np.ndarray,
    employee_ids: np.ndarray,
) -> np.ndarray:
    """Rank descending values with employee ID as deterministic tie-break."""

    return np.lexsort((employee_ids, -values))


def expected_net_value(
    probability: np.ndarray,
    replacement_cost: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Calculate expected avoided cost less intervention cost."""

    economics = config["reference_economics"]
    return probability * float(
        economics["intervention_success_probability"]
    ) * replacement_cost * float(
        economics["replacement_cost_salary_multiplier"]
    ) - float(economics["intervention_cost_usd"])


def select_ranked_rows(
    score: np.ndarray,
    employee_ids: np.ndarray,
    config: dict[str, Any],
    require_positive: bool,
) -> np.ndarray:
    """Select the highest ranked rows under frozen capacity and budget."""

    cost = float(config["reference_economics"]["intervention_cost_usd"])
    capacity = int(config["capacity"]["maximum_employees"])
    budget_capacity = int(float(config["capacity"]["maximum_budget_usd"]) // cost)
    maximum = min(capacity, budget_capacity, len(score))
    order = stable_descending_order(score, employee_ids)
    selected = np.zeros(len(score), dtype=bool)

    for position in order:
        if int(selected.sum()) >= maximum:
            break
        if require_positive and score[position] <= 0:
            break
        selected[position] = True

    return selected


def policy_scores(
    frame: pd.DataFrame,
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    """Return ranking scores for the frozen and sensitivity policies."""

    probability = frame["attrition_probability"].to_numpy(dtype=float)
    salary = frame["base_salary"].to_numpy(dtype=float)
    capped_salary = np.minimum(
        salary,
        float(thresholds["validation_salary_cap_usd"]),
    )
    constant_cost = np.full(
        len(frame),
        float(thresholds["validation_constant_replacement_cost_usd"]),
    )

    return {
        "frozen_expected_value": expected_net_value(
            probability,
            salary,
            config,
        ),
        "probability_only": probability,
        "salary_capped_expected_value": expected_net_value(
            probability,
            capped_salary,
            config,
        ),
        "constant_cost": expected_net_value(
            probability,
            constant_cost,
            config,
        ),
    }


def apply_policies(
    frame: pd.DataFrame,
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Apply the original and committed sensitivity policies."""

    result = frame.copy()
    employee_ids = result["employee_id"].to_numpy(dtype=int)
    scores = policy_scores(result, config, thresholds)

    for policy_key in POLICY_ORDER:
        policy = config["sensitivity_policies"][policy_key]
        score = scores[policy_key]
        result[f"score_{policy_key}"] = score
        result[f"selected_{policy_key}"] = select_ranked_rows(
            score,
            employee_ids,
            config,
            bool(policy["require_positive_score"]),
        )

    return result


def quantile_labels(count: int, noun: str) -> list[str]:
    """Create ordered human-readable quantile labels."""

    labels = [f"Q{index}" for index in range(1, count + 1)]
    labels[0] = f"Q1 lowest {noun}"
    labels[-1] = f"Q{count} highest {noun}"
    return labels


def stable_quantiles(
    values: pd.Series,
    count: int,
    noun: str,
) -> pd.Series:
    """Assign deterministic equal-count quantiles, including tied values."""

    labels = quantile_labels(count, noun)
    ranked = values.rank(method="first")
    return pd.qcut(ranked, count, labels=labels)


def add_salary_groups(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Add absolute, population, within-level, and risk quantiles."""

    result = frame.copy()
    groups = config["salary_groups"]
    absolute = groups["absolute_bands"]
    result["absolute_salary_band"] = pd.cut(
        result["base_salary"],
        bins=absolute["edges"],
        labels=absolute["labels"],
        right=False,
        include_lowest=True,
    )
    result["salary_quintile"] = stable_quantiles(
        result["base_salary"],
        int(groups["population_quantiles"]),
        "pay",
    )
    result["risk_quintile"] = stable_quantiles(
        result["attrition_probability"],
        int(groups["risk_quantiles"]),
        "risk",
    )

    within_level = pd.Series(index=result.index, dtype="object")
    for _, indices in result.groupby("job_level", observed=True).groups.items():
        values = result.loc[indices, "base_salary"]
        within_level.loc[indices] = stable_quantiles(
            values,
            int(groups["within_job_level_quantiles"]),
            "within-level pay",
        ).astype(str)
    result["within_job_level_salary_quintile"] = within_level
    return result


def normalize_population(
    frame: pd.DataFrame,
    period: str,
    snapshot: str,
) -> pd.DataFrame:
    """Validate and normalize one policy-audit population."""

    result = frame.copy()
    result["snapshot_date"] = result["snapshot_date"].astype(str)
    result["base_salary"] = pd.to_numeric(result["base_salary"], errors="raise")
    result["attrition_probability"] = pd.to_numeric(
        result["attrition_probability"],
        errors="raise",
    )
    result["job_level"] = pd.to_numeric(
        result["job_level"],
        errors="raise",
    ).astype(int)
    result["period"] = period

    if set(result["snapshot_date"]) != {snapshot}:
        raise ValueError(
            f"{period} contains unexpected snapshots: "
            f"{sorted(result['snapshot_date'].unique())}"
        )
    if result["employee_id"].duplicated().any():
        raise ValueError(f"{period} contains duplicate employee IDs.")
    if not np.isfinite(result["base_salary"]).all():
        raise ValueError(f"{period} contains nonfinite salary values.")
    if not np.isfinite(result["attrition_probability"]).all():
        raise ValueError(f"{period} contains nonfinite probabilities.")
    return result


def load_populations(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """Load validation, final-test, and current data without outcomes."""

    snapshots = config["snapshots"]
    eligible_levels = set(config["model_eligible_levels"])
    panel_columns = [
        "employee_id",
        "snapshot_date",
        "organizational_level",
        "department_name",
        "employment_type",
        "job_level",
        "base_salary",
    ]
    score_columns = [
        "employee_id",
        "snapshot_date",
        "method",
        "attrition_probability",
    ]
    panel = pd.read_csv(
        MULTI_SNAPSHOT_PATH,
        usecols=panel_columns,
        low_memory=False,
    )
    panel = panel[
        panel["snapshot_date"].astype(str).eq(snapshots["validation"])
        & panel["organizational_level"].isin(eligible_levels)
    ]
    validation_scores = pd.read_csv(
        CALIBRATION_PATH,
        usecols=score_columns,
    )
    validation_scores = validation_scores[
        validation_scores["snapshot_date"].astype(str).eq(snapshots["validation"])
        & validation_scores["method"].eq(config["selected_calibration_method"])
    ].drop(columns="method")
    validation = validation_scores.merge(
        panel,
        on=["employee_id", "snapshot_date"],
        how="inner",
        validate="one_to_one",
    )

    policy_columns = [
        "employee_id",
        "snapshot_date",
        "department_name",
        "organizational_level",
        "employment_type",
        "job_level",
        "base_salary",
        "attrition_probability",
    ]
    final_test = pd.read_csv(FINAL_TEST_PATH, usecols=policy_columns)
    current = pd.read_csv(
        CURRENT_PATH,
        usecols=policy_columns + ["selected_for_human_review"],
    )
    current["committed_selection"] = current["selected_for_human_review"].astype(bool)
    current = current.drop(columns="selected_for_human_review")

    populations = {
        "2024 validation": normalize_population(
            validation,
            "2024 validation",
            snapshots["validation"],
        ),
        "2025 final test": normalize_population(
            final_test,
            "2025 final test",
            snapshots["final_test"],
        ),
        "2026 current": normalize_population(
            current,
            "2026 current",
            snapshots["current"],
        ),
    }
    loaded_columns = set().union(
        *(set(frame.columns) for frame in populations.values())
    )
    return populations, loaded_columns


def frozen_thresholds(
    validation: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float]:
    """Freeze salary cap and constant cost from validation salary only."""

    policies = config["sensitivity_policies"]
    cap_quantile = float(
        policies["salary_capped_expected_value"]["validation_salary_cap_quantile"]
    )
    constant_quantile = float(policies["constant_cost"]["validation_salary_quantile"])
    return {
        "validation_salary_cap_quantile": cap_quantile,
        "validation_salary_cap_usd": float(
            validation["base_salary"].quantile(cap_quantile)
        ),
        "validation_constant_salary_quantile": constant_quantile,
        "validation_constant_replacement_cost_usd": float(
            validation["base_salary"].quantile(constant_quantile)
        ),
    }


def policy_display_names(config: dict[str, Any]) -> dict[str, str]:
    """Return configured display names in policy order."""

    return {
        key: str(config["sensitivity_policies"][key]["display_name"])
        for key in POLICY_ORDER
    }


def build_policy_comparison(
    populations: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compare salary composition and value across policy variants."""

    display_names = policy_display_names(config)
    rows: list[dict[str, Any]] = []

    for period in PERIOD_ORDER:
        frame = populations[period]
        original = frame["selected_frozen_expected_value"].to_numpy(bool)
        original_net = frame["score_frozen_expected_value"].to_numpy(float)
        population_mean_salary = float(frame["base_salary"].mean())

        for policy_key in POLICY_ORDER:
            selected = frame[f"selected_{policy_key}"].to_numpy(bool)
            selected_count = int(selected.sum())
            overlap = int((selected & original).sum())
            union = int((selected | original).sum())
            rows.append(
                {
                    "period": period,
                    "policy_key": policy_key,
                    "policy": display_names[policy_key],
                    "population_rows": len(frame),
                    "selected_count": selected_count,
                    "selection_rate": selected_count / len(frame),
                    "population_mean_salary_usd": population_mean_salary,
                    "selected_mean_salary_usd": float(
                        frame.loc[selected, "base_salary"].mean()
                    ),
                    "selected_median_salary_usd": float(
                        frame.loc[selected, "base_salary"].median()
                    ),
                    "selected_to_population_mean_salary_ratio": (
                        float(frame.loc[selected, "base_salary"].mean())
                        / population_mean_salary
                    ),
                    "selected_mean_probability": float(
                        frame.loc[selected, "attrition_probability"].mean()
                    ),
                    "predicted_net_value_under_original_economics_usd": (
                        float(original_net[selected].sum())
                    ),
                    "overlap_with_frozen_policy": overlap,
                    "jaccard_with_frozen_policy": overlap / union,
                }
            )

    return pd.DataFrame(rows)


def group_policy_rows(
    frame: pd.DataFrame,
    period: str,
    band_type: str,
    band_column: str,
    segment: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate every policy within one salary grouping."""

    display_names = policy_display_names(config)
    rows: list[dict[str, Any]] = []

    for band, group in frame.groupby(band_column, observed=True, sort=False):
        for policy_key in POLICY_ORDER:
            selected = group[f"selected_{policy_key}"].astype(bool)
            count = int(selected.sum())
            rows.append(
                {
                    "period": period,
                    "band_type": band_type,
                    "segment": segment,
                    "salary_band": str(band),
                    "policy_key": policy_key,
                    "policy": display_names[policy_key],
                    "employees": len(group),
                    "mean_salary_usd": float(group["base_salary"].mean()),
                    "mean_probability": float(group["attrition_probability"].mean()),
                    "selected_count": count,
                    "selection_rate": count / len(group),
                }
            )
    return rows


def build_salary_band_metrics(
    populations: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize policy selection by absolute and relative salary groups."""

    rows: list[dict[str, Any]] = []

    for period in PERIOD_ORDER:
        frame = populations[period]
        rows.extend(
            group_policy_rows(
                frame,
                period,
                "absolute_salary_band",
                "absolute_salary_band",
                "All job levels",
                config,
            )
        )
        rows.extend(
            group_policy_rows(
                frame,
                period,
                "population_salary_quintile",
                "salary_quintile",
                "All job levels",
                config,
            )
        )
        for level, group in frame.groupby("job_level", observed=True):
            rows.extend(
                group_policy_rows(
                    group,
                    period,
                    "within_job_level_salary_quintile",
                    "within_job_level_salary_quintile",
                    f"Job Level {int(level)}",
                    config,
                )
            )

    return pd.DataFrame(rows)


def build_risk_salary_intersections(
    populations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Measure salary selection inside coarse probability strata."""

    rows: list[dict[str, Any]] = []

    for period in PERIOD_ORDER:
        frame = populations[period]
        grouped = frame.groupby(
            ["risk_quintile", "salary_quintile"],
            observed=True,
            sort=False,
        )
        for (risk_band, salary_band), group in grouped:
            selected = group["selected_frozen_expected_value"].astype(bool)
            rows.append(
                {
                    "period": period,
                    "risk_quintile": str(risk_band),
                    "salary_quintile": str(salary_band),
                    "employees": len(group),
                    "mean_probability": float(group["attrition_probability"].mean()),
                    "mean_salary_usd": float(group["base_salary"].mean()),
                    "selected_count": int(selected.sum()),
                    "selection_rate": float(selected.mean()),
                }
            )

    return pd.DataFrame(rows)


def build_department_summary(
    populations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Report frozen-policy salary concentration by department."""

    rows: list[dict[str, Any]] = []

    for period in PERIOD_ORDER:
        frame = populations[period]
        for department, group in frame.groupby(
            "department_name",
            observed=True,
        ):
            selected = group["selected_frozen_expected_value"].astype(bool)
            rows.append(
                {
                    "period": period,
                    "department_name": department,
                    "eligible_employees": len(group),
                    "selected_employees": int(selected.sum()),
                    "selection_rate": float(selected.mean()),
                    "population_mean_salary_usd": float(group["base_salary"].mean()),
                    "selected_mean_salary_usd": (
                        float(group.loc[selected, "base_salary"].mean())
                        if selected.any()
                        else np.nan
                    ),
                    "selected_salary_difference_usd": (
                        float(group.loc[selected, "base_salary"].mean())
                        - float(group["base_salary"].mean())
                        if selected.any()
                        else np.nan
                    ),
                    "mean_probability": float(group["attrition_probability"].mean()),
                }
            )

    return pd.DataFrame(rows)


def build_population_summary(
    populations: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create headline salary-allocation findings and review flags."""

    trigger = config["review_triggers"]
    rows: list[dict[str, Any]] = []
    high_salary_label = quantile_labels(5, "pay")[-1]
    low_salary_label = quantile_labels(5, "pay")[0]
    high_risk_label = quantile_labels(5, "risk")[-1]

    for period in PERIOD_ORDER:
        frame = populations[period]
        selected = frame["selected_frozen_expected_value"].astype(bool)
        salary_rates = frame.groupby(
            "salary_quintile",
            observed=True,
        )["selected_frozen_expected_value"].mean()
        lowest_rate = float(salary_rates.loc[low_salary_label])
        highest_rate = float(salary_rates.loc[high_salary_label])
        rate_ratio = highest_rate / lowest_rate if lowest_rate > 0 else np.inf
        highest_risk = frame[frame["risk_quintile"].eq(high_risk_label)]
        high_risk_rates = highest_risk.groupby(
            "salary_quintile",
            observed=True,
        )["selected_frozen_expected_value"].mean()
        within_risk_gap = float(
            high_risk_rates.loc[high_salary_label]
            - high_risk_rates.loc[low_salary_label]
        )
        probability_selected = frame["selected_probability_only"].astype(bool)
        overlap = int((selected & probability_selected).sum())
        salary_ratio = float(frame.loc[selected, "base_salary"].mean()) / float(
            frame["base_salary"].mean()
        )
        flags: list[str] = []
        if salary_ratio > float(trigger["selected_to_population_mean_salary_ratio"]):
            flags.append("selected_mean_salary")
        if rate_ratio > float(
            trigger["highest_to_lowest_salary_quintile_selection_rate_ratio"]
        ):
            flags.append("salary_quintile_selection")
        if within_risk_gap > float(
            trigger["within_highest_risk_quintile_selection_rate_gap"]
        ):
            flags.append("within_risk_salary_selection")

        rows.append(
            {
                "period": period,
                "population_rows": len(frame),
                "selected_count": int(selected.sum()),
                "population_mean_salary_usd": float(frame["base_salary"].mean()),
                "population_median_salary_usd": float(frame["base_salary"].median()),
                "selected_mean_salary_usd": float(
                    frame.loc[selected, "base_salary"].mean()
                ),
                "selected_median_salary_usd": float(
                    frame.loc[selected, "base_salary"].median()
                ),
                "selected_to_population_mean_salary_ratio": salary_ratio,
                "lowest_salary_quintile_selection_rate": lowest_rate,
                "highest_salary_quintile_selection_rate": highest_rate,
                "highest_to_lowest_selection_rate_ratio": rate_ratio,
                "highest_risk_quintile_salary_selection_gap": within_risk_gap,
                "overlap_with_probability_top_700": overlap,
                "frozen_policy_only_selections": int(
                    (selected & ~probability_selected).sum()
                ),
                "probability_policy_only_selections": int(
                    (~selected & probability_selected).sum()
                ),
                "review_flag": bool(flags),
                "review_triggers": ", ".join(flags) if flags else "None",
                "binary_fairness_verdict": "Not assigned",
            }
        )

    return pd.DataFrame(rows)


def append_check(
    checks: list[dict[str, str]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    """Append one validation record."""

    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": str(observed),
            "requirement": requirement,
            "details": details,
        }
    )


def validate_outputs(
    populations: dict[str, pd.DataFrame],
    loaded_columns: set[str],
    thresholds: dict[str, float],
    population_summary: pd.DataFrame,
    salary_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    risk_salary: pd.DataFrame,
    department: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate scope, formulas, reconciliation, and governance."""

    checks: list[dict[str, str]] = []
    tolerance = float(config["validation"]["numeric_tolerance"])
    expected_count = int(config["validation"]["expected_selected_count"])
    selected_counts = {
        period: {key: int(frame[f"selected_{key}"].sum()) for key in POLICY_ORDER}
        for period, frame in populations.items()
    }

    append_check(
        checks,
        "Outcome columns remain unread",
        not bool(loaded_columns & FORBIDDEN_OUTCOME_COLUMNS),
        sorted(loaded_columns & FORBIDDEN_OUTCOME_COLUMNS),
        "No target, termination, or outcome columns",
        "Allocation equity does not need final-test outcomes.",
    )
    append_check(
        checks,
        "All configured populations are complete",
        len(populations) == int(config["validation"]["expected_populations"]),
        {period: len(frame) for period, frame in populations.items()},
        f"{config['validation']['expected_populations']} populations",
        "Validation, final-test, and current allocations are audited.",
    )
    append_check(
        checks,
        "Every sensitivity policy respects capacity",
        all(
            count == expected_count
            for period in selected_counts.values()
            for count in period.values()
        ),
        selected_counts,
        f"{expected_count} selected per policy and population",
        "No sensitivity comparison receives additional capacity.",
    )
    current = populations["2026 current"]
    committed_difference = int(
        (
            current["selected_frozen_expected_value"].astype(bool)
            != current["committed_selection"].astype(bool)
        ).sum()
    )
    append_check(
        checks,
        "Current frozen selections reproduce Checkpoint 46",
        committed_difference == 0,
        committed_difference,
        "0 differing employee flags",
        "The audit measures the deployed plan without changing it.",
    )

    expected = pd.read_csv(
        POLICY_COMPARISON_PATH,
        usecols=[
            "evaluation_period",
            "policy_key",
            "selected_count",
            "average_selected_salary",
        ],
    )
    expected = expected[expected["policy_key"].eq("budget_expected_value")].set_index(
        "evaluation_period"
    )
    reconciliation: dict[str, float] = {}
    for period in ["2024 validation", "2025 final test"]:
        observed = comparison[
            comparison["period"].eq(period)
            & comparison["policy_key"].eq("frozen_expected_value")
        ].iloc[0]
        reconciliation[period] = abs(
            float(observed["selected_mean_salary_usd"])
            - float(expected.loc[period, "average_selected_salary"])
        )
    append_check(
        checks,
        "Selected salary reproduces Checkpoint 46",
        max(reconciliation.values()) <= tolerance,
        reconciliation,
        f"Every difference <= {tolerance}",
        "The post-policy audit starts from the exact tested allocation.",
    )
    append_check(
        checks,
        "Salary sensitivity thresholds use validation only",
        (
            abs(thresholds["validation_salary_cap_usd"] - 111000.0) <= tolerance
            and abs(thresholds["validation_constant_replacement_cost_usd"] - 86400.0)
            <= tolerance
        ),
        thresholds,
        "$111,000 cap and $86,400 constant cost",
        "No final-test or current salary is used to define alternatives.",
    )
    missing_salary_groups = sum(
        int(
            frame[
                [
                    "absolute_salary_band",
                    "salary_quintile",
                    "within_job_level_salary_quintile",
                    "risk_quintile",
                ]
            ]
            .isna()
            .sum()
            .sum()
        )
        for frame in populations.values()
    )
    append_check(
        checks,
        "Every employee receives salary and risk groups",
        missing_salary_groups == 0,
        missing_salary_groups,
        "0 missing group assignments",
        "Pay-band and within-risk comparisons cover each eligible employee.",
    )
    output_tables = {
        "population_summary": population_summary,
        "salary_metrics": salary_metrics,
        "comparison": comparison,
        "risk_salary": risk_salary,
        "department": department,
    }
    aggregate_identifiers = {
        name: sorted(set(table.columns) & {"employee_id", "actual_attrition"})
        for name, table in output_tables.items()
    }
    append_check(
        checks,
        "Saved audit outputs are aggregate only",
        not any(aggregate_identifiers.values()),
        aggregate_identifiers,
        "No employee IDs or outcomes",
        "The artifact answers the policy question without a review list.",
    )
    append_check(
        checks,
        "Salary-driven allocation is explicitly quantified",
        population_summary["review_flag"].all(),
        population_summary[["period", "review_flag", "review_triggers"]].to_dict(
            "records"
        ),
        "Descriptive review result for every period",
        "A review flag documents a mechanism; it is not a legal verdict.",
    )
    governance = config["governance"]
    governance_observed = {
        "no_policy_change": governance["do_not_modify_frozen_policy"],
        "sensitivity_only": governance["sensitivity_policies_are_not_selected"],
        "human_review": governance["human_review_required"],
        "automatic_action": governance["automatic_employment_action_permitted"],
    }
    append_check(
        checks,
        "Governance keeps the tested policy frozen",
        (
            governance_observed["no_policy_change"]
            and governance_observed["sensitivity_only"]
            and governance_observed["human_review"]
            and not governance_observed["automatic_action"]
        ),
        governance_observed,
        "Frozen, sensitivity-only, human-review allocation",
        "A replacement policy requires new holdout or prospective evidence.",
    )
    expected_figures = set(config["figures"]["expected_files"])
    observed_figures = {path.name for path in FIGURE_DIR.glob("*.png")}
    append_check(
        checks,
        "All allocation-equity figures were generated",
        expected_figures == observed_figures,
        sorted(observed_figures),
        sorted(expected_figures),
        "The audit has reproducible reviewer-facing visual evidence.",
    )
    finite_columns = [
        "population_mean_salary_usd",
        "selected_mean_salary_usd",
        "selected_to_population_mean_salary_ratio",
    ]
    append_check(
        checks,
        "Headline salary metrics are finite",
        np.isfinite(population_summary[finite_columns]).all().all(),
        population_summary[finite_columns].min().to_dict(),
        "All finite",
        "Salary comparisons must not contain missing or infinite values.",
    )

    return pd.DataFrame(checks)


def save_figure(
    figure: plt.Figure,
    filename: str,
    config: dict[str, Any],
) -> None:
    """Save and close one configured figure."""

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / filename,
        dpi=int(config["figures"]["dpi"]),
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_salary_comparison(
    comparison: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Compare selected mean salary across policies and populations."""

    pivot = comparison.pivot(
        index="period",
        columns="policy",
        values="selected_mean_salary_usd",
    ).loc[PERIOD_ORDER]
    figure, axis = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Selected Mean Salary by Allocation Policy")
    axis.set_xlabel("")
    axis.set_ylabel("Mean selected salary (USD)")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Policy", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, "policy_selected_salary_comparison.png", config)


def plot_current_quintiles(
    salary_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Plot current selection rates across population salary quintiles."""

    current = salary_metrics[
        salary_metrics["period"].eq("2026 current")
        & salary_metrics["band_type"].eq("population_salary_quintile")
    ]
    pivot = current.pivot(
        index="salary_band",
        columns="policy",
        values="selection_rate",
    )
    ordered = quantile_labels(5, "pay")
    pivot = pivot.reindex(ordered)
    figure, axis = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Current Selection Rate by Salary Quintile")
    axis.set_xlabel("Salary quintile")
    axis.set_ylabel("Selection rate")
    axis.tick_params(axis="x", rotation=15)
    axis.legend(title="Policy", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, "current_pay_quintile_selection.png", config)


def plot_tradeoff(
    comparison: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Plot current selected salary against original economic value."""

    current = comparison[comparison["period"].eq("2026 current")]
    figure, axis = plt.subplots(figsize=(10, 6))
    for row in current.itertuples(index=False):
        axis.scatter(
            row.selected_mean_salary_usd,
            row.predicted_net_value_under_original_economics_usd,
            s=70,
        )
        axis.annotate(
            row.policy,
            (
                row.selected_mean_salary_usd,
                row.predicted_net_value_under_original_economics_usd,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title("Current Policy Value and Salary-Composition Tradeoff")
    axis.set_xlabel("Mean selected salary (USD)")
    axis.set_ylabel("Predicted net value under original economics (USD)")
    axis.grid(alpha=0.25)
    save_figure(figure, "policy_value_equity_tradeoff.png", config)


def print_table(title: str, frame: pd.DataFrame) -> None:
    """Print one readable result table."""

    print("")
    print(title)
    print(frame.to_string(index=False))


def main() -> None:
    """Run the complete post-policy allocation-equity audit."""

    print("")
    print("=" * 50)
    print("Audit retention policy allocation equity")
    print("=" * 50)

    config = load_yaml(CONFIG_PATH)
    required_paths = [
        MULTI_SNAPSHOT_PATH,
        CALIBRATION_PATH,
        FINAL_TEST_PATH,
        CURRENT_PATH,
        POLICY_COMPARISON_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 42/46 inputs:\n" + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old_figure in FIGURE_DIR.glob("*.png"):
        old_figure.unlink()

    raw_populations, loaded_columns = load_populations(config)
    thresholds = frozen_thresholds(
        raw_populations["2024 validation"],
        config,
    )
    populations = {
        period: add_salary_groups(
            apply_policies(frame, config, thresholds),
            config,
        )
        for period, frame in raw_populations.items()
    }

    population_summary = build_population_summary(populations, config)
    salary_metrics = build_salary_band_metrics(populations, config)
    comparison = build_policy_comparison(populations, config)
    risk_salary = build_risk_salary_intersections(populations)
    department = build_department_summary(populations)
    threshold_table = pd.DataFrame([thresholds])

    plot_salary_comparison(comparison, config)
    plot_current_quintiles(salary_metrics, config)
    plot_tradeoff(comparison, config)

    validation = validate_outputs(
        populations,
        loaded_columns,
        thresholds,
        population_summary,
        salary_metrics,
        comparison,
        risk_salary,
        department,
        config,
    )

    population_summary.to_csv(POPULATION_SUMMARY_PATH, index=False)
    salary_metrics.to_csv(SALARY_BAND_PATH, index=False)
    comparison.to_csv(POLICY_COMPARISON_OUTPUT_PATH, index=False)
    risk_salary.to_csv(RISK_SALARY_PATH, index=False)
    department.to_csv(DEPARTMENT_PATH, index=False)
    threshold_table.to_csv(THRESHOLD_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)

    print_table(
        "SALARY-ALLOCATION SUMMARY",
        population_summary[
            [
                "period",
                "population_rows",
                "selected_count",
                "population_mean_salary_usd",
                "selected_mean_salary_usd",
                "selected_to_population_mean_salary_ratio",
                "lowest_salary_quintile_selection_rate",
                "highest_salary_quintile_selection_rate",
                "highest_to_lowest_selection_rate_ratio",
                "overlap_with_probability_top_700",
                "review_flag",
            ]
        ],
    )
    print_table(
        "CURRENT POLICY SENSITIVITY",
        comparison[comparison["period"].eq("2026 current")][
            [
                "policy",
                "selected_count",
                "selected_mean_salary_usd",
                "selected_mean_probability",
                "predicted_net_value_under_original_economics_usd",
                "overlap_with_frozen_policy",
            ]
        ],
    )
    manufacturing = department[
        department["period"].eq("2026 current")
        & department["department_name"].eq("Manufacturing")
    ]
    print_table("CURRENT MANUFACTURING PAY AUDIT", manufacturing)
    print_table("ALLOCATION-EQUITY VALIDATION", validation)

    failed = validation[validation["status"].ne("PASS")]
    if not failed.empty:
        raise RuntimeError(
            "Allocation-equity validation failed:\n"
            + json.dumps(failed.to_dict("records"), indent=2)
        )

    print("")
    print(f"Saved allocation-equity outputs to: {OUTPUT_DIR}")
    print("Outcome columns accessed: 0; frozen-policy changes made: 0")
    print("")
    print("RETENTION POLICY ALLOCATION-EQUITY AUDIT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
