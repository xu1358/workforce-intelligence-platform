"""Audit temporal attrition base-rate drift and calibration transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "temporal_prior_shift.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the committed prior-shift audit contract."""

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Temporal prior-shift configuration must be a mapping.")
    return config


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    """Fail clearly when an upstream aggregate contract has changed."""

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_inputs(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the temporal panel and existing aggregate calibration evidence."""

    paths = config["inputs"]
    panel = pd.read_csv(project_root / paths["temporal_panel"])
    calibration = pd.read_csv(
        project_root / paths["validation_calibration_metrics"]
    )
    final_metrics = pd.read_csv(project_root / paths["final_test_metrics"])

    require_columns(
        panel,
        {
            "employee_id",
            "snapshot_sequence",
            "snapshot_date",
            "prediction_end_date",
            "organizational_level",
            config["columns"]["target"],
        },
        "Temporal panel",
    )
    require_columns(
        calibration,
        {
            "method",
            "rows",
            "positive_cases",
            "observed_attrition_rate",
            "mean_predicted_probability",
            "mean_calibration_gap",
        },
        "Validation calibration metrics",
    )
    require_columns(
        final_metrics,
        {
            "evaluation_period",
            "rows",
            "positive_cases",
            "positive_rate",
            "mean_predicted_probability",
            "mean_calibration_gap",
        },
        "Final-test metrics",
    )
    return panel, calibration, final_metrics


def summarize_cohort(
    frame: pd.DataFrame,
    cohort: str,
    target: str,
) -> pd.DataFrame:
    """Aggregate one row per temporal period for a named population."""

    summary = (
        frame.groupby(
            [
                "snapshot_sequence",
                "snapshot_date",
                "prediction_end_date",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            rows=("employee_id", "size"),
            positive_cases=(target, "sum"),
            positive_rate=(target, "mean"),
        )
        .sort_values("snapshot_sequence")
        .reset_index(drop=True)
    )
    summary["positive_cases"] = summary["positive_cases"].astype(int)
    summary.insert(0, "cohort", cohort)
    summary["absolute_change_from_previous"] = summary[
        "positive_rate"
    ].diff()
    summary["relative_change_from_previous"] = summary[
        "positive_rate"
    ].pct_change()
    first_rate = float(summary.loc[0, "positive_rate"])
    summary["absolute_change_from_first"] = (
        summary["positive_rate"] - first_rate
    )
    summary["relative_change_from_first"] = (
        summary["positive_rate"] / first_rate - 1.0
    )
    return summary


def build_base_rate_summary(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build full-population and model-eligible temporal rate summaries."""

    target = config["columns"]["target"]
    if panel[target].isna().any():
        raise ValueError("Historical temporal outcomes must be complete.")
    values = sorted(panel[target].unique().tolist())
    if values != [0, 1]:
        raise ValueError(f"Temporal outcome must be binary; observed {values}.")

    full = summarize_cohort(panel, "Full historical population", target)
    eligible_levels = config["columns"]["model_eligible_levels"]
    eligible_panel = panel[
        panel["organizational_level"].isin(eligible_levels)
    ].copy()
    eligible = summarize_cohort(
        eligible_panel,
        "Model-eligible population",
        target,
    )

    periods = {
        int(item["snapshot_sequence"]): item["label"]
        for item in config["periods"]
    }
    summary = pd.concat([full, eligible], ignore_index=True)
    summary.insert(
        2,
        "period",
        summary["snapshot_sequence"].map(periods),
    )
    if summary["period"].isna().any():
        raise ValueError("Every temporal period must have a configured label.")
    return summary


def selected_validation_row(
    calibration: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """Return the selected validation calibration row."""

    method = config["columns"]["selected_calibration_method"]
    rows = calibration[calibration["method"].eq(method)]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one validation row for {method}; found {len(rows)}."
        )
    return rows.iloc[0]


def build_calibration_summary(
    calibration: pd.DataFrame,
    final_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Place validation and final-test aggregate calibration side by side."""

    validation = selected_validation_row(calibration, config)
    if len(final_metrics) != 1:
        raise ValueError(
            "Expected exactly one once-only final-test metrics row."
        )
    final = final_metrics.iloc[0]

    records = [
        {
            "period": "2024 validation",
            "evaluation_role": "Calibration selection",
            "rows": int(validation["rows"]),
            "positive_cases": int(validation["positive_cases"]),
            "observed_attrition_rate": float(
                validation["observed_attrition_rate"]
            ),
            "mean_predicted_probability": float(
                validation["mean_predicted_probability"]
            ),
            "calibration_gap": float(validation["mean_calibration_gap"]),
            "final_test_recalibration_permitted": False,
        },
        {
            "period": "2025 final test",
            "evaluation_role": "Once-only frozen evaluation",
            "rows": int(final["rows"]),
            "positive_cases": int(final["positive_cases"]),
            "observed_attrition_rate": float(final["positive_rate"]),
            "mean_predicted_probability": float(
                final["mean_predicted_probability"]
            ),
            "calibration_gap": float(final["mean_calibration_gap"]),
            "final_test_recalibration_permitted": False,
        },
    ]
    result = pd.DataFrame(records)
    result["absolute_calibration_gap"] = result["calibration_gap"].abs()
    return result


def prior_odds_multiplier(source_rate: float, target_rate: float) -> float:
    """Return the scalar odds change implied by two observed prevalences."""

    if not 0.0 < source_rate < 1.0 or not 0.0 < target_rate < 1.0:
        raise ValueError("Rates must be strictly between zero and one.")
    source_odds = source_rate / (1.0 - source_rate)
    target_odds = target_rate / (1.0 - target_rate)
    return target_odds / source_odds


def build_shift_decomposition(
    calibration_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Decompose validation-to-test mean calibration transport."""

    validation = calibration_summary.iloc[0]
    final = calibration_summary.iloc[1]
    observed_change = float(
        final["observed_attrition_rate"]
        - validation["observed_attrition_rate"]
    )
    predicted_change = float(
        final["mean_predicted_probability"]
        - validation["mean_predicted_probability"]
    )
    gap_change = float(
        final["calibration_gap"] - validation["calibration_gap"]
    )
    untracked_change = observed_change - predicted_change

    return pd.DataFrame(
        [
            {
                "source_period": validation["period"],
                "target_period": final["period"],
                "observed_rate_change": observed_change,
                "observed_relative_change": (
                    float(final["observed_attrition_rate"])
                    / float(validation["observed_attrition_rate"])
                    - 1.0
                ),
                "predicted_mean_change": predicted_change,
                "predicted_relative_change": (
                    float(final["mean_predicted_probability"])
                    / float(validation["mean_predicted_probability"])
                    - 1.0
                ),
                "untracked_prevalence_change": untracked_change,
                "calibration_gap_change": gap_change,
                "prior_odds_multiplier": prior_odds_multiplier(
                    float(validation["observed_attrition_rate"]),
                    float(final["observed_attrition_rate"]),
                ),
                "diagnostic_label": config["terminology"]["preferred"],
                "pure_label_shift_proven": False,
                "final_test_adjustment_applied": False,
            }
        ]
    )


def add_check(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> None:
    """Append one human-readable validation check."""

    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def values_close(
    observed: list[float],
    expected: list[float],
    tolerance: float,
) -> bool:
    """Compare same-length numeric sequences."""

    return len(observed) == len(expected) and bool(
        np.allclose(observed, expected, atol=tolerance, rtol=0.0)
    )


def build_checks(
    base_rates: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    decomposition: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate the drift evidence, terminology, and governance boundary."""

    checks: list[dict[str, Any]] = []
    expected = config["expected"]
    tolerance = float(config["thresholds"]["reconciliation_tolerance"])
    aggregate_tolerance = float(
        config["thresholds"]["serialized_aggregate_tolerance"]
    )

    full = base_rates[
        base_rates["cohort"].eq("Full historical population")
    ].sort_values("snapshot_sequence")
    eligible = base_rates[
        base_rates["cohort"].eq("Model-eligible population")
    ].sort_values("snapshot_sequence")

    add_check(
        checks,
        "Full historical period counts reconcile",
        full["rows"].tolist() == expected["full_population_rows"]
        and full["positive_cases"].tolist()
        == expected["full_population_positive_cases"],
        {
            "rows": full["rows"].tolist(),
            "positive_cases": full["positive_cases"].tolist(),
        },
        {
            "rows": expected["full_population_rows"],
            "positive_cases": expected["full_population_positive_cases"],
        },
        "The 9.62%, 10.42%, and 11.91% rates have explicit denominators.",
    )
    add_check(
        checks,
        "Full historical period rates reconcile",
        values_close(
            full["positive_rate"].tolist(),
            expected["full_population_rates"],
            tolerance,
        ),
        full["positive_rate"].tolist(),
        expected["full_population_rates"],
        "The previously undocumented full-population drift is reproducible.",
    )
    add_check(
        checks,
        "Model-eligible period counts reconcile",
        eligible["rows"].tolist() == expected["model_eligible_rows"]
        and eligible["positive_cases"].tolist()
        == expected["model_eligible_positive_cases"],
        {
            "rows": eligible["rows"].tolist(),
            "positive_cases": eligible["positive_cases"].tolist(),
        },
        {
            "rows": expected["model_eligible_rows"],
            "positive_cases": expected["model_eligible_positive_cases"],
        },
        "Calibration transport is evaluated on the actual model population.",
    )
    add_check(
        checks,
        "Model-eligible period rates reconcile",
        values_close(
            eligible["positive_rate"].tolist(),
            expected["model_eligible_rates"],
            tolerance,
        ),
        eligible["positive_rate"].tolist(),
        expected["model_eligible_rates"],
        "Train, validation, and final-test prevalence are reported separately.",
    )

    full_relative = float(full.iloc[-1]["relative_change_from_first"])
    add_check(
        checks,
        "First-to-final full-population drift is material",
        np.isclose(
            full_relative,
            expected["full_population_first_to_final_relative_change"],
            atol=tolerance,
            rtol=0.0,
        )
        and full_relative
        >= float(
            config["thresholds"]["material_relative_base_rate_change"]
        ),
        full_relative,
        expected["full_population_first_to_final_relative_change"],
        "Outcome prevalence rises about 23.74% relative across snapshots.",
    )

    validation = calibration_summary.iloc[0]
    final = calibration_summary.iloc[1]
    add_check(
        checks,
        "Selected validation calibration aggregate reconciles",
        np.isclose(
            validation["mean_predicted_probability"],
            expected["validation_mean_probability"],
            atol=aggregate_tolerance,
            rtol=0.0,
        )
        and np.isclose(
            validation["calibration_gap"],
            expected["validation_calibration_gap"],
            atol=aggregate_tolerance,
            rtol=0.0,
        ),
        {
            "mean_probability": float(
                validation["mean_predicted_probability"]
            ),
            "gap": float(validation["calibration_gap"]),
        },
        {
            "mean_probability": expected["validation_mean_probability"],
            "gap": expected["validation_calibration_gap"],
        },
        "Sigmoid calibration was selected on the 2024 validation period.",
    )
    add_check(
        checks,
        "Once-only final-test aggregate reconciles",
        np.isclose(
            final["mean_predicted_probability"],
            expected["final_test_mean_probability"],
            atol=aggregate_tolerance,
            rtol=0.0,
        )
        and np.isclose(
            final["calibration_gap"],
            expected["final_test_calibration_gap"],
            atol=aggregate_tolerance,
            rtol=0.0,
        ),
        {
            "mean_probability": float(final["mean_predicted_probability"]),
            "gap": float(final["calibration_gap"]),
        },
        {
            "mean_probability": expected["final_test_mean_probability"],
            "gap": expected["final_test_calibration_gap"],
        },
        "The audit reuses the frozen aggregate result instead of rerunning it.",
    )

    shift = decomposition.iloc[0]
    add_check(
        checks,
        "Validation-to-test prevalence shift is explicit",
        np.isclose(
            shift["observed_relative_change"],
            expected["validation_to_test_relative_change"],
            atol=tolerance,
            rtol=0.0,
        ),
        float(shift["observed_relative_change"]),
        expected["validation_to_test_relative_change"],
        "Model-eligible prevalence rises about 13.92% relative after calibration selection.",
    )
    add_check(
        checks,
        "Calibration-gap deterioration is arithmetically complete",
        np.isclose(
            shift["calibration_gap_change"],
            expected["calibration_gap_deterioration"],
            atol=aggregate_tolerance,
            rtol=0.0,
        )
        and np.isclose(
            shift["untracked_prevalence_change"],
            -shift["calibration_gap_change"],
            atol=tolerance,
            rtol=0.0,
        ),
        {
            "gap_change": float(shift["calibration_gap_change"]),
            "untracked_prevalence": float(
                shift["untracked_prevalence_change"]
            ),
        },
        expected["calibration_gap_deterioration"],
        "Observed prevalence rose 0.476 percentage points more than the predicted mean.",
    )

    terminology = config["terminology"]
    add_check(
        checks,
        "Prior-shift terminology remains qualified",
        "does not prove" in terminology["qualified_statement"]
        and terminology["forbidden_claim"]
        not in terminology["qualified_statement"],
        terminology["qualified_statement"],
        "Consistent with prior-probability shift; pure label shift not proven",
        "Prevalence drift alone cannot establish stable class-conditional features.",
    )

    output_columns = set(base_rates.columns)
    output_columns.update(calibration_summary.columns)
    output_columns.update(decomposition.columns)
    prohibited = sorted(
        output_columns
        & {
            "employee_id",
            "actual_attrition",
            "attrition_probability",
            "selected_for_review",
        }
    )
    add_check(
        checks,
        "Saved prior-shift evidence is aggregate only",
        not prohibited,
        prohibited,
        [],
        "No employee-level prediction, outcome, or review flag is exported.",
    )

    governance = config["governance"]
    required_false = [
        "uses_final_test_prediction_rows",
        "recalibrates_on_final_test",
        "changes_selected_model",
        "changes_frozen_policy",
        "changes_dashboard_probabilities",
    ]
    governance_passed = (
        governance["post_evaluation_diagnostic"]
        and governance["final_test_evaluated_once"]
        and governance["aggregates_temporal_outcomes_for_prevalence"]
        and all(not governance[key] for key in required_false)
    )
    add_check(
        checks,
        "Audit governance preserves the frozen decision system",
        governance_passed,
        governance,
        "Post-evaluation, aggregate, diagnostic only",
        "The finding becomes a monitoring limitation, not a post-test model adjustment.",
    )

    return pd.DataFrame(checks)


def plot_base_rates(
    base_rates: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot temporal attrition prevalence for both documented populations."""

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    colors = {
        "Full historical population": "#4472C4",
        "Model-eligible population": "#ED7D31",
    }
    for cohort, group in base_rates.groupby("cohort", sort=False):
        ordered = group.sort_values("snapshot_sequence")
        ax.plot(
            ordered["period"],
            ordered["positive_rate"] * 100,
            marker="o",
            linewidth=2.2,
            label=cohort,
            color=colors[cohort],
        )
        for _, row in ordered.iterrows():
            ax.annotate(
                f"{row['positive_rate']:.2%}",
                (row["period"], row["positive_rate"] * 100),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    ax.set_title("Temporal attrition base-rate drift")
    ax.set_ylabel("Observed 12-month attrition rate (%)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_calibration_transport(
    calibration_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed prevalence beside mean predicted probability."""

    labels = calibration_summary["period"].tolist()
    positions = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    observed = calibration_summary["observed_attrition_rate"] * 100
    predicted = calibration_summary["mean_predicted_probability"] * 100
    ax.bar(
        positions - width / 2,
        observed,
        width,
        label="Observed attrition rate",
        color="#4472C4",
    )
    ax.bar(
        positions + width / 2,
        predicted,
        width,
        label="Mean predicted probability",
        color="#A5A5A5",
    )
    for x_value, value in zip(positions - width / 2, observed):
        ax.text(x_value, value + 0.15, f"{value:.2f}%", ha="center")
    for x_value, value in zip(positions + width / 2, predicted):
        ax.text(x_value, value + 0.15, f"{value:.2f}%", ha="center")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Calibration transport across time")
    ax.set_ylim(0, max(observed.max(), predicted.max()) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def display_table(frame: pd.DataFrame) -> str:
    """Return a readable console table without optional dependencies."""

    return frame.to_string(index=False)


def main() -> None:
    """Run and validate the temporal prior-shift audit."""

    config = load_config()
    panel, calibration, final_metrics = load_inputs(PROJECT_ROOT, config)
    base_rates = build_base_rate_summary(panel, config)
    calibration_summary = build_calibration_summary(
        calibration,
        final_metrics,
        config,
    )
    decomposition = build_shift_decomposition(calibration_summary, config)

    output_config = config["outputs"]
    output_directory = PROJECT_ROOT / output_config["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    base_rates.to_csv(
        output_directory / output_config["files"]["base_rate_summary"],
        index=False,
    )
    calibration_summary.to_csv(
        output_directory / output_config["files"]["calibration_summary"],
        index=False,
    )
    decomposition.to_csv(
        output_directory / output_config["files"]["decomposition"],
        index=False,
    )
    plot_base_rates(
        base_rates,
        output_directory / output_config["figures"][0],
    )
    plot_calibration_transport(
        calibration_summary,
        output_directory / output_config["figures"][1],
    )

    checks = build_checks(
        base_rates,
        calibration_summary,
        decomposition,
        config,
    )
    checks.to_csv(
        output_directory / output_config["files"]["validation"],
        index=False,
    )
    failures = checks[checks["status"].ne("PASS")]

    full = base_rates[
        base_rates["cohort"].eq("Full historical population")
    ][
        [
            "period",
            "rows",
            "positive_cases",
            "positive_rate",
            "relative_change_from_first",
        ]
    ]
    print("\nTEMPORAL BASE-RATE DRIFT")
    print(display_table(full))
    print("\nCALIBRATION TRANSPORT")
    print(display_table(calibration_summary))
    print("\nPRIOR-SHIFT DECOMPOSITION")
    print(display_table(decomposition))
    print("\nTEMPORAL PRIOR-SHIFT VALIDATION")
    print(display_table(checks))

    if not failures.empty:
        raise ValueError(
            "Temporal prior-shift validation failed:\n"
            + display_table(failures)
        )

    print(f"\nSaved prior-shift outputs to: {output_directory}")
    print(
        "Final-test recalibration applied: 0; model, policy, and dashboard "
        "changes made: 0"
    )
    print("\nTEMPORAL PRIOR-SHIFT AUDIT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
