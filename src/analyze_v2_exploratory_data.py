"""Run reproducible exploratory analysis for the Version 2 temporal data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


plt.switch_backend("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "v2_exploratory_analysis.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the Version 2 EDA contract."""

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Version 2 EDA configuration must be a mapping.")
    return config


def load_inputs(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the authoritative historical panel and current population."""

    panel = pd.read_csv(project_root / config["inputs"]["temporal_panel"])
    current = pd.read_csv(project_root / config["inputs"]["current_population"])
    required = {
        config["columns"]["employee_id"],
        config["columns"]["snapshot_sequence"],
        config["columns"]["target"],
        config["columns"]["review_flag"],
        *config["columns"]["review_fields"],
        *config["columns"]["group_dimensions"],
    }
    missing_panel = sorted(required - set(panel.columns))
    missing_current = sorted(required - set(current.columns))
    if missing_panel or missing_current:
        raise ValueError(
            "Version 2 EDA inputs are incomplete: "
            f"panel={missing_panel}, current={missing_current}"
        )
    return panel, current


def build_dataset_profile(
    panel: pd.DataFrame,
    current: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize the two authoritative Version 2 analytical populations."""

    employee = config["columns"]["employee_id"]
    repeated = int(panel.groupby(employee).size().gt(1).sum())
    records = [
        {
            "population": "Historical temporal panel",
            "rows": len(panel),
            "unique_employees": panel[employee].nunique(),
            "columns": len(panel.columns),
            "snapshot_count": panel[
                config["columns"]["snapshot_sequence"]
            ].nunique(),
            "employees_repeated_across_snapshots": repeated,
            "outcome_role": "Training outcomes only for EDA",
        },
        {
            "population": "Current active scoring population",
            "rows": len(current),
            "unique_employees": current[employee].nunique(),
            "columns": len(current.columns),
            "snapshot_count": 1,
            "employees_repeated_across_snapshots": 0,
            "outcome_role": "Future outcomes unknown",
        },
    ]
    return pd.DataFrame(records)


def build_period_structure(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Describe every snapshot without inspecting reserved-period outcomes."""

    sequence = config["columns"]["snapshot_sequence"]
    employee = config["columns"]["employee_id"]
    target = config["columns"]["target"]
    review_flag = config["columns"]["review_flag"]
    records: list[dict[str, Any]] = []

    for period in config["periods"]:
        frame = panel[panel[sequence].eq(period["snapshot_sequence"])]
        target_rate = (
            float(frame[target].mean())
            if period["role"] == "training_eda"
            else np.nan
        )
        records.append(
            {
                "snapshot_sequence": period["snapshot_sequence"],
                "period": period["label"],
                "eda_role": period["role"],
                "rows": len(frame),
                "unique_employees": frame[employee].nunique(),
                "missing_review_rows": int(frame[review_flag].sum()),
                "missing_review_rate": float(frame[review_flag].mean()),
                "attrition_rate_used_for_eda": target_rate,
            }
        )
    return pd.DataFrame(records)


def build_missingness_summary(
    panel: pd.DataFrame,
    current: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Quantify missing fields without replacing missingness with V1 results."""

    target = config["columns"]["target"]
    records: list[dict[str, Any]] = []
    for column in panel.columns:
        historical_missing = int(panel[column].isna().sum())
        current_missing = int(current[column].isna().sum())
        if historical_missing == 0 and current_missing == 0:
            continue
        records.append(
            {
                "column": column,
                "historical_missing_rows": historical_missing,
                "historical_missing_rate": historical_missing / len(panel),
                "current_missing_rows": current_missing,
                "current_missing_rate": current_missing / len(current),
                "interpretation": (
                    "Future outcome intentionally unknown"
                    if column == target
                    else "Availability represented explicitly"
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["historical_missing_rate", "current_missing_rate"],
        ascending=False,
    )


def build_training_review_missingness(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Measure the review-history signal using training outcomes only."""

    sequence = config["columns"]["snapshot_sequence"]
    target = config["columns"]["target"]
    review_flag = config["columns"]["review_flag"]
    training_sequence = next(
        period["snapshot_sequence"]
        for period in config["periods"]
        if period["role"] == "training_eda"
    )
    training = panel[panel[sequence].eq(training_sequence)]
    summary = (
        training.groupby(review_flag, dropna=False)[target]
        .agg(rows="size", positive_cases="sum", attrition_rate="mean")
        .reset_index()
    )
    summary["review_history"] = summary[review_flag].map(
        {0: "Prior review present", 1: "No prior review"}
    )
    summary["analysis_role"] = "Training-only descriptive EDA"
    return summary[
        [
            review_flag,
            "review_history",
            "rows",
            "positive_cases",
            "attrition_rate",
            "analysis_role",
        ]
    ]


def build_training_group_rates(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create training-only descriptive group summaries."""

    sequence = config["columns"]["snapshot_sequence"]
    target = config["columns"]["target"]
    training_sequence = next(
        period["snapshot_sequence"]
        for period in config["periods"]
        if period["role"] == "training_eda"
    )
    training = panel[panel[sequence].eq(training_sequence)]
    minimum_rows = int(config["thresholds"]["minimum_group_rows"])
    summaries: list[pd.DataFrame] = []

    for dimension in config["columns"]["group_dimensions"]:
        grouped = (
            training.groupby(dimension, dropna=False)[target]
            .agg(rows="size", positive_cases="sum", attrition_rate="mean")
            .reset_index()
            .rename(columns={dimension: "group"})
        )
        grouped = grouped[grouped["rows"].ge(minimum_rows)].copy()
        grouped.insert(0, "dimension", dimension)
        grouped["population_share"] = grouped["rows"] / len(training)
        grouped["analysis_role"] = "Training-only descriptive EDA"
        summaries.append(grouped)

    return pd.concat(summaries, ignore_index=True)


def build_v1_v2_missingness_comparison(
    review_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Show why the Version 1 missingness signal cannot be reused."""

    baseline = config["historical_v1_baseline"]
    review_flag = config["columns"]["review_flag"]
    missing = review_summary[review_summary[review_flag].eq(1)].iloc[0]
    present = review_summary[review_summary[review_flag].eq(0)].iloc[0]
    records = [
        {
            "dataset": baseline["label"],
            "rows": baseline["rows"],
            "missing_review_rows": baseline["missing_review_rows"],
            "missing_review_rate": baseline["missing_review_rate"],
            "missing_review_attrition_rate": baseline[
                "missing_review_attrition_rate"
            ],
            "review_present_attrition_rate": baseline[
                "review_present_attrition_rate"
            ],
            "missing_minus_present_gap": baseline[
                "missing_review_attrition_rate"
            ]
            - baseline["review_present_attrition_rate"],
            "authoritative_for_v2": False,
        },
        {
            "dataset": "Version 2 training snapshot",
            "rows": int(missing["rows"] + present["rows"]),
            "missing_review_rows": int(missing["rows"]),
            "missing_review_rate": float(
                missing["rows"] / (missing["rows"] + present["rows"])
            ),
            "missing_review_attrition_rate": float(missing["attrition_rate"]),
            "review_present_attrition_rate": float(present["attrition_rate"]),
            "missing_minus_present_gap": float(
                missing["attrition_rate"] - present["attrition_rate"]
            ),
            "authoritative_for_v2": True,
        },
    ]
    return pd.DataFrame(records)


def add_check(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
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


def build_checks(
    panel: pd.DataFrame,
    current: pd.DataFrame,
    profile: pd.DataFrame,
    period_structure: pd.DataFrame,
    review_summary: pd.DataFrame,
    group_rates: pd.DataFrame,
    comparison: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate results, leakage boundaries, and aggregate-only outputs."""

    expected = config["expected"]
    tolerance = float(config["thresholds"]["reconciliation_tolerance"])
    target = config["columns"]["target"]
    review_flag = config["columns"]["review_flag"]
    review_fields = config["columns"]["review_fields"]
    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "Authoritative Version 2 populations reconcile",
        len(panel) == expected["historical_rows"]
        and panel[config["columns"]["employee_id"]].nunique()
        == expected["historical_unique_employees"]
        and len(current) == expected["current_rows"],
        {
            "historical_rows": len(panel),
            "historical_unique_employees": panel[
                config["columns"]["employee_id"]
            ].nunique(),
            "current_rows": len(current),
        },
        {
            "historical_rows": expected["historical_rows"],
            "historical_unique_employees": expected[
                "historical_unique_employees"
            ],
            "current_rows": expected["current_rows"],
        },
        "The EDA describes the same temporal populations used by Version 2.",
    )
    add_check(
        checks,
        "Snapshot row counts reconcile",
        period_structure["rows"].tolist() == expected["period_rows"],
        period_structure["rows"].tolist(),
        expected["period_rows"],
        "The temporal panel is not replaced by the 7,386-row V1 snapshot.",
    )
    missing_mask = panel[review_fields].isna().all(axis=1)
    current_missing_mask = current[review_fields].isna().all(axis=1)
    add_check(
        checks,
        "Review-missingness flag matches feature availability",
        bool(missing_mask.eq(panel[review_flag].astype(bool)).all())
        and bool(
            current_missing_mask.eq(current[review_flag].astype(bool)).all()
        ),
        {
            "historical_mismatches": int(
                missing_mask.ne(panel[review_flag].astype(bool)).sum()
            ),
            "current_mismatches": int(
                current_missing_mask.ne(
                    current[review_flag].astype(bool)
                ).sum()
            ),
        },
        "0 mismatches",
        "Missing review history is an explicit feature state, not an imputation accident.",
    )
    add_check(
        checks,
        "Version 2 review availability reconciles",
        int(panel[review_flag].sum())
        == expected["historical_missing_review_rows"]
        and np.isclose(
            panel[review_flag].mean(),
            expected["historical_missing_review_rate"],
            atol=tolerance,
            rtol=0.0,
        )
        and int(current[review_flag].sum())
        == expected["current_missing_review_rows"]
        and np.isclose(
            current[review_flag].mean(),
            expected["current_missing_review_rate"],
            atol=tolerance,
            rtol=0.0,
        ),
        {
            "historical": {
                "rows": int(panel[review_flag].sum()),
                "rate": float(panel[review_flag].mean()),
            },
            "current": {
                "rows": int(current[review_flag].sum()),
                "rate": float(current[review_flag].mean()),
            },
        },
        {
            "historical_rows": expected["historical_missing_review_rows"],
            "current_rows": expected["current_missing_review_rows"],
        },
        "The V1 24.44% missingness rate is not presented as a V2 result.",
    )
    add_check(
        checks,
        "Review availability is reported by snapshot",
        bool(
            np.allclose(
                period_structure["missing_review_rate"],
                expected["missing_review_rate_by_period"],
                atol=tolerance,
                rtol=0.0,
            )
        ),
        period_structure["missing_review_rate"].tolist(),
        expected["missing_review_rate_by_period"],
        "Review history accumulates as the synthetic workforce matures.",
    )
    missing = review_summary[review_summary[review_flag].eq(1)].iloc[0]
    present = review_summary[review_summary[review_flag].eq(0)].iloc[0]
    add_check(
        checks,
        "Training-only missingness signal reconciles",
        int(missing["rows"]) == expected["training_missing_review_rows"]
        and int(missing["positive_cases"])
        == expected["training_missing_review_positive_cases"]
        and int(present["rows"]) == expected["training_review_present_rows"]
        and int(present["positive_cases"])
        == expected["training_review_present_positive_cases"]
        and np.isclose(
            missing["attrition_rate"],
            expected["training_missing_review_attrition_rate"],
            atol=tolerance,
            rtol=0.0,
        )
        and np.isclose(
            present["attrition_rate"],
            expected["training_review_present_attrition_rate"],
            atol=tolerance,
            rtol=0.0,
        ),
        {
            "missing": missing[
                ["rows", "positive_cases", "attrition_rate"]
            ].to_dict(),
            "present": present[
                ["rows", "positive_cases", "attrition_rate"]
            ].to_dict(),
        },
        {
            "missing_rate": expected[
                "training_missing_review_attrition_rate"
            ],
            "present_rate": expected[
                "training_review_present_attrition_rate"
            ],
        },
        "The V2 training gap is 0.58 points, not the V1 8.71-point gap.",
    )
    add_check(
        checks,
        "Reserved outcomes remain outside exploratory analysis",
        period_structure.loc[
            period_structure["eda_role"].eq("structure_only"),
            "attrition_rate_used_for_eda",
        ]
        .isna()
        .all(),
        period_structure[
            ["snapshot_sequence", "eda_role", "attrition_rate_used_for_eda"]
        ].to_dict("records"),
        "Validation and final-test EDA rates are blank",
        "Only training outcomes support pre-model descriptive relationships.",
    )
    add_check(
        checks,
        "Current outcomes remain unknown",
        bool(current[target].isna().all()),
        int(current[target].notna().sum()),
        0,
        "Current scoring rows are not mislabeled with future outcomes.",
    )
    add_check(
        checks,
        "V1 and V2 missingness findings remain separated",
        comparison["authoritative_for_v2"].tolist() == [False, True]
        and np.isclose(
            comparison.iloc[1]["missing_minus_present_gap"],
            expected["training_missingness_gap"],
            atol=tolerance,
            rtol=0.0,
        ),
        comparison[
            ["dataset", "missing_minus_present_gap", "authoritative_for_v2"]
        ].to_dict("records"),
        "Only the Version 2 training row is authoritative",
        "The strong V1 missingness association is not carried into V2 claims.",
    )
    aggregate_outputs = [profile, period_structure, review_summary, group_rates]
    leaked_columns = sorted(
        {
            column
            for frame in aggregate_outputs
            for column in frame.columns
            if column == config["columns"]["employee_id"]
        }
    )
    add_check(
        checks,
        "Saved exploratory outputs are aggregate only",
        leaked_columns == [],
        leaked_columns,
        [],
        "No employee-level outcome or score file is produced.",
    )
    governance = config["governance"]
    add_check(
        checks,
        "EDA governance preserves the frozen decision system",
        governance["version_2_is_authoritative"]
        and governance["training_outcomes_only_for_eda"]
        and not governance["validation_outcomes_used_for_eda"]
        and not governance["final_test_outcomes_used_for_eda"]
        and not governance["causal_claims_permitted"]
        and not governance["changes_selected_model"]
        and not governance["changes_frozen_policy"]
        and not governance["changes_dashboard_probabilities"],
        governance,
        "Training-only, descriptive, no downstream changes",
        "Checkpoint 66 documents the data; it does not reopen model development.",
    )
    return pd.DataFrame(checks)


def plot_review_availability(
    period_structure: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot missing prior-review history across snapshots."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = period_structure["missing_review_rate"] * 100
    ax.plot(
        period_structure["period"],
        values,
        color="#2F6B8A",
        marker="o",
        linewidth=2.5,
    )
    for index, value in enumerate(values):
        ax.text(index, value + 2, f"{value:.1f}%", ha="center")
    ax.set_ylabel("Rows with no prior review (%)")
    ax.set_title("Version 2 review history becomes more available over time")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_departments(
    group_rates: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot training attrition rates by department."""

    frame = group_rates[
        group_rates["dimension"].eq("department_name")
    ].sort_values("attrition_rate")
    fig, ax = plt.subplots(figsize=(8, 5))
    values = frame["attrition_rate"] * 100
    ax.barh(frame["group"].astype(str), values, color="#4C956C")
    for index, value in enumerate(values):
        ax.text(value + 0.15, index, f"{value:.2f}%", va="center")
    ax.set_xlabel("Training-period attrition rate")
    ax.set_title("Descriptive training rates by department")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_missingness_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare the V1 and V2 missing-review associations."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions = np.arange(len(comparison))
    width = 0.34
    missing = comparison["missing_review_attrition_rate"] * 100
    present = comparison["review_present_attrition_rate"] * 100
    ax.bar(
        positions - width / 2,
        missing,
        width,
        label="No prior review",
        color="#C8553D",
    )
    ax.bar(
        positions + width / 2,
        present,
        width,
        label="Prior review present",
        color="#2F6B8A",
    )
    ax.set_xticks(positions, comparison["dataset"])
    ax.set_ylabel("Attrition rate (%)")
    ax.set_title("The Version 1 missingness signal does not transfer to V2")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def display_table(frame: pd.DataFrame) -> str:
    """Return a readable console table."""

    return frame.to_string(index=False)


def main() -> None:
    """Run the Version 2 exploratory analysis and validation."""

    config = load_config()
    panel, current = load_inputs(PROJECT_ROOT, config)
    profile = build_dataset_profile(panel, current, config)
    periods = build_period_structure(panel, config)
    missingness = build_missingness_summary(panel, current, config)
    review_summary = build_training_review_missingness(panel, config)
    group_rates = build_training_group_rates(panel, config)
    comparison = build_v1_v2_missingness_comparison(review_summary, config)
    checks = build_checks(
        panel,
        current,
        profile,
        periods,
        review_summary,
        group_rates,
        comparison,
        config,
    )

    output_config = config["outputs"]
    output_directory = PROJECT_ROOT / output_config["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "dataset_profile": profile,
        "period_structure": periods,
        "missingness": missingness,
        "review_missingness": review_summary,
        "group_rates": group_rates,
        "v1_v2_comparison": comparison,
        "validation": checks,
    }
    for key, frame in frames.items():
        frame.to_csv(
            output_directory / output_config["files"][key],
            index=False,
        )

    plot_review_availability(
        periods,
        output_directory / output_config["figures"][0],
    )
    plot_training_departments(
        group_rates,
        output_directory / output_config["figures"][1],
    )
    plot_missingness_comparison(
        comparison,
        output_directory / output_config["figures"][2],
    )

    print("\nVERSION 2 DATASET PROFILE")
    print(display_table(profile))
    print("\nVERSION 2 PERIOD STRUCTURE")
    print(display_table(periods))
    print("\nTRAINING REVIEW-MISSINGNESS RESULT")
    print(display_table(review_summary))
    print("\nV1–V2 MISSINGNESS COMPARISON")
    print(display_table(comparison))
    print("\nVERSION 2 EXPLORATORY VALIDATION")
    print(display_table(checks))

    failures = checks[checks["status"].ne("PASS")]
    if not failures.empty:
        raise ValueError(
            "Version 2 exploratory analysis validation failed:\n"
            + display_table(failures)
        )

    print(f"\nSaved Version 2 exploratory outputs to: {output_directory}")
    print(
        "Validation/final-test outcomes used for EDA: 0; "
        "model, policy, and dashboard changes made: 0"
    )
    print("\nVERSION 2 EXPLORATORY ANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
