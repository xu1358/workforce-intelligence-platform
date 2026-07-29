"""Analyze employee time-to-exit with censoring-aware survival methods.

Checkpoint 55 is an optional methodological extension. It builds one
employee-level survival cohort, estimates Kaplan-Meier retention curves, and
fits a penalized Cox proportional-hazards model using baseline-at-hire
features only.

Active employees are right-censored at the configured as-of date. Saved
artifacts are aggregate summaries: no names, employee identifiers,
employee-level survival estimates, review selections, or current scores are
exported. Results describe synthetic associations and do not change the
primary classifier, final test, frozen intervention policy, or dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any

MATPLOTLIB_CACHE_DIR = (
    Path(tempfile.gettempdir()) / "workforce_intelligence_matplotlib"
)
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import (
    multivariate_logrank_test,
    proportional_hazard_test,
)
from lifelines.utils import concordance_index
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "survival_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

CONFIG_PATH = PROJECT_ROOT / "config" / "survival_analysis.yaml"

POPULATION_SUMMARY_PATH = OUTPUT_DIR / "survival_population_summary.csv"
OVERALL_CURVE_PATH = OUTPUT_DIR / "survival_overall_curve.csv"
HORIZON_SUMMARY_PATH = OUTPUT_DIR / "survival_horizon_summary.csv"
GROUP_SUMMARY_PATH = OUTPUT_DIR / "survival_group_summary.csv"
LOGRANK_PATH = OUTPUT_DIR / "survival_logrank_tests.csv"
COX_HAZARD_PATH = OUTPUT_DIR / "cox_hazard_ratios.csv"
COX_CV_PATH = OUTPUT_DIR / "cox_cross_validation.csv"
COX_PH_PATH = OUTPUT_DIR / "cox_proportional_hazards_test.csv"
COX_MANIFEST_PATH = OUTPUT_DIR / "cox_feature_manifest.csv"
VALIDATION_PATH = OUTPUT_DIR / "survival_validation.csv"

HIRE_NOTE_PATTERN = re.compile(
    r"department_id=(?P<hire_department_id>\d+), "
    r"location_id=(?P<hire_location_id>\d+), "
    r"job_role_id=(?P<hire_job_role_id>\d+)"
)


def load_survival_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Load the committed survival-analysis contract."""

    if not path.exists():
        raise FileNotFoundError(f"Missing survival-analysis config: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")
    return loaded


def load_source_tables() -> dict[str, pd.DataFrame]:
    """Load only source tables needed for baseline survival analysis."""

    paths = {
        "employees": RAW_DIR / "employees.csv",
        "events": RAW_DIR / "employee_events.csv",
        "compensation": RAW_DIR / "compensation_history.csv",
        "departments": RAW_DIR / "departments.csv",
        "locations": RAW_DIR / "locations.csv",
        "roles": RAW_DIR / "job_roles.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing survival source tables: {missing}")

    tables = {name: pd.read_csv(path) for name, path in paths.items()}
    for column in ["hire_date", "termination_date"]:
        tables["employees"][column] = pd.to_datetime(
            tables["employees"][column],
            errors="coerce",
        )
    tables["compensation"]["effective_date"] = pd.to_datetime(
        tables["compensation"]["effective_date"],
        errors="coerce",
    )
    return tables


def parse_hire_event_notes(events: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct department, location, and job role at hire."""

    hires = events.loc[
        events["event_type"].eq("Hire"),
        ["employee_id", "notes"],
    ].copy()
    if not hires["employee_id"].is_unique:
        raise ValueError("Hire events are not unique by employee.")

    extracted = hires["notes"].str.extract(HIRE_NOTE_PATTERN)
    if extracted.isna().any(axis=None):
        raise ValueError("One or more Hire event notes cannot be parsed.")

    extracted = extracted.astype(int)
    parsed = pd.concat(
        [
            hires[["employee_id"]].reset_index(drop=True),
            extracted.reset_index(drop=True),
        ],
        axis=1,
    )
    return parsed


def build_survival_cohort(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the all-employee and model-eligible survival cohorts."""

    employees = tables["employees"].copy()
    as_of_date = pd.Timestamp(config["cohort"]["as_of_date"])
    hire_dimensions = parse_hire_event_notes(tables["events"])

    hire_compensation = tables["compensation"].loc[
        tables["compensation"]["change_reason"].eq("Hire"),
        ["employee_id", "effective_date", "base_salary"],
    ].copy()
    if not hire_compensation["employee_id"].is_unique:
        raise ValueError("Hire compensation is not unique by employee.")
    hire_compensation = hire_compensation.rename(
        columns={
            "effective_date": "hire_compensation_date",
            "base_salary": "hire_base_salary",
        }
    )

    departments = tables["departments"].rename(
        columns={"department_name": "hire_department_name"}
    )
    locations = tables["locations"].rename(
        columns={
            "city": "hire_city",
            "region": "hire_region",
            "location_type": "hire_location_type",
        }
    )
    roles = tables["roles"].rename(
        columns={
            "job_title": "hire_job_title",
            "job_family": "hire_job_family",
            "job_level": "hire_job_level",
        }
    )

    cohort = (
        employees.merge(
            hire_dimensions,
            on="employee_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            hire_compensation,
            on="employee_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            departments,
            left_on="hire_department_id",
            right_on="department_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            locations,
            left_on="hire_location_id",
            right_on="location_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            roles,
            left_on="hire_job_role_id",
            right_on="job_role_id",
            how="left",
            validate="many_to_one",
        )
    )

    cohort["event_observed"] = cohort["termination_date"].notna().astype(int)
    cohort["observation_end_date"] = cohort["termination_date"].fillna(as_of_date)
    tenure_days = (
        cohort["observation_end_date"] - cohort["hire_date"]
    ).dt.days.astype(float)
    if config["cohort"]["inclusive_tenure_days"]:
        tenure_days += 1.0
    cohort["duration_days"] = tenure_days
    cohort["duration_months"] = (
        cohort["duration_days"] / float(config["cohort"]["days_per_month"])
    )
    cohort["age_at_hire"] = (
        cohort["hire_date"].dt.year - cohort["birth_year"]
    ).astype(float)
    midpoint = (
        cohort["salary_band_min"].astype(float)
        + cohort["salary_band_max"].astype(float)
    ) / 2.0
    cohort["salary_position_at_hire"] = (
        cohort["hire_base_salary"].astype(float) / midpoint
    )
    cohort["hire_job_level"] = cohort["hire_job_level"].astype("Int64").astype(str)

    protected = set(config["cohort"]["protected_levels"])
    eligible = set(config["cohort"]["outcome_eligible_levels"])
    cohort["survival_model_eligible"] = cohort["organizational_level"].isin(
        eligible
    )
    cohort["protected_level"] = cohort["organizational_level"].isin(protected)

    model_cohort = cohort.loc[cohort["survival_model_eligible"]].copy()
    return cohort, model_cohort


def fit_kaplan_meier(
    cohort: pd.DataFrame,
) -> KaplanMeierFitter:
    """Fit one Kaplan-Meier retention estimator."""

    estimator = KaplanMeierFitter(label="Overall retention")
    estimator.fit(
        cohort["duration_months"],
        event_observed=cohort["event_observed"],
    )
    return estimator


def build_overall_curve(
    estimator: KaplanMeierFitter,
) -> pd.DataFrame:
    """Return the full Kaplan-Meier step curve and confidence interval."""

    survival = estimator.survival_function_.rename(
        columns={estimator._label: "survival_probability"}
    )
    confidence = estimator.confidence_interval_.copy()
    confidence.columns = [
        "survival_lower_95",
        "survival_upper_95",
    ]
    curve = survival.join(confidence).reset_index()
    curve = curve.rename(columns={curve.columns[0]: "duration_months"})
    curve["cumulative_attrition_probability"] = (
        1.0 - curve["survival_probability"]
    )
    return curve


def build_horizon_summary(
    estimator: KaplanMeierFitter,
    horizons: list[int],
) -> pd.DataFrame:
    """Summarize overall retention at committed tenure horizons."""

    rows = []
    confidence = estimator.confidence_interval_
    for horizon in horizons:
        survival = float(estimator.predict(horizon))
        index = confidence.index[confidence.index <= horizon]
        if len(index) == 0:
            lower = 1.0
            upper = 1.0
        else:
            latest = confidence.loc[index[-1]]
            lower = float(latest.iloc[0])
            upper = float(latest.iloc[1])
        rows.append(
            {
                "horizon_months": int(horizon),
                "retention_probability": survival,
                "retention_lower_95": lower,
                "retention_upper_95": upper,
                "cumulative_attrition_probability": 1.0 - survival,
            }
        )
    return pd.DataFrame(rows)


def build_group_summaries(
    cohort: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build group Kaplan-Meier summaries and omnibus log-rank tests."""

    horizons = list(config["kaplan_meier"]["horizons_months"])
    group_features = list(config["kaplan_meier"]["group_features"])
    minimum_size = int(config["kaplan_meier"]["minimum_group_size"])
    summaries: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []

    for feature in group_features:
        labels = cohort[feature].astype(str)
        result = multivariate_logrank_test(
            cohort["duration_months"],
            labels,
            cohort["event_observed"],
        )
        tests.append(
            {
                "group_feature": feature,
                "groups": int(labels.nunique()),
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
                "descriptive_difference_detected": bool(result.p_value < 0.05),
                "causal_interpretation_permitted": False,
            }
        )

        for group_name, group in cohort.groupby(feature, dropna=False):
            if len(group) < minimum_size:
                continue
            estimator = fit_kaplan_meier(group)
            row: dict[str, Any] = {
                "group_feature": feature,
                "group": str(group_name),
                "employees": int(len(group)),
                "observed_exits": int(group["event_observed"].sum()),
                "right_censored": int((1 - group["event_observed"]).sum()),
                "observed_exit_share": float(group["event_observed"].mean()),
                "median_retention_months": (
                    float(estimator.median_survival_time_)
                    if np.isfinite(estimator.median_survival_time_)
                    else np.nan
                ),
            }
            for horizon in horizons:
                row[f"retention_{horizon}m"] = float(estimator.predict(horizon))
            summaries.append(row)

    return pd.DataFrame(summaries), pd.DataFrame(tests)


def _safe_level_name(value: Any) -> str:
    """Create deterministic encoded feature suffixes."""

    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text or "Missing"


def build_cox_design(
    cohort: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a full-rank Cox matrix from baseline-at-hire features."""

    policy = config["cox_model"]
    numeric = list(policy["numeric_features"])
    categorical = list(policy["categorical_features"])
    references = dict(policy["reference_categories"])

    required = [
        "duration_months",
        "event_observed",
        *numeric,
        *categorical,
    ]
    if cohort[required].isna().any(axis=None):
        missing = cohort[required].isna().sum()
        raise ValueError(
            f"Cox baseline features contain missing values: "
            f"{missing[missing.gt(0)].to_dict()}"
        )

    design = cohort[["duration_months", "event_observed"]].reset_index(drop=True)
    manifest_rows: list[dict[str, Any]] = []

    for feature in numeric:
        values = cohort[feature].astype(float)
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=0))
        if standard_deviation <= 0:
            raise ValueError(f"Numeric Cox feature has no variation: {feature}")
        encoded = f"{feature}_z"
        design[encoded] = ((values - mean) / standard_deviation).to_numpy()
        manifest_rows.append(
            {
                "encoded_feature": encoded,
                "raw_feature": feature,
                "feature_type": "numeric_standardized",
                "reference_category": np.nan,
                "level": np.nan,
                "numeric_mean": mean,
                "numeric_standard_deviation": standard_deviation,
                "interpretation": (
                    f"{feature.replace('_', ' ').title()} "
                    "(per 1 SD increase)"
                ),
            }
        )

    for feature in categorical:
        values = cohort[feature].astype(str)
        levels = sorted(values.unique().tolist())
        reference = str(references[feature])
        if reference not in levels:
            raise ValueError(
                f"Reference category {reference!r} is absent for {feature}."
            )
        for level in levels:
            if level == reference:
                continue
            encoded = f"{feature}_{_safe_level_name(level)}"
            if encoded in design:
                raise ValueError(f"Duplicate encoded Cox feature: {encoded}")
            design[encoded] = values.eq(level).astype(float).to_numpy()
            manifest_rows.append(
                {
                    "encoded_feature": encoded,
                    "raw_feature": feature,
                    "feature_type": "categorical_indicator",
                    "reference_category": reference,
                    "level": level,
                    "numeric_mean": np.nan,
                    "numeric_standard_deviation": np.nan,
                    "interpretation": f"{level} compared with {reference}",
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    return design, manifest


def fit_cox_model(
    design: pd.DataFrame,
    config: dict[str, Any],
) -> CoxPHFitter:
    """Fit the committed penalized proportional-hazards model."""

    policy = config["cox_model"]
    model = CoxPHFitter(
        penalizer=float(policy["penalizer"]),
        l1_ratio=float(policy["l1_ratio"]),
    )
    model.fit(
        design,
        duration_col="duration_months",
        event_col="event_observed",
        show_progress=False,
    )
    return model


def build_hazard_ratio_table(
    model: CoxPHFitter,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Create an interpretable Cox hazard-ratio table."""

    summary = model.summary.reset_index().rename(columns={"covariate": "encoded_feature"})
    keep = [
        "encoded_feature",
        "coef",
        "exp(coef)",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
        "se(coef)",
        "z",
        "p",
    ]
    table = summary[keep].rename(
        columns={
            "coef": "log_hazard_coefficient",
            "exp(coef)": "hazard_ratio",
            "exp(coef) lower 95%": "hazard_ratio_lower_95",
            "exp(coef) upper 95%": "hazard_ratio_upper_95",
            "se(coef)": "standard_error",
            "p": "p_value",
        }
    )
    table = table.merge(
        manifest,
        on="encoded_feature",
        how="left",
        validate="one_to_one",
    )
    table["association_direction"] = np.where(
        table["hazard_ratio"].gt(1.0),
        "Higher modeled exit hazard",
        "Lower modeled exit hazard",
    )
    table["absolute_log_hazard"] = table["log_hazard_coefficient"].abs()
    return table.sort_values(
        ["absolute_log_hazard", "encoded_feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_cross_validation(
    design: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Measure out-of-fold concordance with deterministic stratification."""

    policy = config["cox_model"]
    folds = int(policy["cross_validation_folds"])
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(policy["random_seed"]),
    )
    target = design["event_observed"].astype(int)
    rows: list[dict[str, Any]] = []

    for fold, (fit_index, validation_index) in enumerate(
        splitter.split(design, target),
        start=1,
    ):
        fit_frame = design.iloc[fit_index].copy()
        validation_frame = design.iloc[validation_index].copy()
        model = fit_cox_model(fit_frame, config)
        partial_hazard = model.predict_partial_hazard(
            validation_frame
        ).to_numpy()
        score = concordance_index(
            validation_frame["duration_months"],
            -partial_hazard,
            validation_frame["event_observed"],
        )
        rows.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_frame)),
                "validation_rows": int(len(validation_frame)),
                "fit_events": int(fit_frame["event_observed"].sum()),
                "validation_events": int(
                    validation_frame["event_observed"].sum()
                ),
                "row_overlap": int(
                    len(set(fit_index).intersection(set(validation_index)))
                ),
                "concordance_index": float(score),
            }
        )

    return pd.DataFrame(rows)


def build_proportional_hazards_table(
    model: CoxPHFitter,
    design: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run a rank-time proportional-hazards diagnostic."""

    threshold = float(
        config["cox_model"]["proportional_hazards_flag_p_value"]
    )
    result = proportional_hazard_test(
        model,
        design,
        time_transform="rank",
    ).summary.reset_index()
    first_column = result.columns[0]
    result = result.rename(columns={first_column: "encoded_feature"})
    result = result.merge(
        manifest[["encoded_feature", "raw_feature", "interpretation"]],
        on="encoded_feature",
        how="left",
        validate="one_to_one",
    )
    result["configured_flag_p_value"] = threshold
    result["proportional_hazards_review_flag"] = result["p"].lt(threshold)
    result["diagnostic_only_not_model_retuning"] = True
    return result.sort_values(["p", "encoded_feature"]).reset_index(drop=True)


def build_population_summary(
    all_cohort: pd.DataFrame,
    model_cohort: pd.DataFrame,
    estimator: KaplanMeierFitter,
    cox_full_concordance: float,
    cross_validation: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize censoring, duration, and Cox discrimination."""

    median = estimator.median_survival_time_
    return pd.DataFrame(
        [
            {
                "as_of_date": config["cohort"]["as_of_date"],
                "all_employees": int(len(all_cohort)),
                "model_eligible_employees": int(len(model_cohort)),
                "protected_employees_excluded": int(
                    all_cohort["protected_level"].sum()
                ),
                "observed_exits": int(model_cohort["event_observed"].sum()),
                "right_censored_active_employees": int(
                    (1 - model_cohort["event_observed"]).sum()
                ),
                "observed_exit_share": float(
                    model_cohort["event_observed"].mean()
                ),
                "median_observed_duration_months": float(
                    model_cohort["duration_months"].median()
                ),
                "maximum_observed_duration_months": float(
                    model_cohort["duration_months"].max()
                ),
                "kaplan_meier_median_retention_months": (
                    float(median) if np.isfinite(median) else np.nan
                ),
                "kaplan_meier_median_reached": bool(np.isfinite(median)),
                "cox_full_cohort_concordance": float(cox_full_concordance),
                "cox_cv_mean_concordance": float(
                    cross_validation["concordance_index"].mean()
                ),
                "cox_cv_standard_deviation": float(
                    cross_validation["concordance_index"].std(ddof=1)
                ),
            }
        ]
    )


def validation_row(
    check: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> dict[str, Any]:
    """Create one validation result row."""

    return {
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "requirement": requirement,
        "details": details,
    }


def validate_outputs(
    all_cohort: pd.DataFrame,
    model_cohort: pd.DataFrame,
    overall_curve: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    logrank_tests: pd.DataFrame,
    design: pd.DataFrame,
    manifest: pd.DataFrame,
    hazard_ratios: pd.DataFrame,
    cross_validation: pd.DataFrame,
    ph_tests: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate censoring, leakage controls, diagnostics, and governance."""

    cohort_policy = config["cohort"]
    cox_policy = config["cox_model"]
    governance = config["governance"]
    as_of_date = pd.Timestamp(cohort_policy["as_of_date"])
    expected_horizons = list(config["kaplan_meier"]["horizons_months"])
    expected_groups = list(config["kaplan_meier"]["group_features"])
    expected_figures = list(config["figures"]["expected_files"])

    source_date_invalid = (
        model_cohort["hire_date"].gt(model_cohort["observation_end_date"])
        | model_cohort["observation_end_date"].gt(as_of_date)
    ).sum()
    status_mismatch = (
        model_cohort["event_observed"].eq(1)
        != model_cohort["employment_status"].eq("Terminated")
    ).sum()
    termination_mismatch = (
        model_cohort["event_observed"].eq(1)
        != model_cohort["termination_date"].notna()
    ).sum()
    direct_leakage = sorted(
        set(manifest["raw_feature"]).intersection(
            {
                "termination_date",
                "termination_type",
                "employment_status",
                "event_observed",
                "current_retention_probability",
                "selected_for_human_review",
            }
        )
    )
    forbidden_exports = sorted(
        set().union(*(set(frame.columns) for frame in [
            overall_curve,
            horizon_summary,
            group_summary,
            logrank_tests,
            hazard_ratios,
            cross_validation,
            ph_tests,
        ])).intersection({"employee_id", "first_name", "last_name"})
    )
    mean_concordance = float(cross_validation["concordance_index"].mean())
    ph_flags = int(ph_tests["proportional_hazards_review_flag"].sum())
    checks = [
        validation_row(
            "Configured as-of date is exact",
            as_of_date == pd.Timestamp("2026-06-30"),
            as_of_date.date(),
            "2026-06-30",
            "Active employees are censored at the generated-history boundary.",
        ),
        validation_row(
            "All-employee cohort is unique and complete",
            len(all_cohort) == 10000 and all_cohort["employee_id"].is_unique,
            f"{len(all_cohort)} rows; duplicates="
            f"{all_cohort['employee_id'].duplicated().sum()}",
            "10,000 rows; 0 duplicates",
            "Every synthetic employee has one time-to-event record.",
        ),
        validation_row(
            "Model cohort uses outcome-eligible levels",
            set(model_cohort["organizational_level"])
            == set(cohort_policy["outcome_eligible_levels"]),
            sorted(model_cohort["organizational_level"].unique()),
            cohort_policy["outcome_eligible_levels"],
            "Protected hierarchy levels remain in workforce totals but not the Cox model.",
        ),
        validation_row(
            "Model cohort is sufficiently large",
            len(model_cohort) >= int(cohort_policy["minimum_rows"]),
            len(model_cohort),
            f">= {cohort_policy['minimum_rows']}",
            "Survival estimates require a useful at-risk population.",
        ),
        validation_row(
            "Observed exits are sufficient",
            int(model_cohort["event_observed"].sum())
            >= int(cohort_policy["minimum_events"]),
            int(model_cohort["event_observed"].sum()),
            f">= {cohort_policy['minimum_events']}",
            "Observed events identify the retention curve and hazard associations.",
        ),
        validation_row(
            "Right-censored active records are sufficient",
            int((1 - model_cohort["event_observed"]).sum())
            >= int(cohort_policy["minimum_censored"]),
            int((1 - model_cohort["event_observed"]).sum()),
            f">= {cohort_policy['minimum_censored']}",
            "Active employees contribute observed tenure without invented exits.",
        ),
        validation_row(
            "Event indicators reconcile to status",
            status_mismatch == 0 and termination_mismatch == 0,
            f"status={status_mismatch}; termination={termination_mismatch}",
            "0 mismatches",
            "Only known terminations count as observed exit events.",
        ),
        validation_row(
            "Observation dates are chronologically valid",
            source_date_invalid == 0,
            int(source_date_invalid),
            "0 invalid dates",
            "No record starts after its end or extends past the as-of date.",
        ),
        validation_row(
            "Durations are positive and finite",
            model_cohort["duration_months"].gt(0).all()
            and np.isfinite(model_cohort["duration_months"]).all(),
            (
                f"min={model_cohort['duration_months'].min():.6f}; "
                f"max={model_cohort['duration_months'].max():.6f}"
            ),
            "All finite and > 0",
            "Inclusive tenure days support same-day events.",
        ),
        validation_row(
            "Hire dimensions reconstruct every employee",
            model_cohort[
                [
                    "hire_department_name",
                    "hire_region",
                    "hire_job_level",
                ]
            ].notna().all(axis=None),
            int(
                model_cohort[
                    [
                        "hire_department_name",
                        "hire_region",
                        "hire_job_level",
                    ]
                ].isna().sum().sum()
            ),
            "0 missing values",
            "Hire-event notes prevent later transfers from becoming baseline features.",
        ),
        validation_row(
            "Hire compensation aligns with hire date",
            model_cohort["hire_compensation_date"]
            .eq(model_cohort["hire_date"])
            .all(),
            int(
                model_cohort["hire_compensation_date"]
                .ne(model_cohort["hire_date"])
                .sum()
            ),
            "0 mismatches",
            "Salary position is calculated from the original hire record.",
        ),
        validation_row(
            "Kaplan-Meier curve is bounded and nonincreasing",
            overall_curve["survival_probability"].between(0, 1).all()
            and overall_curve["survival_probability"].diff().fillna(0).le(1e-12).all(),
            (
                f"min={overall_curve['survival_probability'].min():.6f}; "
                f"increases="
                f"{overall_curve['survival_probability'].diff().gt(1e-12).sum()}"
            ),
            "Inside [0, 1]; 0 increases",
            "Estimated retention cannot increase after an exit event.",
        ),
        validation_row(
            "Committed retention horizons are complete",
            horizon_summary["horizon_months"].tolist() == expected_horizons,
            horizon_summary["horizon_months"].tolist(),
            expected_horizons,
            "The table supports short- and medium-tenure interpretation.",
        ),
        validation_row(
            "Every configured subgroup is summarized",
            sorted(group_summary["group_feature"].unique())
            == sorted(expected_groups),
            sorted(group_summary["group_feature"].unique()),
            sorted(expected_groups),
            "Group curves are descriptive and use baseline-at-hire labels.",
        ),
        validation_row(
            "Every subgroup has an omnibus log-rank test",
            sorted(logrank_tests["group_feature"].unique())
            == sorted(expected_groups),
            sorted(logrank_tests["group_feature"].unique()),
            sorted(expected_groups),
            "P-values flag distribution differences but do not establish causes.",
        ),
        validation_row(
            "Cox design contains no direct leakage features",
            not direct_leakage,
            direct_leakage,
            [],
            "Only fixed demographics and reconstructed hire-time fields are modeled.",
        ),
        validation_row(
            "Cox encoded columns are unique and finite",
            manifest["encoded_feature"].is_unique
            and np.isfinite(
                design.drop(columns=["duration_months", "event_observed"])
            ).all(axis=None),
            (
                f"{len(manifest)} features; duplicates="
                f"{manifest['encoded_feature'].duplicated().sum()}"
            ),
            "Unique, finite encoded features",
            "Every hazard ratio maps to one baseline feature contrast.",
        ),
        validation_row(
            "Cox hazard ratios are positive and finite",
            hazard_ratios["hazard_ratio"].gt(0).all()
            and np.isfinite(
                hazard_ratios[
                    [
                        "hazard_ratio",
                        "hazard_ratio_lower_95",
                        "hazard_ratio_upper_95",
                    ]
                ]
            ).all(axis=None),
            f"{len(hazard_ratios)} finite estimates",
            f"{len(manifest)} finite positive estimates",
            "Exponentiated Cox coefficients must define valid hazard ratios.",
        ),
        validation_row(
            "Cross-validation folds are complete and disjoint",
            cross_validation["fold"].tolist()
            == list(range(1, int(cox_policy["cross_validation_folds"]) + 1))
            and cross_validation["row_overlap"].eq(0).all(),
            (
                f"folds={cross_validation['fold'].tolist()}; "
                f"max_overlap={cross_validation['row_overlap'].max()}"
            ),
            (
                f"{cox_policy['cross_validation_folds']} folds; "
                "0 row overlap"
            ),
            "Concordance is measured on rows not fitted in that fold.",
        ),
        validation_row(
            "Cross-validated concordance is plausible",
            float(cox_policy["minimum_mean_concordance"])
            <= mean_concordance
            <= float(cox_policy["maximum_mean_concordance"]),
            round(mean_concordance, 6),
            (
                f"{cox_policy['minimum_mean_concordance']}–"
                f"{cox_policy['maximum_mean_concordance']}"
            ),
            "The baseline model shows modest signal without implausible separation.",
        ),
        validation_row(
            "Proportional-hazards diagnostics are complete",
            set(ph_tests["encoded_feature"]) == set(manifest["encoded_feature"]),
            f"{len(ph_tests)} tests; review_flags={ph_flags}",
            f"{len(manifest)} tests; flags reported, not hidden",
            "A flag is a modeling caveat and does not automatically invalidate the extension.",
        ),
        validation_row(
            "Saved analysis tables are aggregate only",
            not forbidden_exports,
            forbidden_exports,
            [],
            "No employee IDs or personal names are written to survival outputs.",
        ),
        validation_row(
            "Primary model and frozen policy remain isolated",
            not governance["modifies_primary_classifier"]
            and not governance["accesses_reserved_test_target"]
            and not governance["modifies_frozen_policy"]
            and not governance["modifies_dashboard"],
            {
                key: governance[key]
                for key in [
                    "modifies_primary_classifier",
                    "accesses_reserved_test_target",
                    "modifies_frozen_policy",
                    "modifies_dashboard",
                ]
            },
            "All False",
            "Checkpoint 55 is descriptive and methodological only.",
        ),
        validation_row(
            "Survival governance prohibits automatic action",
            governance["human_review_required"]
            and not governance["automatic_employment_action_permitted"]
            and governance["associations_are_not_causes"],
            (
                f"human_review={governance['human_review_required']}; "
                f"automatic_action="
                f"{governance['automatic_employment_action_permitted']}; "
                f"noncausal={governance['associations_are_not_causes']}"
            ),
            "human_review=True; automatic_action=False; noncausal=True",
            "Time-to-event associations cannot authorize employment decisions.",
        ),
        validation_row(
            "All survival figures were generated",
            sorted(path.name for path in FIGURE_DIR.glob("*.png"))
            == sorted(expected_figures),
            sorted(path.name for path in FIGURE_DIR.glob("*.png")),
            sorted(expected_figures),
            "The notebook and documentation use reproducible aggregate charts.",
        ),
    ]

    validation = pd.DataFrame(checks)
    failures = validation.loc[validation["status"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(
            "Survival validation failed:\n"
            + failures[["check", "observed", "requirement"]].to_string(index=False)
        )
    return validation


def _save_figure(
    figure: Figure,
    path: Path,
    dpi: int,
) -> None:
    """Save and close one figure."""

    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_overall_curve_figure(
    estimator: KaplanMeierFitter,
    path: Path,
    dpi: int,
) -> None:
    """Save the overall retention curve with censor marks."""

    figure, axis = plt.subplots(figsize=(9, 5.5))
    estimator.plot_survival_function(
        ax=axis,
        ci_show=True,
        show_censors=True,
        censor_styles={"ms": 3, "marker": "|"},
        color="#1f5a85",
    )
    axis.set(
        title="Synthetic employee retention over tenure",
        xlabel="Months since hire",
        ylabel="Estimated probability still employed",
        ylim=(0.45, 1.01),
    )
    axis.grid(alpha=0.2)
    _save_figure(figure, path, dpi)


def save_group_curve_figure(
    cohort: pd.DataFrame,
    feature: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    """Save Kaplan-Meier curves for one baseline subgroup."""

    figure, axis = plt.subplots(figsize=(10, 6))
    for group_name, group in sorted(
        cohort.groupby(feature),
        key=lambda item: str(item[0]),
    ):
        estimator = KaplanMeierFitter(label=str(group_name))
        estimator.fit(
            group["duration_months"],
            group["event_observed"],
        )
        estimator.plot_survival_function(
            ax=axis,
            ci_show=False,
            show_censors=False,
        )
    axis.set(
        title=title,
        xlabel="Months since hire",
        ylabel="Estimated probability still employed",
        ylim=(0.45, 1.01),
    )
    axis.grid(alpha=0.2)
    axis.legend(title=feature.replace("_", " ").title(), fontsize=8)
    _save_figure(figure, path, dpi)


def save_hazard_ratio_figure(
    hazard_ratios: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Save a forest plot of adjusted Cox hazard ratios."""

    ordered = hazard_ratios.sort_values(
        ["hazard_ratio", "encoded_feature"],
        ascending=[True, True],
    ).copy()
    positions = np.arange(len(ordered))
    labels = [
        row.interpretation
        for row in ordered.itertuples(index=False)
    ]
    figure, axis = plt.subplots(
        figsize=(10, max(6, 0.38 * len(ordered))),
    )
    lower_error = (
        ordered["hazard_ratio"] - ordered["hazard_ratio_lower_95"]
    ).to_numpy()
    upper_error = (
        ordered["hazard_ratio_upper_95"] - ordered["hazard_ratio"]
    ).to_numpy()
    axis.errorbar(
        ordered["hazard_ratio"],
        positions,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        color="#1f5a85",
        ecolor="#718096",
        capsize=3,
    )
    axis.axvline(1.0, color="#b83227", linestyle="--", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set(
        title="Adjusted baseline-at-hire hazard ratios",
        xlabel="Hazard ratio with 95% confidence interval",
        ylabel="",
    )
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, path, dpi)


def main() -> None:
    """Run the complete Checkpoint 55 analysis."""

    config = load_survival_config()
    tables = load_source_tables()
    all_cohort, model_cohort = build_survival_cohort(tables, config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    estimator = fit_kaplan_meier(model_cohort)
    overall_curve = build_overall_curve(estimator)
    horizon_summary = build_horizon_summary(
        estimator,
        list(config["kaplan_meier"]["horizons_months"]),
    )
    group_summary, logrank_tests = build_group_summaries(
        model_cohort,
        config,
    )

    design, manifest = build_cox_design(model_cohort, config)
    cox_model = fit_cox_model(design, config)
    hazard_ratios = build_hazard_ratio_table(cox_model, manifest)
    cross_validation = build_cross_validation(design, config)
    ph_tests = build_proportional_hazards_table(
        cox_model,
        design,
        manifest,
        config,
    )
    population_summary = build_population_summary(
        all_cohort,
        model_cohort,
        estimator,
        float(cox_model.concordance_index_),
        cross_validation,
        config,
    )

    dpi = int(config["figures"]["dpi"])
    save_overall_curve_figure(
        estimator,
        FIGURE_DIR / "overall_survival_curve.png",
        dpi,
    )
    save_group_curve_figure(
        model_cohort,
        "employment_type",
        "Retention by employment type at hire",
        FIGURE_DIR / "survival_by_employment_type.png",
        dpi,
    )
    save_group_curve_figure(
        model_cohort,
        "hire_department_name",
        "Retention by department at hire",
        FIGURE_DIR / "survival_by_hire_department.png",
        dpi,
    )
    save_hazard_ratio_figure(
        hazard_ratios,
        FIGURE_DIR / "cox_hazard_ratios.png",
        dpi,
    )

    validation = validate_outputs(
        all_cohort,
        model_cohort,
        overall_curve,
        horizon_summary,
        group_summary,
        logrank_tests,
        design,
        manifest,
        hazard_ratios,
        cross_validation,
        ph_tests,
        config,
    )

    outputs = {
        POPULATION_SUMMARY_PATH: population_summary,
        OVERALL_CURVE_PATH: overall_curve,
        HORIZON_SUMMARY_PATH: horizon_summary,
        GROUP_SUMMARY_PATH: group_summary,
        LOGRANK_PATH: logrank_tests,
        COX_HAZARD_PATH: hazard_ratios,
        COX_CV_PATH: cross_validation,
        COX_PH_PATH: ph_tests,
        COX_MANIFEST_PATH: manifest,
        VALIDATION_PATH: validation,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)

    print("\nSURVIVAL POPULATION")
    print(population_summary.to_string(index=False))

    print("\nKAPLAN-MEIER RETENTION HORIZONS")
    print(horizon_summary.to_string(index=False))

    print("\nBASELINE GROUP RETENTION SUMMARY")
    display_columns = [
        "group_feature",
        "group",
        "employees",
        "observed_exits",
        "right_censored",
        "retention_12m",
        "retention_36m",
        "retention_60m",
    ]
    print(group_summary[display_columns].to_string(index=False))

    print("\nOMNIBUS LOG-RANK TESTS")
    print(logrank_tests.to_string(index=False))

    print("\nCOX BASELINE HAZARD RATIOS")
    display_hazards = hazard_ratios[
        [
            "interpretation",
            "hazard_ratio",
            "hazard_ratio_lower_95",
            "hazard_ratio_upper_95",
            "p_value",
            "association_direction",
        ]
    ]
    print(display_hazards.to_string(index=False))

    print("\nCOX CROSS-VALIDATION")
    print(cross_validation.to_string(index=False))
    print(
        "Mean concordance: "
        f"{cross_validation['concordance_index'].mean():.6f}"
    )

    print("\nPROPORTIONAL-HAZARDS DIAGNOSTICS")
    print(
        ph_tests[
            [
                "encoded_feature",
                "test_statistic",
                "p",
                "proportional_hazards_review_flag",
            ]
        ].to_string(index=False)
    )

    print("\nSURVIVAL ANALYSIS VALIDATION")
    print(validation.to_string(index=False))

    print(f"\nSaved survival outputs to: {OUTPUT_DIR}")
    print(
        "Employee-level survival rows saved: 0; "
        "primary model, final test, policy, and dashboard changes made: 0"
    )
    print("\nEMPLOYEE SURVIVAL ANALYSIS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
