"""Diagnose redundancy and coefficient stability for retention features.

Only the configured development snapshots are used for target-informed
diagnostics. The reserved 2025 snapshot remains untouched for later
temporal test evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression

from retention_feature_policy import (
    build_preprocessor,
    load_feature_policy,
    normalize_feature_frame,
    selected_features,
    validate_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "retention_multi_snapshot.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_DIAGNOSTICS_PATH = (
    OUTPUT_DIR / "feature_diagnostics.csv"
)
NUMERIC_CORRELATION_PATH = (
    OUTPUT_DIR / "numeric_correlation_diagnostics.csv"
)
CATEGORICAL_ASSOCIATION_PATH = (
    OUTPUT_DIR / "categorical_association_diagnostics.csv"
)
VIF_PATH = OUTPUT_DIR / "vif_diagnostics.csv"
CONDITION_PATH = OUTPUT_DIR / "condition_number_diagnostics.csv"
MANIFEST_PATH = OUTPUT_DIR / "encoded_feature_manifest.csv"
COEFFICIENT_PATH = OUTPUT_DIR / "coefficient_stability.csv"
VALIDATION_PATH = OUTPUT_DIR / "feature_policy_validation.csv"
BOOTSTRAP_SUMMARY_PATH = OUTPUT_DIR / "bootstrap_run_summary.csv"


def load_dataset() -> pd.DataFrame:
    """Load the multi-snapshot historical panel."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing retention_multi_snapshot.csv. "
            "Run Checkpoint 38 first."
        )

    return pd.read_csv(DATA_PATH)


def corrected_cramers_v(
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Calculate bias-corrected Cramer's V without scipy."""

    table = pd.crosstab(
        first.fillna("<MISSING>").astype(str),
        second.fillna("<MISSING>").astype(str),
    )
    observed = table.to_numpy(dtype=float)
    sample_size = observed.sum()
    rows, columns = observed.shape

    if sample_size <= 1 or rows <= 1 or columns <= 1:
        return 0.0

    expected = (
        observed.sum(axis=1, keepdims=True)
        @ observed.sum(axis=0, keepdims=True)
        / sample_size
    )
    valid = expected > 0
    chi_square = float(
        (((observed - expected) ** 2) / expected)[valid].sum()
    )
    phi_squared = chi_square / sample_size
    correction = (
        (columns - 1) * (rows - 1) / (sample_size - 1)
    )
    corrected_phi = max(0.0, phi_squared - correction)
    corrected_rows = rows - ((rows - 1) ** 2) / (
        sample_size - 1
    )
    corrected_columns = columns - ((columns - 1) ** 2) / (
        sample_size - 1
    )
    denominator = min(corrected_rows - 1, corrected_columns - 1)

    if denominator <= 0:
        return 0.0

    return float(np.sqrt(corrected_phi / denominator))


def pairwise_numeric_correlations(
    development: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Calculate candidate and selected numeric feature correlations."""

    excluded = set(policy["metadata_columns"]) | {
        policy["target_column"],
        "job_level",
    }
    candidate_features = [
        column
        for column in development.select_dtypes(
            include=[np.number]
        ).columns
        if column not in excluded
    ]

    records: list[dict[str, Any]] = []
    stages = {
        "candidate": candidate_features,
        "selected": list(policy["numerical_features"]),
    }
    threshold = float(
        policy["diagnostic_thresholds"]["high_pairwise_correlation"]
    )

    for stage, features in stages.items():
        numeric_frame = pd.DataFrame(
            development.loc[:, features]
        )
        correlation = numeric_frame.corr(method="pearson")
        for first_index, first in enumerate(features):
            for second in features[first_index + 1 :]:
                value = correlation.loc[first, second]
                if pd.isna(value):
                    continue
                records.append(
                    {
                        "stage": stage,
                        "first_feature": first,
                        "second_feature": second,
                        "correlation": float(value),
                        "absolute_correlation": abs(float(value)),
                        "above_threshold": abs(float(value))
                        >= threshold,
                    }
                )

    return pd.DataFrame(records).sort_values(
        ["stage", "absolute_correlation"],
        ascending=[True, False],
    )


def categorical_associations(
    development: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Calculate candidate categorical feature associations."""

    excluded = set(policy["metadata_columns"]) | {
        policy["target_column"]
    }
    features: list[str] = []
    categorical_numeric_features = {
        "job_level",
        "promotion_recommended",
    }
    for column in development.columns:
        if column in excluded:
            continue

        values = pd.Series(
            development.loc[:, column],
            index=development.index,
        )
        if (
            not is_numeric_dtype(values.dtype)
            or column in categorical_numeric_features
        ):
            features.append(column)

    records: list[dict[str, Any]] = []
    for first_index, first in enumerate(features):
        for second in features[first_index + 1 :]:
            first_values = pd.Series(
                development.loc[:, first],
                index=development.index,
            )
            second_values = pd.Series(
                development.loc[:, second],
                index=development.index,
            )
            records.append(
                {
                    "first_feature": first,
                    "second_feature": second,
                    "corrected_cramers_v": corrected_cramers_v(
                        first_values,
                        second_values,
                    ),
                    "first_cardinality": int(
                        development[first].nunique(dropna=False)
                    ),
                    "second_cardinality": int(
                        development[second].nunique(dropna=False)
                    ),
                }
            )

    return pd.DataFrame(records).sort_values(
        "corrected_cramers_v",
        ascending=False,
    )


def strongest_association_lookup(
    numeric_correlations: pd.DataFrame,
    categorical_table: pd.DataFrame,
) -> dict[str, tuple[str, float, str]]:
    """Return each raw feature's strongest same-type association."""

    lookup: dict[str, tuple[str, float, str]] = {}
    candidate_numeric = numeric_correlations.loc[
        numeric_correlations["stage"].eq("candidate")
    ]
    for row in candidate_numeric.itertuples(index=False):
        for feature, other in [
            (row.first_feature, row.second_feature),
            (row.second_feature, row.first_feature),
        ]:
            current = lookup.get(feature)
            value = float(row.absolute_correlation)
            if current is None or value > current[1]:
                lookup[feature] = (
                    other,
                    value,
                    "Pearson correlation",
                )

    for row in categorical_table.itertuples(
        index=False,
        name=None,
    ):
        (
            first_feature,
            second_feature,
            cramers_v,
            _first_cardinality,
            _second_cardinality,
        ) = row
        for feature, other in [
            (first_feature, second_feature),
            (second_feature, first_feature),
        ]:
            current = lookup.get(feature)
            value = float(cramers_v)
            if current is None or value > current[1]:
                lookup[feature] = (
                    other,
                    value,
                    "Corrected Cramer's V",
                )

    return lookup


def feature_inventory(
    development: pd.DataFrame,
    policy: dict[str, Any],
    numeric_correlations: pd.DataFrame,
    categorical_table: pd.DataFrame,
) -> pd.DataFrame:
    """Create a complete keep, drop, metadata, and target inventory."""

    selected_numeric = set(policy["numerical_features"])
    selected_categorical = set(policy["categorical_features"])
    metadata = set(policy["metadata_columns"])
    target = policy["target_column"]
    dropped = policy["dropped_features"]
    association = strongest_association_lookup(
        numeric_correlations,
        categorical_table,
    )

    records: list[dict[str, Any]] = []
    for feature in development.columns:
        if feature in selected_numeric:
            decision = "KEEP"
            role = "Numerical model feature"
            reason = (
                "Retained after semantic and statistical redundancy review."
            )
        elif feature in selected_categorical:
            decision = "KEEP"
            role = "Categorical model feature"
            reference = policy["reference_categories"][feature][0]
            reason = (
                f"Retained with {reference!r} as its reference category."
            )
        elif feature in metadata:
            decision = "EXCLUDE"
            role = "Metadata"
            reason = (
                "Retained for auditing or splitting, not model fitting."
            )
        elif feature == target:
            decision = "TARGET"
            role = "Outcome"
            reason = "Prediction target; never used as an input feature."
        else:
            decision = "DROP"
            role = "Redundant or unsuitable feature"
            reason = dropped[feature]

        partner, strength, method = association.get(
            feature,
            (None, np.nan, None),
        )
        records.append(
            {
                "feature": feature,
                "decision": decision,
                "role": role,
                "raw_dtype": str(development[feature].dtype),
                "missing_rate": float(
                    development[feature].isna().mean()
                ),
                "unique_values": int(
                    development[feature].nunique(dropna=False)
                ),
                "strongest_associated_feature": partner,
                "association_strength": strength,
                "association_method": method,
                "reason": reason,
            }
        )

    return pd.DataFrame(records)


def encoded_feature_manifest(
    preprocessor: Any,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Describe each encoded model column and its comparison basis."""

    names = list(preprocessor.get_feature_names_out())
    numerical = set(policy["numerical_features"])
    categorical = list(policy["categorical_features"])
    records: list[dict[str, Any]] = []

    for encoded_name in names:
        if encoded_name in numerical:
            records.append(
                {
                    "encoded_feature": encoded_name,
                    "source_feature": encoded_name,
                    "feature_type": "Numerical",
                    "category": None,
                    "reference_category": None,
                    "coefficient_comparison": (
                        "One standard-deviation increase after imputation"
                    ),
                }
            )
            continue

        source = next(
            feature
            for feature in sorted(
                categorical,
                key=len,
                reverse=True,
            )
            if encoded_name.startswith(f"{feature}_")
        )
        category = encoded_name[len(source) + 1 :]
        reference = str(policy["reference_categories"][source][0])
        records.append(
            {
                "encoded_feature": encoded_name,
                "source_feature": source,
                "feature_type": "Categorical indicator",
                "category": category,
                "reference_category": reference,
                "coefficient_comparison": (
                    f"{category!r} compared with reference {reference!r}"
                ),
            }
        )

    return pd.DataFrame(records)


def standardized_encoded_matrix(
    encoded: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Center and scale every encoded column for matrix diagnostics."""

    centered = encoded - encoded.mean(axis=0)
    standard_deviation = centered.std(axis=0)
    nonconstant = standard_deviation > 1e-12
    standardized = (
        centered[:, nonconstant] / standard_deviation[nonconstant]
    )
    kept_names = [
        name
        for name, keep in zip(names, nonconstant)
        if keep
    ]

    return standardized, kept_names


def matrix_diagnostics(
    encoded: np.ndarray,
    names: list[str],
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Calculate VIF, rank, and condition number."""

    standardized, kept_names = standardized_encoded_matrix(
        encoded,
        names,
    )
    singular_values = np.linalg.svd(
        standardized,
        compute_uv=False,
    )
    condition_number = float(
        singular_values[0] / singular_values[-1]
    )
    matrix_rank = int(np.linalg.matrix_rank(standardized))

    correlation = np.corrcoef(standardized, rowvar=False)
    try:
        inverse_correlation = np.linalg.inv(correlation)
        inverse_method = "inverse"
    except np.linalg.LinAlgError:
        inverse_correlation = np.linalg.pinv(correlation)
        inverse_method = "pseudoinverse"

    vif_values = np.diag(inverse_correlation)
    acceptable_vif = float(
        policy["diagnostic_thresholds"]["acceptable_maximum_vif"]
    )
    elevated_vif = float(
        policy["diagnostic_thresholds"]["elevated_vif"]
    )
    vif = pd.DataFrame(
        {
            "encoded_feature": kept_names,
            "vif": vif_values,
        }
    )
    vif["assessment"] = np.select(
        [
            vif["vif"].le(acceptable_vif),
            vif["vif"].le(elevated_vif),
        ],
        [
            "Acceptable",
            "Elevated",
        ],
        default="High",
    )
    vif = vif.sort_values("vif", ascending=False)

    condition_threshold = float(
        policy["diagnostic_thresholds"][
            "acceptable_condition_number"
        ]
    )
    condition = pd.DataFrame(
        {
            "metric": [
                "Development rows",
                "Raw selected features",
                "Encoded features",
                "Matrix rank",
                "Condition number",
                "Maximum VIF",
                "VIF matrix method",
            ],
            "value": [
                len(encoded),
                len(selected_features(policy)),
                len(names),
                matrix_rank,
                condition_number,
                float(vif["vif"].max()),
                inverse_method,
            ],
            "requirement": [
                "> 0",
                "Policy defined",
                "Unique encoded columns",
                str(len(kept_names)),
                f"<= {condition_threshold}",
                f"<= {acceptable_vif}",
                "inverse preferred",
            ],
        }
    )

    return vif, condition, matrix_rank


def cluster_bootstrap_coefficients(
    encoded: np.ndarray,
    target: np.ndarray,
    employee_ids: np.ndarray,
    names: list[str],
    manifest: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap regularized coefficients by employee cluster."""

    settings = policy["coefficient_bootstrap"]
    iterations = int(settings["iterations"])
    random_seed = int(settings["random_seed"])
    confidence_level = float(settings["confidence_level"])
    minimum_stability = float(settings["minimum_sign_stability"])
    rng = np.random.default_rng(random_seed)

    unique_employees = np.unique(employee_ids)
    positions = {
        employee_id: np.flatnonzero(employee_ids == employee_id)
        for employee_id in unique_employees
    }

    model_arguments = {
        "C": float(settings["logistic_c"]),
        "class_weight": settings["class_weight"],
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": random_seed,
    }
    full_model = LogisticRegression(**model_arguments)
    full_model.fit(encoded, target)
    full_coefficients = full_model.coef_[0]

    bootstrap_values = np.empty((iterations, len(names)))
    completed = 0
    attempts = 0
    maximum_attempts = iterations * 3

    while completed < iterations and attempts < maximum_attempts:
        attempts += 1
        sampled_employees = rng.choice(
            unique_employees,
            size=len(unique_employees),
            replace=True,
        )
        sampled_positions = np.concatenate(
            [positions[employee_id] for employee_id in sampled_employees]
        )
        sampled_target = target[sampled_positions]
        if np.unique(sampled_target).size < 2:
            continue

        model = LogisticRegression(**model_arguments)
        model.fit(encoded[sampled_positions], sampled_target)
        bootstrap_values[completed] = model.coef_[0]
        completed += 1

        if completed % 50 == 0:
            print(
                f"Completed coefficient bootstrap: "
                f"{completed}/{iterations}"
            )

    if completed != iterations:
        raise RuntimeError(
            f"Only {completed} of {iterations} bootstraps completed."
        )

    alpha = (1.0 - confidence_level) / 2.0
    lower = np.quantile(bootstrap_values, alpha, axis=0)
    upper = np.quantile(bootstrap_values, 1.0 - alpha, axis=0)
    median = np.median(bootstrap_values, axis=0)
    positive_share = (bootstrap_values > 0).mean(axis=0)
    negative_share = (bootstrap_values < 0).mean(axis=0)
    sign_stability = np.maximum(positive_share, negative_share)
    interval_excludes_zero = (lower > 0) | (upper < 0)
    reportable = (
        interval_excludes_zero
        & (sign_stability >= minimum_stability)
    )

    direction = np.where(
        reportable & (median > 0),
        "Higher predicted attrition risk",
        np.where(
            reportable & (median < 0),
            "Lower predicted attrition risk",
            "Unstable or inconclusive",
        ),
    )

    coefficients = manifest.copy()
    coefficients["full_sample_coefficient"] = full_coefficients
    coefficients["bootstrap_mean"] = bootstrap_values.mean(axis=0)
    coefficients["bootstrap_median"] = median
    coefficients["bootstrap_standard_deviation"] = (
        bootstrap_values.std(axis=0, ddof=1)
    )
    coefficients["bootstrap_lower_95"] = lower
    coefficients["bootstrap_upper_95"] = upper
    coefficients["median_odds_ratio"] = np.exp(median)
    coefficients["positive_share"] = positive_share
    coefficients["negative_share"] = negative_share
    coefficients["sign_stability"] = sign_stability
    coefficients["interval_excludes_zero"] = interval_excludes_zero
    coefficients["reportable_direction"] = reportable
    coefficients["direction"] = direction
    coefficients["absolute_bootstrap_median"] = np.abs(median)
    coefficients = coefficients.sort_values(
        "absolute_bootstrap_median",
        ascending=False,
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "Development snapshots",
                "Development rows",
                "Unique employee clusters",
                "Positive cases",
                "Bootstrap iterations",
                "Bootstrap attempts",
                "Encoded coefficients",
                "Reportable coefficient directions",
                "Reserved holdout target rows used",
            ],
            "value": [
                ", ".join(policy["development_snapshots"]),
                len(target),
                len(unique_employees),
                int(target.sum()),
                completed,
                attempts,
                len(names),
                int(reportable.sum()),
                0,
            ],
        }
    )

    return coefficients, summary


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


def main() -> None:
    """Run the complete Checkpoint 39 diagnostic workflow."""

    dataset = load_dataset()
    policy = load_feature_policy()
    development_dates = set(policy["development_snapshots"])
    holdout_date = str(policy["reserved_holdout_snapshot"])
    eligible_levels = set(
        policy["model_population"]["eligible_organizational_levels"]
    )
    model_population = dataset.loc[
        dataset["organizational_level"].isin(
            sorted(eligible_levels)
        )
    ].copy()
    excluded_population_rows = len(dataset) - len(model_population)

    development = model_population.loc[
        model_population["snapshot_date"].astype(str).isin(
            development_dates
        )
    ].copy()
    holdout_rows = int(
        model_population["snapshot_date"]
        .astype(str)
        .eq(holdout_date)
        .sum()
    )

    checks = validate_policy(development, policy)
    target_column = policy["target_column"]
    target = development[target_column].astype(int).to_numpy()

    numeric_correlations = pairwise_numeric_correlations(
        development,
        policy,
    )
    categorical_table = categorical_associations(
        development,
        policy,
    )
    inventory = feature_inventory(
        development,
        policy,
        numeric_correlations,
        categorical_table,
    )

    normalized = normalize_feature_frame(development, policy)
    preprocessor = build_preprocessor(policy)
    encoded_result = preprocessor.fit_transform(normalized)
    encoded = np.asarray(encoded_result, dtype=float)
    names = list(preprocessor.get_feature_names_out())
    manifest = encoded_feature_manifest(preprocessor, policy)

    vif, condition, matrix_rank = matrix_diagnostics(
        encoded,
        names,
        policy,
    )
    coefficients, bootstrap_summary = (
        cluster_bootstrap_coefficients(
            encoded=encoded,
            target=target,
            employee_ids=development["employee_id"].to_numpy(),
            names=names,
            manifest=manifest,
            policy=policy,
        )
    )

    expected_development_dates = set(policy["development_snapshots"])
    observed_development_dates = set(
        development["snapshot_date"].astype(str)
    )
    add_check(
        checks,
        "Model population uses outcome-eligible levels",
        set(development["organizational_level"]) == eligible_levels,
        sorted(development["organizational_level"].unique()),
        str(sorted(eligible_levels)),
        (
            "Protected hierarchy levels remain in workforce data but "
            "are excluded from retention-model diagnostics."
        ),
    )
    add_check(
        checks,
        "Development snapshots are exact",
        observed_development_dates == expected_development_dates,
        sorted(observed_development_dates),
        str(sorted(expected_development_dates)),
        "Only development periods may inform the feature policy.",
    )
    add_check(
        checks,
        "Reserved holdout remains unused",
        holdout_rows > 0
        and int(
            bootstrap_summary.loc[
                bootstrap_summary["metric"].eq(
                    "Reserved holdout target rows used"
                ),
                "value",
            ].iloc[0]
        )
        == 0,
        f"{holdout_rows} reserved rows; 0 target rows used",
        "Holdout exists and target usage equals 0",
        "The 2025 outcomes remain untouched for later final testing.",
    )
    add_check(
        checks,
        "Encoded feature names are unique",
        len(names) == len(set(names)),
        len(names),
        f"{len(names)} unique names",
        "Every model coefficient must map to one encoded feature.",
    )
    add_check(
        checks,
        "Encoded matrix has full column rank",
        matrix_rank == len(names),
        matrix_rank,
        str(len(names)),
        "Reference encoding must remove exact dummy-variable traps.",
    )

    maximum_vif = float(vif["vif"].max())
    acceptable_vif = float(
        policy["diagnostic_thresholds"]["acceptable_maximum_vif"]
    )
    add_check(
        checks,
        "Maximum VIF is acceptable",
        maximum_vif <= acceptable_vif,
        round(maximum_vif, 4),
        f"<= {acceptable_vif}",
        "Selected encoded columns should avoid severe multicollinearity.",
    )

    condition_number = float(
        condition.loc[
            condition["metric"].eq("Condition number"),
            "value",
        ].iloc[0]
    )
    condition_threshold = float(
        policy["diagnostic_thresholds"][
            "acceptable_condition_number"
        ]
    )
    add_check(
        checks,
        "Condition number is acceptable",
        condition_number <= condition_threshold,
        round(condition_number, 4),
        f"<= {condition_threshold}",
        "The standardized design matrix should remain well conditioned.",
    )

    completed_bootstraps = int(
        bootstrap_summary.loc[
            bootstrap_summary["metric"].eq("Bootstrap iterations"),
            "value",
        ].iloc[0]
    )
    requested_bootstraps = int(
        policy["coefficient_bootstrap"]["iterations"]
    )
    add_check(
        checks,
        "Cluster bootstraps complete",
        completed_bootstraps == requested_bootstraps,
        completed_bootstraps,
        str(requested_bootstraps),
        "Employees, rather than individual rows, are resampled.",
    )
    finite_coefficients = bool(
        np.isfinite(
            coefficients[
                [
                    "bootstrap_median",
                    "bootstrap_lower_95",
                    "bootstrap_upper_95",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    add_check(
        checks,
        "Bootstrap coefficients are finite",
        finite_coefficients,
        finite_coefficients,
        "True",
        "No coefficient interval may contain NaN or infinity.",
    )

    validation = pd.DataFrame(checks)
    failures = validation.loc[validation["status"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(
            "Checkpoint 39 validation failed:\n"
            + failures.to_string(index=False)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        FEATURE_DIAGNOSTICS_PATH: inventory,
        NUMERIC_CORRELATION_PATH: numeric_correlations,
        CATEGORICAL_ASSOCIATION_PATH: categorical_table,
        VIF_PATH: vif,
        CONDITION_PATH: condition,
        MANIFEST_PATH: manifest,
        COEFFICIENT_PATH: coefficients,
        VALIDATION_PATH: validation,
        BOOTSTRAP_SUMMARY_PATH: bootstrap_summary,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)

    print("\nFEATURE POLICY SUMMARY")
    print(
        f"Development rows: {len(development):,}\n"
        f"Protected-level rows excluded: "
        f"{excluded_population_rows:,}\n"
        f"Reserved holdout rows: {holdout_rows:,}\n"
        f"Original dataset columns: {len(dataset.columns)}\n"
        f"Selected raw model features: "
        f"{len(selected_features(policy))}\n"
        f"Encoded model features: {len(names)}\n"
        f"Dropped redundant features: "
        f"{len(policy['dropped_features'])}"
    )

    print("\nMATRIX DIAGNOSTICS")
    print(condition.to_string(index=False))

    print("\nHIGHEST VIF VALUES")
    print(vif.head(12).to_string(index=False))

    print("\nSTRONGEST ORIGINAL NUMERIC CORRELATIONS")
    print(
        numeric_correlations.loc[
            numeric_correlations["stage"].eq("candidate")
        ]
        .head(12)
        .to_string(index=False)
    )

    print("\nSTRONGEST ORIGINAL CATEGORICAL ASSOCIATIONS")
    print(categorical_table.head(12).to_string(index=False))

    print("\nMOST STABLE COEFFICIENT DIRECTIONS")
    print(
        coefficients.loc[
            coefficients["reportable_direction"],
            [
                "encoded_feature",
                "bootstrap_median",
                "bootstrap_lower_95",
                "bootstrap_upper_95",
                "sign_stability",
                "direction",
            ],
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nFEATURE POLICY VALIDATION")
    print(validation.to_string(index=False))
    print(
        "\nFEATURE REDUNDANCY AND STABILITY DIAGNOSTICS "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
