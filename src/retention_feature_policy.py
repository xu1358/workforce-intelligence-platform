"""Reusable preprocessing policy for Version 2 retention models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT / "config" / "retention_feature_policy.yaml"
)


def load_feature_policy(
    path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    """Load the committed Version 2 feature policy."""

    if not path.exists():
        raise FileNotFoundError(f"Missing feature policy: {path}")

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selected_features(policy: dict[str, Any]) -> list[str]:
    """Return numerical and categorical features in model order."""

    return (
        list(policy["numerical_features"])
        + list(policy["categorical_features"])
    )


def normalize_feature_frame(
    dataset: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Apply semantic normalization before sklearn preprocessing."""

    features = pd.DataFrame(
        dataset.loc[:, selected_features(policy)].copy()
    )

    job_level = pd.Series(
        pd.to_numeric(
            features["job_level"],
            errors="coerce",
        ),
        index=features.index,
    )
    features["job_level"] = (
        job_level
        .astype("Int64")
        .astype("string")
    )

    promotion = pd.Series(
        features["promotion_recommended"],
        index=features.index,
        dtype="string",
    )
    promotion = promotion.str.strip().str.title()
    promotion = promotion.where(promotion.isin(["False", "True"]))
    features["promotion_recommended"] = promotion.fillna(
        policy["preprocessing"]["promotion_missing_value"]
    )

    if policy["preprocessing"][
        "promotion_recency_missing_when_no_prior"
    ]:
        no_prior_promotion = pd.Series(
            features["no_prior_promotion"],
            index=features.index,
        )
        features.loc[
            no_prior_promotion.eq(1),
            "months_since_promotion",
        ] = np.nan

    return features


def build_preprocessor(
    policy: dict[str, Any],
) -> ColumnTransformer:
    """Build reference-category preprocessing from the policy."""

    numerical_features = list(policy["numerical_features"])
    categorical_features = list(policy["categorical_features"])
    category_order = [
        list(policy["reference_categories"][feature])
        for feature in categorical_features
    ]

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    categories=category_order,  # pyright: ignore[reportArgumentType]
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )


def validate_policy(
    dataset: pd.DataFrame,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate feature coverage and configured reference categories."""

    records: list[dict[str, Any]] = []
    target = policy["target_column"]
    metadata = set(policy["metadata_columns"])
    selected = set(selected_features(policy))
    dropped = set(policy["dropped_features"])
    assigned = metadata | selected | dropped | {target}
    actual = set(dataset.columns)

    records.append(
        {
            "check": "Every dataset column has a policy",
            "status": "PASS" if assigned == actual else "FAIL",
            "observed": len(assigned),
            "requirement": len(actual),
            "details": (
                f"Unassigned={sorted(actual - assigned)}; "
                f"unknown={sorted(assigned - actual)}"
            ),
        }
    )

    overlap = (
        (metadata & selected)
        | (metadata & dropped)
        | (selected & dropped)
    )
    records.append(
        {
            "check": "Policy groups do not overlap",
            "status": "PASS" if not overlap else "FAIL",
            "observed": sorted(overlap),
            "requirement": "No overlap",
            "details": "Metadata, selected, and dropped groups are exclusive.",
        }
    )

    missing_selected = selected - actual
    records.append(
        {
            "check": "Selected features exist",
            "status": "PASS" if not missing_selected else "FAIL",
            "observed": sorted(missing_selected),
            "requirement": "No missing selected features",
            "details": "Every selected feature must exist in the panel.",
        }
    )

    normalized = normalize_feature_frame(dataset, policy)
    missing_references: dict[str, str] = {}
    for feature in policy["categorical_features"]:
        reference = str(policy["reference_categories"][feature][0])
        observed = set(normalized[feature].dropna().astype(str))
        if reference not in observed:
            missing_references[feature] = reference

    records.append(
        {
            "check": "Reference categories exist",
            "status": "PASS" if not missing_references else "FAIL",
            "observed": missing_references or "All present",
            "requirement": "Every reference appears in development data",
            "details": "The first configured category is omitted by encoding.",
        }
    )

    failures = [
        record for record in records if record["status"] == "FAIL"
    ]
    if failures:
        raise ValueError(
            "Feature policy validation failed:\n"
            + pd.DataFrame(failures).to_string(index=False)
        )

    return records
