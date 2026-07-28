"""Tests for the selected retention-feature preprocessing policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.retention_feature_policy import (
    build_preprocessor,
    load_feature_policy,
    normalize_feature_frame,
    selected_features,
    validate_policy,
)


def policy_dataset(policy: dict[str, Any]) -> pd.DataFrame:
    """Build a small frame that covers every configured category."""

    row_count = max(len(values) for values in policy["reference_categories"].values())
    data: dict[str, Any] = {}

    for index, column in enumerate(policy["metadata_columns"]):
        data[column] = [
            (100_000 + row if column == "employee_id" else f"metadata-{index}-{row}")
            for row in range(row_count)
        ]

    for index, column in enumerate(policy["numerical_features"]):
        data[column] = [float(row + index + 1) for row in range(row_count)]

    for column in policy["categorical_features"]:
        categories = policy["reference_categories"][column]
        data[column] = [categories[row % len(categories)] for row in range(row_count)]

    for index, column in enumerate(policy["dropped_features"]):
        data[column] = [f"dropped-{index}-{row}" for row in range(row_count)]

    data[policy["target_column"]] = [row % 2 for row in range(row_count)]

    return pd.DataFrame(data)


def test_selected_features_preserve_committed_order(
    feature_policy: dict[str, Any],
) -> None:
    """Model inputs should remain numerical first, then categorical."""

    result = selected_features(feature_policy)

    assert result == (
        feature_policy["numerical_features"] + feature_policy["categorical_features"]
    )
    assert len(result) == 21


def test_normalization_handles_job_level_and_missing_promotion(
    feature_policy: dict[str, Any],
) -> None:
    """Semantic cleanup should create stable categorical representations."""

    frame = policy_dataset(feature_policy).iloc[:2].copy()
    frame["job_level"] = frame["job_level"].astype("object")
    frame.loc[0, "job_level"] = 2
    frame.loc[1, "job_level"] = 3
    frame.loc[0, "promotion_recommended"] = None
    frame.loc[1, "promotion_recommended"] = " true "

    result = normalize_feature_frame(frame, feature_policy)

    assert result["job_level"].tolist() == ["2", "3"]
    assert result["promotion_recommended"].tolist() == [
        "False",
        "True",
    ]


def test_no_prior_promotion_masks_recency(
    feature_policy: dict[str, Any],
) -> None:
    """Promotion recency is undefined when no promotion has occurred."""

    frame = policy_dataset(feature_policy).iloc[:2].copy()
    frame["months_since_promotion"] = [12.0, 24.0]
    frame["no_prior_promotion"] = [1, 0]

    result = normalize_feature_frame(frame, feature_policy)

    assert np.isnan(result.loc[0, "months_since_promotion"])
    assert result.loc[1, "months_since_promotion"] == 24.0


def test_preprocessor_creates_full_rank_policy_width(
    feature_policy: dict[str, Any],
) -> None:
    """Reference-category encoding should produce 35 model columns."""

    frame = policy_dataset(feature_policy)
    normalized = normalize_feature_frame(frame, feature_policy)
    preprocessor = build_preprocessor(feature_policy)
    matrix = preprocessor.fit_transform(normalized)
    names = preprocessor.get_feature_names_out()

    assert matrix.shape == (len(frame), 35)
    assert len(names) == len(set(names)) == 35
    assert np.isfinite(matrix).all()


def test_unknown_category_is_ignored_at_transform_time(
    feature_policy: dict[str, Any],
) -> None:
    """A new category should not change the fitted feature width."""

    frame = policy_dataset(feature_policy)
    preprocessor = build_preprocessor(feature_policy)
    preprocessor.fit(normalize_feature_frame(frame, feature_policy))

    new_row = frame.iloc[[0]].copy()
    new_row["department_name"] = "New Department"
    with pytest.warns(UserWarning, match="unknown categories"):
        transformed = preprocessor.transform(
            normalize_feature_frame(new_row, feature_policy)
        )

    assert transformed.shape == (1, 35)
    assert np.isfinite(transformed).all()


def test_complete_policy_dataset_passes_validation(
    feature_policy: dict[str, Any],
) -> None:
    """Every panel column must be assigned to exactly one policy group."""

    checks = validate_policy(
        policy_dataset(feature_policy),
        feature_policy,
    )

    assert checks
    assert {record["status"] for record in checks} == {"PASS"}


def test_loading_missing_policy_fails(tmp_path: Path) -> None:
    """A missing policy file must produce an actionable error."""

    with pytest.raises(FileNotFoundError, match="Missing feature policy"):
        load_feature_policy(tmp_path / "missing.yaml")
