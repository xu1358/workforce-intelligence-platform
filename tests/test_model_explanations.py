"""Tests for aggregate retention-model explanations and stability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_retention_explanations import (
    build_global_importance,
    build_quartile_summary,
    encoded_feature_to_raw,
    group_encoded_contributions,
    linear_shap_values,
    spearman_rank_correlation,
    stable_importance_order,
    top_k_jaccard,
)


def simple_linear_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    """Fit one small deterministic linear classifier."""

    frame = pd.DataFrame(
        {
            "first": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "second": [1.0, 0.5, 1.5, 3.0, 2.5, 4.0],
        }
    )
    target = np.array([0, 0, 0, 1, 1, 1])
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                StandardScaler(),
                ["first", "second"],
            )
        ],
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(frame, target)
    return model, frame


def test_committed_explanation_policy_is_governed(
    project_root: Any,
) -> None:
    """The explanation contract must prohibit causal and automatic use."""

    import yaml

    path = project_root / "config" / "retention_explanations.yaml"
    with path.open(encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)

    assert policy["model_contract"]["selected_base_model"] == "Logistic Regression"
    assert policy["explanation"]["output_scale"] == "base_model_log_odds"
    assert not policy["explanation"]["save_employee_level_explanations"]
    assert policy["governance"]["explanations_are_associations_not_causes"]
    assert not policy["governance"]["automatic_employment_action_permitted"]
    assert not policy["governance"]["policy_retuning_permitted"]


def test_linear_shap_values_exactly_reconstruct_log_odds() -> None:
    """Base value plus contributions must equal the model decision."""

    model, frame = simple_linear_pipeline()
    _, values, base_value, decision = linear_shap_values(
        model,
        frame,
        frame,
    )

    reconstructed = base_value + values.sum(axis=1)
    assert np.allclose(reconstructed, decision, atol=1e-12)


def test_linear_shap_background_is_centered() -> None:
    """Using the background as scoring data should center mean values."""

    model, frame = simple_linear_pipeline()
    _, values, _, _ = linear_shap_values(
        model,
        frame,
        frame,
    )

    assert np.allclose(values.mean(axis=0), 0.0, atol=1e-12)


def test_encoded_feature_mapping_and_grouping() -> None:
    """One-hot columns should sum into their original raw feature."""

    encoded_names = [
        "tenure_years",
        "department_name_Finance",
        "department_name_Sales",
    ]
    encoded_values = np.array(
        [
            [0.5, 0.2, 0.0],
            [-0.5, 0.0, -0.1],
        ]
    )
    raw_features = ["tenure_years", "department_name"]

    grouped = group_encoded_contributions(
        encoded_names,
        encoded_values,
        raw_features,
        ["department_name"],
    )

    assert (
        encoded_feature_to_raw(
            "department_name_Finance",
            raw_features,
            ["department_name"],
        )
        == "department_name"
    )
    assert np.allclose(
        grouped,
        np.array([[0.5, 0.2], [-0.5, -0.1]]),
    )
    assert np.allclose(
        grouped.sum(axis=1),
        encoded_values.sum(axis=1),
    )


def test_stable_importance_order_breaks_ties_by_name() -> None:
    """Equal importances should use an alphabetical feature tie-break."""

    order = stable_importance_order(
        np.array([0.5, 0.8, 0.8]),
        ["zeta", "beta", "alpha"],
    )

    assert order.tolist() == [2, 1, 0]


def test_rank_correlation_detects_same_and_reversed_order() -> None:
    """Spearman helper should cover perfect agreement and reversal."""

    first = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.isclose(
        spearman_rank_correlation(first, first),
        1.0,
    )
    assert np.isclose(
        spearman_rank_correlation(first, first[::-1]),
        -1.0,
    )


def test_top_k_jaccard_uses_feature_sets() -> None:
    """Top-k overlap should ignore within-set ordering."""

    first = np.array([4.0, 3.0, 2.0, 1.0])
    second = np.array([3.0, 4.0, 1.0, 2.0])
    names = ["a", "b", "c", "d"]

    assert np.isclose(
        top_k_jaccard(first, second, names, top_k=2),
        1.0,
    )


def test_global_importance_is_ranked_and_normalized() -> None:
    """Aggregate importance shares must sum to one."""

    values = np.array(
        [
            [1.0, 0.2],
            [-1.0, 0.4],
        ]
    )
    summary = build_global_importance(
        ["first", "second"],
        values,
    )

    assert summary["raw_feature"].tolist() == ["first", "second"]
    assert summary["importance_rank"].tolist() == [1, 2]
    assert np.isclose(summary["importance_share"].sum(), 1.0)


def test_quartile_summary_is_aggregate_only() -> None:
    """Quartile outputs should contain all groups without row identifiers."""

    values = np.arange(16, dtype=float).reshape(8, 2) / 10.0
    probability = np.linspace(0.05, 0.40, num=8)
    labels = [
        "Lowest probability",
        "Lower-middle probability",
        "Upper-middle probability",
        "Highest probability",
    ]
    summary = build_quartile_summary(
        ["first", "second"],
        values,
        probability,
        labels,
    )

    assert len(summary) == 8
    assert set(summary["probability_quartile"]) == set(labels)
    assert set(summary["employees"]) == {2}
    assert "employee_id" not in summary.columns
    assert "attrition_next_12m" not in summary.columns
