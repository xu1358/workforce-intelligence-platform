"""Tests for the isolated IBM attrition benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_ibm_attrition import (
    add_age_band,
    calculate_metrics,
    context_comparison,
    stable_top_fraction_mask,
)
from download_ibm_attrition_benchmark import (
    load_config,
    validate_source_frame,
)


def benchmark_config(project_root: Path) -> dict[str, Any]:
    """Load the committed external-benchmark contract."""

    return load_config(project_root / "config" / "ibm_attrition_benchmark.yaml")


def schema_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Create a minimal in-memory frame matching the pinned source schema."""

    schema = config["schema"]
    rows = int(schema["expected_rows"])
    frame = pd.DataFrame(
        {
            column: np.zeros(rows, dtype=int)
            for column in schema["expected_columns_in_order"]
        }
    )
    frame["EmployeeNumber"] = np.arange(1, rows + 1)
    frame["Attrition"] = str(schema["negative_label"])
    frame.loc[
        : int(schema["expected_positive_cases"]) - 1,
        "Attrition",
    ] = str(schema["positive_label"])

    for column, value in schema["constant_columns"].items():
        frame[column] = value

    return frame[list(schema["expected_columns_in_order"])]


def test_benchmark_source_is_pinned_and_licensed(
    project_root: Path,
) -> None:
    """The external file should have immutable provenance."""

    config = benchmark_config(project_root)
    source = config["source"]

    assert len(source["source_commit"]) == 40
    assert source["source_commit"] in source["raw_url"]
    assert len(source["sha256"]) == 64
    assert source["fictional_data"] is True
    assert "ODbL" in source["license_name"]
    assert "DbCL" in source["license_name"]

    notebook_path = project_root / config["artifacts"]["supporting_notebook"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    notebook_text = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    ).lower()

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell["outputs"]
    )
    assert "fictional" in notebook_text
    assert "not directly comparable" in notebook_text


def test_pinned_schema_validation_accepts_matching_frame(
    project_root: Path,
) -> None:
    """The source validator should accept the committed schema contract."""

    config = benchmark_config(project_root)

    validate_source_frame(schema_frame(config), config)


def test_pinned_schema_validation_rejects_changed_target(
    project_root: Path,
) -> None:
    """A silent source-label change must fail immediately."""

    config = benchmark_config(project_root)
    frame = schema_frame(config)
    frame.loc[0, "Attrition"] = "Unexpected"

    try:
        validate_source_frame(frame, config)
    except ValueError as error:
        assert "target counts changed" in str(error)
    else:
        raise AssertionError("Changed target values were not rejected.")


def test_feature_policy_excludes_target_identifier_and_gender(
    project_root: Path,
) -> None:
    """Sensitive, identifying, and outcome columns must not enter the model."""

    config = benchmark_config(project_root)
    policy = config["feature_policy"]
    selected = set(policy["numerical_features"]) | set(policy["categorical_features"])
    excluded = set(policy["excluded_features"])

    assert selected.isdisjoint(excluded)
    assert {"Attrition", "EmployeeNumber", "Gender"} <= excluded
    assert not {"Attrition", "EmployeeNumber", "Gender"} & selected


def test_top_fraction_selection_has_stable_tie_breaking() -> None:
    """Equal scores should be ordered by the committed identifier."""

    probabilities = np.array([0.8, 0.8, 0.8, 0.1])
    identifiers = np.array([30, 10, 20, 40])

    selected = stable_top_fraction_mask(
        probabilities,
        identifiers,
        fraction=0.50,
    )

    assert selected.tolist() == [False, True, True, False]


def test_metric_formulas_reconcile_on_hand_example() -> None:
    """Top-k counts, precision, capture, and lift should be transparent."""

    target = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.1])
    identifiers = np.array([1, 2, 3, 4])

    metrics = calculate_metrics(
        "test",
        target,
        probabilities,
        identifiers,
        top_fraction=0.50,
    )

    assert metrics["selected_count"] == 2
    assert metrics["selected_positive_cases"] == 1
    assert metrics["top_fraction_precision"] == 0.5
    assert metrics["top_fraction_capture"] == 0.5
    assert metrics["top_fraction_lift"] == 1.0


def test_age_band_boundaries_are_complete(
    project_root: Path,
) -> None:
    """Every configured boundary age should map to exactly one band."""

    config = benchmark_config(project_root)
    frame = pd.DataFrame({"Age": [18, 29, 30, 39, 40, 49, 50, 65]})

    bands = add_age_band(frame, config)

    assert bands.tolist() == [
        "Under 30",
        "Under 30",
        "30-39",
        "30-39",
        "40-49",
        "40-49",
        "50+",
        "50+",
    ]


def test_context_comparison_forbids_direct_performance_claim(
    project_root: Path,
) -> None:
    """Different targets and validation designs must stay contextual."""

    config = benchmark_config(project_root)
    metrics = pd.DataFrame(
        [
            {
                "method": "Sigmoid-calibrated Logistic Regression",
                "rows": 1470,
                "positive_rate": 0.16,
                "pr_auc": 0.60,
                "roc_auc": 0.82,
                "brier_score": 0.10,
                "top_fraction_lift": 4.0,
            }
        ]
    )

    comparison = context_comparison(metrics, config)

    assert not comparison["directly_comparable"].any()
    assert set(config["isolation"].values()) == {False}
