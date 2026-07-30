"""Tests for the temporal base-rate and prior-shift audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit_temporal_prior_shift import (
    build_base_rate_summary,
    build_calibration_summary,
    build_shift_decomposition,
    prior_odds_multiplier,
)


def load_contract(project_root: Path) -> dict:
    """Load the committed prior-shift contract."""

    with (project_root / "config" / "temporal_prior_shift.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_base_rate_summary_keeps_full_and_model_populations_separate() -> None:
    """Protected hierarchy rows must not silently change model prevalence."""

    panel = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "snapshot_sequence": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "snapshot_date": [
                "2023-06-30",
                "2023-06-30",
                "2023-06-30",
                "2024-06-30",
                "2024-06-30",
                "2024-06-30",
                "2025-06-30",
                "2025-06-30",
                "2025-06-30",
            ],
            "prediction_end_date": [
                "2024-06-30",
                "2024-06-30",
                "2024-06-30",
                "2025-06-30",
                "2025-06-30",
                "2025-06-30",
                "2026-06-30",
                "2026-06-30",
                "2026-06-30",
            ],
            "organizational_level": [
                "Individual Contributor",
                "Team Manager",
                "Senior Manager",
            ]
            * 3,
            "attrition_next_12m": [0, 0, 1, 0, 1, 1, 1, 1, 0],
        }
    )
    config = {
        "columns": {
            "target": "attrition_next_12m",
            "model_eligible_levels": [
                "Individual Contributor",
                "Team Manager",
            ],
        },
        "periods": [
            {"snapshot_sequence": 1, "label": "train"},
            {"snapshot_sequence": 2, "label": "validation"},
            {"snapshot_sequence": 3, "label": "test"},
        ],
    }

    summary = build_base_rate_summary(panel, config)
    full = summary[summary["cohort"].eq("Full historical population")].sort_values(
        "snapshot_sequence"
    )
    eligible = summary[summary["cohort"].eq("Model-eligible population")].sort_values(
        "snapshot_sequence"
    )

    assert full["positive_rate"].tolist() == [1 / 3, 2 / 3, 2 / 3]
    assert eligible["positive_rate"].tolist() == [0.0, 0.5, 1.0]
    assert full["relative_change_from_first"].iloc[-1] == 1.0


def test_calibration_transport_uses_existing_aggregate_rows(
    project_root: Path,
) -> None:
    """The audit must not need employee-level final-test predictions."""

    contract = load_contract(project_root)
    expected = contract["expected"]
    validation = pd.DataFrame(
        [
            {
                "method": "Sigmoid",
                "rows": 5521,
                "positive_cases": 586,
                "observed_attrition_rate": expected["model_eligible_rates"][1],
                "mean_predicted_probability": expected["validation_mean_probability"],
                "mean_calibration_gap": expected["validation_calibration_gap"],
            }
        ]
    )
    final = pd.DataFrame(
        [
            {
                "evaluation_period": "2025 final test",
                "rows": 6641,
                "positive_cases": 803,
                "positive_rate": expected["model_eligible_rates"][2],
                "mean_predicted_probability": expected["final_test_mean_probability"],
                "mean_calibration_gap": expected["final_test_calibration_gap"],
            }
        ]
    )

    summary = build_calibration_summary(validation, final, contract)
    decomposition = build_shift_decomposition(summary, contract).iloc[0]

    assert summary["rows"].tolist() == [5521, 6641]
    assert not summary["final_test_recalibration_permitted"].any()
    assert np.isclose(
        decomposition["observed_relative_change"],
        expected["validation_to_test_relative_change"],
    )
    assert np.isclose(
        decomposition["calibration_gap_change"],
        expected["calibration_gap_deterioration"],
    )
    assert bool(decomposition["pure_label_shift_proven"]) is False
    assert bool(decomposition["final_test_adjustment_applied"]) is False


def test_prior_odds_multiplier_is_a_diagnostic_scalar() -> None:
    """Higher target prevalence should increase the implied prior odds."""

    multiplier = prior_odds_multiplier(0.10614019199420394, 0.1209155247703659)

    assert np.isclose(multiplier, 1.1583531809364793)


def test_contract_prohibits_final_test_recalibration(project_root: Path) -> None:
    """Observed test drift must remain a limitation, not a tuning input."""

    contract = load_contract(project_root)
    governance = contract["governance"]
    thresholds = contract["thresholds"]

    assert governance["post_evaluation_diagnostic"] is True
    assert governance["final_test_evaluated_once"] is True
    assert governance["aggregates_temporal_outcomes_for_prevalence"] is True
    assert governance["uses_final_test_prediction_rows"] is False
    assert governance["recalibrates_on_final_test"] is False
    assert governance["changes_selected_model"] is False
    assert governance["changes_frozen_policy"] is False
    assert governance["changes_dashboard_probabilities"] is False
    assert thresholds["reconciliation_tolerance"] <= 1e-12
    assert thresholds["serialized_aggregate_tolerance"] == 1e-9


def test_audit_source_avoids_row_level_final_predictions(
    project_root: Path,
) -> None:
    """The production audit must reuse aggregates instead of review rows."""

    source = (project_root / "src" / "audit_temporal_prior_shift.py").read_text(
        encoding="utf-8"
    )

    assert "retention_final_test_predictions.csv" not in source
    assert "retention_final_test_model_metrics.csv" not in source
    assert 'config["inputs"]' in source


def test_public_documentation_names_and_quantifies_prior_shift(
    project_root: Path,
) -> None:
    """The previously hidden calibration-transport limitation must be public."""

    paths = [
        project_root / "README.md",
        project_root / "docs" / "temporal_prior_shift.md",
        project_root / "docs" / "calibration_analysis.md",
        project_root
        / "notebooks"
        / "portfolio"
        / "02_model_development_and_final_evaluation.ipynb",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for phrase in [
        "prior-probability shift",
        "9.62%",
        "10.42%",
        "11.91%",
        "12.09%",
        "11.02%",
        "−1.07 percentage points",
    ]:
        assert phrase in combined
