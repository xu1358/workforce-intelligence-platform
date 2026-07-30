"""Contracts separating synthetic-generator checks from model evaluation."""

from __future__ import annotations

from pathlib import Path

import yaml

from validate_generator_signal_governance import build_checks


def load_contract(project_root: Path) -> dict:
    """Load the committed generator-signal governance contract."""

    with (project_root / "config" / "generator_signal_governance.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_generator_signal_governance_passes(project_root: Path) -> None:
    """All chronology, terminology, evidence, and wording checks must pass."""

    checks = build_checks(project_root, load_contract(project_root))

    assert checks
    assert not [check for check in checks if check["status"] != "PASS"]


def test_hazard_config_has_no_model_performance_target(
    project_root: Path,
) -> None:
    """The generator config must not appear to set classifier performance."""

    with (project_root / "config" / "attrition_hazard_config.yaml").open(
        encoding="utf-8"
    ) as handle:
        hazard_config = yaml.safe_load(handle)

    assert "calibration_targets" not in hazard_config
    targets = hazard_config["generator_acceptance_targets"]
    assert "model_roc_auc" not in targets
    assert targets["observable_signal_diagnostic"]["metric"] == (
        "five_fold_logistic_regression_roc_auc"
    )


def test_final_test_is_not_forced_into_generator_band(
    project_root: Path,
) -> None:
    """The unchanged final result must remain evidence against post-test tuning."""

    contract = load_contract(project_root)
    diagnostic = contract["diagnostic_contract"]
    later = contract["later_model_evidence"]

    assert later["final_test_roc_auc"] < diagnostic["minimum"]
    assert later["final_test_evaluated_once"]
    assert not later["post_test_retuning_performed"]


def test_checkpoint_is_naming_and_documentation_only(
    project_root: Path,
) -> None:
    """The fix must not change data, model, policy, or dashboard results."""

    governance = load_contract(project_root)["governance"]

    assert governance["documentation_and_naming_only"]
    assert not governance["changes_synthetic_records"]
    assert not governance["reruns_generator"]
    assert not governance["changes_model_results"]
    assert not governance["reopens_final_test"]
    assert not governance["changes_frozen_policy"]
    assert not governance["changes_dashboard_data"]
