"""Regression contracts for executable attrition-hazard configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from validate_hazard_configuration import build_checks


def load_contract(project_root: Path) -> dict:
    """Load the committed hazard-configuration contract."""

    with (project_root / "config" / "hazard_configuration_contract.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_hazard_configuration_contract_passes(project_root: Path) -> None:
    """Every declared simulation setting must affect executable behavior."""

    checks = build_checks(project_root, load_contract(project_root))

    assert checks
    assert not [check for check in checks if check["status"] != "PASS"]


def test_runtime_reference_map_covers_all_simulation_keys(
    project_root: Path,
) -> None:
    """The validation map itself cannot omit a YAML simulation key."""

    contract = load_contract(project_root)
    with (project_root / "config" / "attrition_hazard_config.yaml").open(
        encoding="utf-8"
    ) as handle:
        hazard_config = yaml.safe_load(handle)

    assert set(contract["required_settings"]) == set(hazard_config["simulation"])


def test_contract_declares_no_downstream_result_changes(
    project_root: Path,
) -> None:
    """Configuration wiring must not be disguised policy retuning."""

    governance = load_contract(project_root)["governance"]

    assert governance["preserves_committed_default_behavior"]
    assert not governance["changes_reference_random_seed"]
    assert not governance["changes_model_results"]
    assert not governance["changes_frozen_policy"]
    assert not governance["changes_dashboard_data"]
