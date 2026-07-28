"""Shared fixtures for the Workforce Intelligence Platform tests."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SOURCE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one committed YAML configuration."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root."""

    return PROJECT_ROOT


@pytest.fixture(scope="session")
def hazard_config(project_root: Path) -> dict[str, Any]:
    """Load the attrition-hazard configuration."""

    return load_yaml(project_root / "config" / "attrition_hazard_config.yaml")


@pytest.fixture(scope="session")
def temporal_config(project_root: Path) -> dict[str, Any]:
    """Load the temporal-snapshot configuration."""

    return load_yaml(project_root / "config" / "temporal_snapshots.yaml")


@pytest.fixture(scope="session")
def feature_policy(project_root: Path) -> dict[str, Any]:
    """Load the Version 2 feature policy."""

    return load_yaml(project_root / "config" / "retention_feature_policy.yaml")


@pytest.fixture(scope="session")
def split_strategy(project_root: Path) -> dict[str, Any]:
    """Load the temporal model-split strategy."""

    return load_yaml(project_root / "config" / "model_validation_strategy.yaml")


@pytest.fixture(scope="session")
def cost_policy(project_root: Path) -> dict[str, Any]:
    """Load the retention-cost assumptions."""

    return load_yaml(project_root / "config" / "retention_cost_assumptions.yaml")


@pytest.fixture(scope="session")
def retention_policy(project_root: Path) -> dict[str, Any]:
    """Load the frozen retention-policy configuration."""

    return load_yaml(project_root / "config" / "retention_policy.yaml")


@pytest.fixture(scope="session")
def stability_config(project_root: Path) -> dict[str, Any]:
    """Load the workforce-stability configuration."""

    return load_yaml(project_root / "config" / "workforce_stability.yaml")
