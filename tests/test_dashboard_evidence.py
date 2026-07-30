"""Regression tests for current Version 2 dashboard evidence."""

from __future__ import annotations

from pathlib import Path

import yaml

from validate_dashboard_evidence import (
    dashboard_screenshot_references,
    extract_streamlit_tabs,
    screenshot_files,
)


EXPECTED_TABS = [
    "Overview",
    "Workforce",
    "Workforce Stability",
    "Recruiting",
    "Retention Risk",
    "Model Performance",
]

EXPECTED_LEGACY_SCREENSHOTS = {
    "docs/screenshots/01_overview.png",
    "docs/screenshots/02(1)_workforce.png",
    "docs/screenshots/02_workforce.png",
    "docs/screenshots/03(1)_recruiting.png",
    "docs/screenshots/03_recruiting.png",
    "docs/screenshots/04_retention_risk.png",
    "docs/screenshots/05_risk_table.png",
    "docs/screenshots/06(1)_model_performance.png",
    "docs/screenshots/06_model_performance.png",
}


def load_config(project_root: Path) -> dict:
    """Load the dashboard evidence contract."""

    with (project_root / "config" / "dashboard_evidence.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_reported_legacy_screenshot_population_is_exact(
    project_root: Path,
) -> None:
    """All nine review findings must remain named in the regression contract."""

    config = load_config(project_root)

    assert set(config["legacy_v1_screenshots"]) == EXPECTED_LEGACY_SCREENSHOTS


def test_live_dashboard_has_six_current_tabs(project_root: Path) -> None:
    """The source application must expose the complete Version 2 navigation."""

    config = load_config(project_root)
    tabs = extract_streamlit_tabs(project_root / config["dashboard_application"])

    assert tabs == EXPECTED_TABS
    assert tabs == config["expected_tabs"]


def test_current_branch_contains_no_static_dashboard_images(
    project_root: Path,
) -> None:
    """Unverified UI captures must not survive as contradictory orphans."""

    config = load_config(project_root)
    directory = project_root / config["screenshot_directory"]

    assert screenshot_files(directory) == []
    assert all(
        not (project_root / path).exists() for path in config["legacy_v1_screenshots"]
    )


def test_markdown_does_not_reference_removed_screenshots(
    project_root: Path,
) -> None:
    """Reviewer navigation must not contain broken screenshot embeds."""

    assert dashboard_screenshot_references(project_root) == []


def test_dashboard_guide_uses_current_v2_contract(project_root: Path) -> None:
    """The text evidence must agree with the live application and V2 README."""

    config = load_config(project_root)
    text = (project_root / config["dashboard_documentation"]).read_text(
        encoding="utf-8"
    )

    assert all(f"### {tab}" in text for tab in EXPECTED_TABS)
    assert all(value in text for value in config["required_current_values"])
    assert all(
        value not in text for value in config["forbidden_current_dashboard_values"]
    )
    assert "Static screenshots are intentionally not committed" in text


def test_github_actions_runs_dashboard_evidence_validator(
    project_root: Path,
) -> None:
    """CI must block a future reintroduction of the stale gallery."""

    workflow = (
        project_root / ".github" / "workflows" / "python-quality.yml"
    ).read_text(encoding="utf-8")

    assert "python src/validate_dashboard_evidence.py" in workflow


def test_checkpoint_runner_removes_only_named_legacy_images(
    project_root: Path,
) -> None:
    """The Windows installer must delete the exact nine reported files."""

    runner = (
        project_root / "scripts" / "checkpoints" / "run_checkpoint67.ps1"
    ).read_text(encoding="utf-8")

    for path in EXPECTED_LEGACY_SCREENSHOTS:
        assert path.replace("/", "\\") in runner

    assert "Remove-Item -LiteralPath" in runner
    assert "-Recurse" not in runner
    assert "src\\validate_dashboard_evidence.py" in runner
    assert "$Runner quality" in runner


def test_checkpoint_is_presentation_only(project_root: Path) -> None:
    """Screenshot cleanup must not mutate the decision system."""

    governance = load_config(project_root)["governance"]

    assert governance == {
        "changes_dashboard_application": False,
        "changes_dashboard_data": False,
        "changes_model_results": False,
        "changes_frozen_policy": False,
        "removes_legacy_visual_evidence_only": True,
    }
