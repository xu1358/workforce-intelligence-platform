"""Tests for local and GitHub code-quality configuration."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

import yaml


def load_workflow(path: Path) -> dict[str, Any]:
    """Load GitHub workflow YAML without YAML 1.1 boolean coercion."""

    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def test_ruff_configuration_has_committed_scope(
    project_root: Path,
) -> None:
    """Ruff must enforce correctness rules across Python 3.12 code."""

    with (project_root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    ruff = config["tool"]["ruff"]
    lint = ruff["lint"]

    assert ruff["target-version"] == "py312"
    assert set(lint["select"]) == {"E4", "E7", "E9", "F"}
    assert set(lint["per-file-ignores"]) == {
        "src/analyze_employee_survival.py",
        "src/analyze_model_fairness.py",
        "src/analyze_retention_explanations.py",
        "src/evaluate_retention_ranking.py",
        "src/optimize_retention_policy.py",
    }


def test_workflow_has_expected_triggers(project_root: Path) -> None:
    """Quality automation must run on pushes, PRs, and manual requests."""

    workflow = load_workflow(
        project_root / ".github" / "workflows" / "python-quality.yml"
    )

    assert set(workflow["on"]) == {
        "push",
        "pull_request",
        "workflow_dispatch",
    }
    assert workflow["on"]["push"]["branches"] == [
        "master",
        "revision-v2",
    ]
    assert workflow["on"]["pull_request"]["branches"] == [
        "master",
        "revision-v2",
    ]


def test_workflow_uses_pinned_official_actions(
    project_root: Path,
) -> None:
    """External workflow actions should use explicit major versions."""

    workflow = load_workflow(
        project_root / ".github" / "workflows" / "python-quality.yml"
    )
    steps = workflow["jobs"]["quality"]["steps"]
    uses = [step["uses"] for step in steps if "uses" in step]

    assert uses == [
        "actions/checkout@v6",
        "actions/setup-python@v5",
    ]


def test_workflow_runs_compile_lint_format_and_tests(
    project_root: Path,
) -> None:
    """GitHub must run the same four quality gates used locally."""

    workflow = load_workflow(
        project_root / ".github" / "workflows" / "python-quality.yml"
    )
    steps = workflow["jobs"]["quality"]["steps"]
    commands = "\n".join(step["run"] for step in steps if "run" in step)

    assert "compileall src dashboard tests" in commands
    assert "ruff check src dashboard tests" in commands
    assert "ruff format --check tests" in commands
    assert "pytest -q" in commands


def test_local_runner_matches_github_quality_gates(
    project_root: Path,
) -> None:
    """The developer command should reproduce the GitHub checks."""

    runner = (project_root / "scripts" / "run_checkpoint50.ps1").read_text(
        encoding="utf-8"
    )

    assert "compileall src dashboard tests" in runner
    assert "ruff check src dashboard tests" in runner
    assert "ruff format --check tests" in runner
    assert "pytest -q" in runner


def test_quality_dependencies_are_declared(
    project_root: Path,
) -> None:
    """Fresh environments must install both testing and linting tools."""

    requirements = (
        (project_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    )

    assert any(line.startswith("pytest") for line in requirements)
    assert any(line.startswith("ruff") for line in requirements)
