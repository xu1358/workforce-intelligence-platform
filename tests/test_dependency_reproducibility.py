"""Tests for the committed Python dependency environment."""

from __future__ import annotations

from pathlib import Path

import yaml

from validate_dependency_environment import (
    exact_version,
    parse_requirements,
    sha256_file,
)


def load_contract(project_root: Path) -> dict:
    """Load the dependency reproducibility contract."""

    with (project_root / "config" / "dependency_reproducibility.yaml").open(
        encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)


def test_dependency_contract_pins_python_and_pip(project_root: Path) -> None:
    """The interpreter and resolver boundaries must be explicit."""

    contract = load_contract(project_root)

    assert contract["python"] == {
        "supported_major_minor": "3.12",
        "ci_version": "3.12.4",
    }
    assert contract["bootstrap"]["pip_version"] == "26.0.1"
    assert contract["bootstrap"]["setuptools_version"] == "83.0.0"
    assert contract["bootstrap"]["wheel_version"] == "0.47.0"


def test_every_direct_dependency_is_exactly_pinned(project_root: Path) -> None:
    """requirements.txt must not contain bare names or version ranges."""

    contract = load_contract(project_root)
    direct = parse_requirements(project_root / contract["files"]["direct_requirements"])

    assert len(direct) == contract["expected_counts"]["direct_dependencies"]
    assert not [
        item.requirement.name
        for item in direct.values()
        if exact_version(item.requirement) is None
    ]


def test_core_versions_match_reviewed_compatibility_set(project_root: Path) -> None:
    """The key data-science stack must match its tested versions."""

    contract = load_contract(project_root)
    direct = parse_requirements(project_root / contract["files"]["direct_requirements"])

    observed = {
        item.requirement.name: exact_version(item.requirement)
        for item in direct.values()
        if item.requirement.name in contract["core_dependencies"]
    }

    assert observed == contract["core_dependencies"]
    assert observed["pandas"] == "2.2.3"
    assert observed["scikit-learn"] == "1.8.0"


def test_transitive_lock_is_exact_and_hashed(project_root: Path) -> None:
    """Every locked distribution must have a version and artifact hash."""

    contract = load_contract(project_root)
    locked = parse_requirements(project_root / contract["files"]["lock_file"])

    assert len(locked) == contract["expected_counts"]["locked_distributions"]
    assert all(exact_version(item.requirement) for item in locked.values())
    assert all(item.hash_count > 0 for item in locked.values())


def test_direct_dependencies_reconcile_to_lock(project_root: Path) -> None:
    """The generated lock must contain every reviewed direct pin."""

    contract = load_contract(project_root)
    direct = parse_requirements(project_root / contract["files"]["direct_requirements"])
    locked = parse_requirements(project_root / contract["files"]["lock_file"])

    assert {name: exact_version(item.requirement) for name, item in direct.items()} == {
        name: exact_version(locked[name].requirement) for name in direct
    }


def test_requirement_fingerprints_match_contract(project_root: Path) -> None:
    """Committed dependency files must match their SHA-256 fingerprints."""

    contract = load_contract(project_root)
    direct_path = project_root / contract["files"]["direct_requirements"]
    lock_path = project_root / contract["files"]["lock_file"]

    assert sha256_file(direct_path) == contract["files"]["direct_requirements_sha256"]
    assert sha256_file(lock_path) == contract["files"]["lock_file_sha256"]


def test_ci_installs_and_validates_hashed_lock(project_root: Path) -> None:
    """GitHub Actions must use the same immutable environment contract."""

    contract = load_contract(project_root)
    workflow = (
        project_root / ".github" / "workflows" / "python-quality.yml"
    ).read_text(encoding="utf-8")

    assert f'python-version: "{contract["python"]["ci_version"]}"' in workflow
    assert (
        f'python -m pip install "pip=={contract["bootstrap"]["pip_version"]}" '
        f'"setuptools=={contract["bootstrap"]["setuptools_version"]}" '
        f'"wheel=={contract["bootstrap"]["wheel_version"]}"' in workflow
    )
    assert (
        "python -m pip install --no-build-isolation "
        "--require-hashes -r requirements-lock.txt" in workflow
    )
    assert "python -m pip check" in workflow
    assert "python src/validate_dependency_environment.py" in workflow


def test_checkpoint57_runner_enforces_environment(project_root: Path) -> None:
    """The local checkpoint must perform the same dependency checks as CI."""

    runner = (
        project_root / "scripts" / "checkpoints" / "run_checkpoint57.ps1"
    ).read_text(encoding="utf-8")

    assert "validate_dependency_environment.py" in runner
    assert "-m pip check" in runner
    assert "ruff check src dashboard tests" in runner
    assert "pytest -q" in runner


def test_end_to_end_pipeline_fails_fast_on_dependency_drift(
    project_root: Path,
) -> None:
    """The full pipeline should validate dependencies before expensive work."""

    runner = (project_root / "scripts" / "run_project.py").read_text(encoding="utf-8")

    dependency_position = runner.index("validate_dependency_environment.py")
    generation_position = runner.index("generate_reference_data.py")

    assert dependency_position < generation_position


def test_dependency_update_process_is_documented(project_root: Path) -> None:
    """Reviewers and maintainers need an intentional update procedure."""

    documentation = (
        (project_root / "docs" / "dependency_reproducibility.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    for phrase in [
        "requirements-lock.txt",
        "--require-hashes",
        "sha-256",
        "full automated test suite",
        "dependency update",
        "pandas 2.2.3",
        "scikit-learn 1.8.0",
    ]:
        assert phrase in documentation
