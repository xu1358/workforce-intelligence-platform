"""Tests for the recruiter-facing README contract."""

from __future__ import annotations

from pathlib import Path

from validate_readme_portfolio import (
    load_contract,
    normalize_whitespace,
    unresolved_links,
    validate_readme,
)


def readme_contract(project_root: Path) -> dict:
    """Load the project README contract."""

    return load_contract(project_root / "config" / "readme_portfolio.yaml")


def test_readme_contains_required_sections_and_safeguards(
    project_root: Path,
) -> None:
    """The entry point must explain the decision and its use boundaries."""

    contract = readme_contract(project_root)
    readme = (project_root / contract["readme_path"]).read_text(encoding="utf-8")

    for heading in contract["required_headings"]:
        assert heading in readme

    normalized_readme = normalize_whitespace(readme.lower())

    for phrase in contract["required_phrases"]:
        assert normalize_whitespace(phrase.lower()) in normalized_readme


def test_readme_headline_metrics_follow_contract(
    project_root: Path,
) -> None:
    """Every committed headline value should remain visible."""

    contract = readme_contract(project_root)
    readme = (project_root / contract["readme_path"]).read_text(encoding="utf-8")

    for value in contract["headline_metrics"].values():
        assert str(value) in readme


def test_readme_does_not_restore_stale_version_1_claims(
    project_root: Path,
) -> None:
    """Obsolete threshold and calibration claims must stay removed."""

    contract = readme_contract(project_root)
    readme = (
        (project_root / contract["readme_path"]).read_text(encoding="utf-8").lower()
    )

    for claim in contract["forbidden_stale_claims"]:
        assert claim.lower() not in readme


def test_readme_local_links_resolve(
    project_root: Path,
) -> None:
    """Every local Markdown link should point to a committed file."""

    contract = readme_contract(project_root)
    readme = (project_root / contract["readme_path"]).read_text(encoding="utf-8")

    assert unresolved_links(readme, project_root) == []


def test_readme_code_fences_are_balanced(
    project_root: Path,
) -> None:
    """GitHub rendering should not be broken by an open code block."""

    contract = readme_contract(project_root)
    readme = (project_root / contract["readme_path"]).read_text(encoding="utf-8")
    fence_count = sum(line.strip().startswith("```") for line in readme.splitlines())

    assert fence_count > 0
    assert fence_count % 2 == 0


def test_complete_readme_validation_passes(
    project_root: Path,
) -> None:
    """The production validator should approve every README contract."""

    checks = validate_readme(project_root)

    assert checks
    assert {check["status"] for check in checks} == {"PASS"}
