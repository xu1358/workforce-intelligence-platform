"""Tests for the curated portfolio-notebook contract."""

from __future__ import annotations

import json
from pathlib import Path

from validate_portfolio_notebooks import (
    first_markdown_heading,
    load_manifest,
    markdown_text,
    validate_portfolio,
)


def test_manifest_defines_four_ordered_notebooks(
    project_root: Path,
) -> None:
    """The reviewer path should remain concise and explicitly ordered."""

    manifest = load_manifest(project_root / "config" / "portfolio_notebooks.yaml")
    entries = manifest["notebooks"]

    assert manifest["expected_portfolio_notebooks"] == 4
    assert [entry["order"] for entry in entries] == [1, 2, 3, 4]
    assert len({entry["file"] for entry in entries}) == 4
    assert len({entry["title"] for entry in entries}) == 4


def test_manifest_notebooks_exist_and_match_titles(
    project_root: Path,
) -> None:
    """Committed filenames, notebook headings, and manifest titles must agree."""

    manifest = load_manifest(project_root / "config" / "portfolio_notebooks.yaml")
    portfolio = project_root / manifest["portfolio_directory"]

    for entry in manifest["notebooks"]:
        path = portfolio / entry["file"]
        assert path.is_file()

        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert first_markdown_heading(notebook) == f"# {entry['title']}"


def test_portfolio_notebooks_are_executed_without_errors(
    project_root: Path,
) -> None:
    """Every code cell should have saved evidence and no failed output."""

    manifest = load_manifest(project_root / "config" / "portfolio_notebooks.yaml")
    portfolio = project_root / manifest["portfolio_directory"]

    for entry in manifest["notebooks"]:
        notebook = json.loads((portfolio / entry["file"]).read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

        assert cells
        assert all(cell["execution_count"] is not None for cell in cells)
        assert not [
            output
            for cell in cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]


def test_portfolio_notebooks_include_narrative_safeguards(
    project_root: Path,
) -> None:
    """Each notebook should disclose scope and guide the reviewer."""

    manifest = load_manifest(project_root / "config" / "portfolio_notebooks.yaml")
    portfolio = project_root / manifest["portfolio_directory"]

    for entry in manifest["notebooks"]:
        notebook = json.loads((portfolio / entry["file"]).read_text(encoding="utf-8"))
        markdown = markdown_text(notebook)

        assert "synthetic" in markdown.lower()
        assert "Portfolio navigation" in markdown

        for heading in entry["required_headings"]:
            assert heading in markdown


def test_notebook_index_links_the_complete_sequence(
    project_root: Path,
) -> None:
    """The notebook directory should provide one clear portfolio entry point."""

    manifest = load_manifest(project_root / "config" / "portfolio_notebooks.yaml")
    index = (project_root / manifest["index_file"]).read_text(encoding="utf-8")

    for entry in manifest["notebooks"]:
        assert f"portfolio/{entry['file']}" in index


def test_complete_portfolio_validation_passes(
    project_root: Path,
) -> None:
    """The production validator should approve every committed contract."""

    checks = validate_portfolio(project_root)

    assert checks
    assert {check["status"] for check in checks} == {"PASS"}
