"""Tests for the curated portfolio-notebook contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from validate_portfolio_notebooks import (
    code_cells,
    first_markdown_heading,
    load_manifest,
    markdown_text,
    notebook_text,
    validate_portfolio,
)

EXPECTED_EXECUTED_SUPPORTING_NOTEBOOKS = [
    "20_v2_data_validation.ipynb",
    "21_temporal_dataset_validation.ipynb",
    "22_feature_diagnostics.ipynb",
    "23_model_comparison_v2.ipynb",
    "24_calibration_analysis.ipynb",
    "25_ranking_analysis.ipynb",
    "26_fairness_analysis.ipynb",
    "27_retention_cost_model.ipynb",
    "28_retention_policy_analysis.ipynb",
    "29_dashboard_current_state_validation.ipynb",
    "30_manufacturing_workforce_stability.ipynb",
    "34_retention_policy_equity.ipynb",
    "35_deployed_policy_fairness.ipynb",
    "36_temporal_prior_shift.ipynb",
    "37_v2_exploratory_analysis.ipynb",
    "38_v2_department_history.ipynb",
]


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


def test_version_2_supporting_notebooks_save_reviewable_outputs(
    project_root: Path,
) -> None:
    """Version 2 technical notebooks must not return to blank templates."""

    manifest = load_manifest(project_root / "config" / "portfolio_notebooks.yaml")
    configured = manifest["executed_supporting_notebooks"]
    minimum_outputs = manifest["minimum_supporting_code_outputs_per_notebook"]

    assert configured == EXPECTED_EXECUTED_SUPPORTING_NOTEBOOKS

    for filename in configured:
        path = project_root / "notebooks" / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = code_cells(notebook)
        outputs = [output for cell in cells for output in cell.get("outputs", [])]

        assert cells
        assert all(cell["execution_count"] is not None for cell in cells)
        assert len(outputs) >= minimum_outputs
        assert not [
            output for output in outputs if output.get("output_type") == "error"
        ]
        assert not {
            marker
            for marker in ["c:\\users\\", "/workspace/", "/home/"]
            if marker in notebook_text(notebook).lower()
        }
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        assert path.stat().st_size <= manifest["maximum_notebook_size_bytes"]


def test_validator_rejects_blank_version_2_supporting_notebook(
    project_root: Path,
    tmp_path: Path,
) -> None:
    """A cleared supporting notebook must fail the production validator."""

    temporary_root = tmp_path / "project"
    shutil.copytree(project_root / "notebooks", temporary_root / "notebooks")
    manifest = load_manifest(project_root / "config" / "portfolio_notebooks.yaml")
    blank_path = (
        temporary_root / "notebooks" / manifest["executed_supporting_notebooks"][0]
    )
    notebook = json.loads(blank_path.read_text(encoding="utf-8"))

    for cell in code_cells(notebook):
        cell["execution_count"] = None
        cell["outputs"] = []

    blank_path.write_text(
        json.dumps(notebook, indent=1) + "\n",
        encoding="utf-8",
    )
    checks = validate_portfolio(temporary_root, manifest)
    execution_check = next(
        check
        for check in checks
        if check["check"] == "Reviewer-visible supporting code cells are executed"
    )
    output_check = next(
        check
        for check in checks
        if check["check"] == "Reviewer-visible supporting outputs are embedded"
    )

    assert execution_check["status"] == "FAIL"
    assert output_check["status"] == "FAIL"
    assert blank_path.name in execution_check["observed"]
    assert blank_path.name in output_check["observed"]


def test_complete_portfolio_validation_passes(
    project_root: Path,
) -> None:
    """The production validator should approve every committed contract."""

    checks = validate_portfolio(project_root)

    assert checks
    assert {check["status"] for check in checks} == {"PASS"}
