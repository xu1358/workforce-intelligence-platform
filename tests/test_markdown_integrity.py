"""Regression tests for reviewer-facing Markdown rendering."""

from __future__ import annotations

from pathlib import Path

import yaml

from validate_markdown_docs import (
    audit_documents,
    closing_fence_matches,
    parse_fenced_blocks,
    validate_mermaid_contract,
)


REGRESSION_DOCUMENTS = {
    "docs/attrition_hazard_design.md",
    "docs/data_model.md",
    "docs/department_history_bias.md",
    "docs/dashboard.md",
    "docs/end_to_end_validation.md",
    "docs/exploratory_analysis.md",
    "docs/generator_audit.md",
    "docs/portfolio_presentation.md",
}


def load_config(project_root: Path) -> dict:
    """Load the Markdown rendering contract."""

    with (project_root / "config" / "markdown_integrity.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_reported_regression_documents_remain_in_contract(
    project_root: Path,
) -> None:
    """Every document named in the review issue must stay protected."""

    config = load_config(project_root)

    assert set(config["regression_documents"]) == REGRESSION_DOCUMENTS


def test_every_reviewer_visible_markdown_fence_is_closed(
    project_root: Path,
) -> None:
    """No Markdown document may end inside a fenced block."""

    config = load_config(project_root)
    audits, _ = audit_documents(project_root, config)

    assert len(audits) >= config["minimum_markdown_files"]
    assert not [audit.file for audit in audits if audit.unclosed_fences > 0]


def test_data_model_mermaid_er_diagram_is_closed_and_complete(
    project_root: Path,
) -> None:
    """The main ER diagram must remain renderable on GitHub."""

    config = load_config(project_root)
    _, blocks_by_file = audit_documents(project_root, config)

    valid, details = validate_mermaid_contract(
        blocks_by_file,
        config["mermaid_contract"],
    )

    assert valid, details


def test_parser_reports_an_unclosed_fence() -> None:
    """A missing closing delimiter must fail deterministically."""

    blocks, issues = parse_fenced_blocks("# Example\n\n```python\nprint('open')\n")

    assert blocks == []
    assert len(issues) == 1
    assert issues[0].opening_line == 3
    assert issues[0].language == "python"


def test_parser_accepts_backtick_and_tilde_fences() -> None:
    """Both CommonMark fence marker styles must be supported."""

    text = "```text\nalpha\n```\n\n~~~powershell\nWrite-Host ok\n~~~~\n"
    blocks, issues = parse_fenced_blocks(text)

    assert not issues
    assert [block.language for block in blocks] == ["text", "powershell"]
    assert [block.content for block in blocks] == ["alpha", "Write-Host ok"]


def test_shorter_or_wrong_marker_does_not_close_active_fence() -> None:
    """Only a sufficiently long matching marker may close a block."""

    assert closing_fence_matches("````", "```")
    assert not closing_fence_matches("``", "```")
    assert not closing_fence_matches("~~~", "```")


def test_github_workflow_runs_markdown_validator(project_root: Path) -> None:
    """Remote CI must prevent a recurrence before merging."""

    workflow = (
        project_root / ".github" / "workflows" / "python-quality.yml"
    ).read_text(encoding="utf-8")

    assert "python src/validate_markdown_docs.py" in workflow


def test_checkpoint_runner_reproduces_markdown_gate(project_root: Path) -> None:
    """The local checkpoint must execute the same validator."""

    runner = (
        project_root / "scripts" / "checkpoints" / "run_checkpoint58.ps1"
    ).read_text(encoding="utf-8")

    assert "src\\validate_markdown_docs.py" in runner
    assert "pytest -q" in runner
