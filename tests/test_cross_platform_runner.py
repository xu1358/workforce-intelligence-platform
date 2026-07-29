"""Contracts for the reviewer-facing cross-platform project runner."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

from scripts.run_project import (
    CURRENT_DELIVERABLE_STEPS,
    DATA_GENERATION_STEPS,
    EXTERNAL_BENCHMARK_STEPS,
    POSTGRES_STEPS,
    QUALITY_STEPS,
    VERSION_2_MODELING_STEPS,
    build_parser,
    build_pipeline_steps,
    notebook_steps,
    organize_checkpoint_runners,
)
from validate_cross_platform_runner import build_checks


def load_contract(project_root: Path) -> dict:
    """Load the cross-platform execution contract."""

    with (project_root / "config" / "cross_platform_runner.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_canonical_runner_exposes_committed_commands(project_root: Path) -> None:
    """The CLI must offer every documented reviewer command."""

    contract = load_contract(project_root)
    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    )

    assert set(subparser_action.choices) == set(contract["commands"])


def test_default_pipeline_contains_every_version_2_stage() -> None:
    """Default execution must preserve the complete Version 2 pipeline."""

    args = Namespace(
        skip_quality=False,
        skip_data_generation=False,
        skip_postgres=False,
        include_legacy_v1=False,
        skip_external_benchmark=False,
    )
    steps = build_pipeline_steps(args)

    expected = (
        *QUALITY_STEPS,
        *DATA_GENERATION_STEPS,
        *VERSION_2_MODELING_STEPS,
        *POSTGRES_STEPS,
        *CURRENT_DELIVERABLE_STEPS,
        *EXTERNAL_BENCHMARK_STEPS,
    )

    assert steps == list(expected)


def test_pipeline_options_remove_only_requested_stage_groups() -> None:
    """Portable skip flags must have narrow and predictable effects."""

    args = Namespace(
        skip_quality=True,
        skip_data_generation=True,
        skip_postgres=True,
        include_legacy_v1=False,
        skip_external_benchmark=True,
    )
    steps = build_pipeline_steps(args)

    assert steps == [
        *VERSION_2_MODELING_STEPS,
        *CURRENT_DELIVERABLE_STEPS,
    ]


def test_runner_commands_use_active_python_without_shell_syntax() -> None:
    """Subprocess commands must remain portable argument sequences."""

    all_steps = (
        *QUALITY_STEPS,
        *DATA_GENERATION_STEPS,
        *VERSION_2_MODELING_STEPS,
        *POSTGRES_STEPS,
        *CURRENT_DELIVERABLE_STEPS,
        *EXTERNAL_BENCHMARK_STEPS,
    )

    assert all(isinstance(step.command, tuple) for step in all_steps)
    assert not [
        part
        for step in all_steps
        for part in step.command
        if part in {"&&", "||", ";", "powershell", "pwsh", "bash", "sh"}
    ]


def test_notebook_execution_comes_from_committed_manifest() -> None:
    """The portable notebook command must execute every support file."""

    steps = notebook_steps()

    assert len(steps) == 13
    assert sum("nbconvert" in step.command for step in steps) == 12
    assert all(step.command[-1].startswith("notebooks/") for step in steps[:-1])
    assert steps[-1].command[-1].endswith("validate_portfolio_notebooks.py")


def test_checkpoint_journal_is_outside_scripts_root(project_root: Path) -> None:
    """Reviewer-facing scripts root must not contain checkpoint journals."""

    contract = load_contract(project_root)
    scripts_directory = project_root / "scripts"
    archive = project_root / contract["checkpoint_archive"]
    history = contract["checkpoint_history"]

    assert not list(scripts_directory.glob("run_checkpoint*.ps1"))
    assert len(list(archive.glob("run_checkpoint*.ps1"))) == history["expected_count"]
    assert sorted(
        path.name for path in scripts_directory.iterdir() if path.is_file()
    ) == sorted(contract["root_script_files"])


def test_checkpoint_organizer_moves_without_deleting(tmp_path: Path) -> None:
    """The one-time migration must preserve checkpoint file contents."""

    scripts_directory = tmp_path / "scripts"
    scripts_directory.mkdir()
    first = scripts_directory / "run_checkpoint36.ps1"
    second = scripts_directory / "run_checkpoint37.ps1"
    helper = scripts_directory / "execute_supporting_notebooks.ps1"
    first.write_text("checkpoint 36\n", encoding="utf-8")
    second.write_text("checkpoint 37\n", encoding="utf-8")
    helper.write_text("notebooks\n", encoding="utf-8")

    moved = organize_checkpoint_runners(tmp_path)

    assert len(moved) == 3
    assert not first.exists()
    assert not second.exists()
    assert not helper.exists()
    assert (scripts_directory / "checkpoints" / "run_checkpoint36.ps1").read_text(
        encoding="utf-8"
    ) == "checkpoint 36\n"
    assert (scripts_directory / "checkpoints" / "run_checkpoint37.ps1").read_text(
        encoding="utf-8"
    ) == "checkpoint 37\n"
    assert (scripts_directory / "checkpoints" / helper.name).read_text(
        encoding="utf-8"
    ) == "notebooks\n"


def test_windows_wrapper_delegates_to_canonical_runner(project_root: Path) -> None:
    """PowerShell compatibility must not duplicate pipeline orchestration."""

    wrapper = (project_root / "scripts" / "run_end_to_end.ps1").read_text(
        encoding="utf-8"
    )

    assert "run_project.py" in wrapper
    assert '"pipeline"' in wrapper
    assert "generate_reference_data.py" not in wrapper


def test_all_documented_checkpoint_paths_use_archive(project_root: Path) -> None:
    """Text contracts must not point reviewers back to the scripts root."""

    roots = [
        project_root / "README.md",
        project_root / "config",
        project_root / "docs",
        project_root / "notebooks" / "README.md",
        project_root / "tests",
    ]
    stale: list[str] = []
    forward_reference = "scripts/" + "run_checkpoint"
    backward_reference = "scripts\\\\" + "run_checkpoint"

    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix not in {
                ".md",
                ".py",
                ".yaml",
                ".yml",
            }:
                continue
            text = path.read_text(encoding="utf-8")
            if forward_reference in text:
                stale.append(path.relative_to(project_root).as_posix())
            if backward_reference in text:
                stale.append(path.relative_to(project_root).as_posix())

    assert not stale


def test_reviewer_facing_runner_validation_passes(project_root: Path) -> None:
    """The aggregate cross-platform contract must remain green."""

    checks = build_checks(project_root, load_contract(project_root))

    assert not [check["check"] for check in checks if check["status"] != "PASS"]
