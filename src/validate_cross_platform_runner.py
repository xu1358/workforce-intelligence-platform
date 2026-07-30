"""Validate the cross-platform project runner and archived script layout."""

from __future__ import annotations

from argparse import Namespace
import csv
import json
from pathlib import Path
import platform
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_project import (  # noqa: E402
    V2_POSTGRES_DATASET_STEP,
    V2_POSTGRES_VALIDATION_STEP,
    V2_SQL_EQUIVALENCE_STEP,
    VERSION_2_ANALYSIS_STEPS,
    build_parser,
    build_pipeline_steps,
    notebook_steps,
)


def load_config(path: Path) -> dict[str, Any]:
    """Load the committed cross-platform runner contract."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def command_names() -> set[str]:
    """Return subcommands exposed by the canonical parser."""

    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    )
    return set(subparser_action.choices)


def default_pipeline_steps() -> list:
    """Return the complete default Version 2 pipeline."""

    args = Namespace(
        skip_quality=False,
        skip_data_generation=False,
        skip_postgres=False,
        include_legacy_v1=False,
        skip_external_benchmark=False,
    )
    return build_pipeline_steps(args)


def stale_checkpoint_references(project_root: Path) -> list[str]:
    """Find text contracts that still point to the scripts root."""

    roots = [
        project_root / "README.md",
        project_root / "config",
        project_root / "docs",
        project_root / "notebooks" / "README.md",
        project_root / "tests",
    ]
    forward_reference = "scripts/" + "run_checkpoint"
    backward_reference = "scripts\\\\" + "run_checkpoint"
    stale: set[str] = set()

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
            if forward_reference in text or backward_reference in text:
                stale.add(path.relative_to(project_root).as_posix())

    return sorted(stale)


def build_checks(
    project_root: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the committed cross-platform execution checks."""

    runner = project_root / config["canonical_runner"]
    wrapper = project_root / config["windows_wrapper"]
    archive = project_root / config["checkpoint_archive"]
    scripts_directory = project_root / "scripts"
    commands = command_names()
    pipeline = default_pipeline_steps()
    all_command_parts = [part for step in pipeline for part in step.command]
    shell_tokens = {
        "&&",
        "||",
        ";",
        "powershell",
        "pwsh",
        "bash",
        "sh",
    }
    shell_specific = [part for part in all_command_parts if part in shell_tokens]
    root_files = sorted(
        path.name for path in scripts_directory.iterdir() if path.is_file()
    )
    checkpoint_files = sorted(archive.glob("run_checkpoint*.ps1"))
    archived_powershell = [
        *checkpoint_files,
        archive / "execute_supporting_notebooks.ps1",
    ]
    root_resolution_failures = []
    for path in archived_powershell:
        content = path.read_text(encoding="utf-8")
        if (
            "$ScriptsDirectory = Split-Path -Parent $PSScriptRoot" not in content
            or "$ProjectRoot = Split-Path -Parent $ScriptsDirectory" not in content
            or "$ProjectRoot = Split-Path -Parent $PSScriptRoot" in content
        ):
            root_resolution_failures.append(path.name)
    wrapper_text = wrapper.read_text(encoding="utf-8")
    workflow_text = (
        project_root / ".github" / "workflows" / "python-quality.yml"
    ).read_text(encoding="utf-8")
    stale_references = stale_checkpoint_references(project_root)
    sql_stage_order = [
        pipeline.index(V2_POSTGRES_DATASET_STEP),
        pipeline.index(V2_POSTGRES_VALIDATION_STEP),
        pipeline.index(V2_SQL_EQUIVALENCE_STEP),
        pipeline.index(VERSION_2_ANALYSIS_STEPS[0]),
    ]
    governance = config["governance"]
    governance_ok = (
        governance["orchestration_only"]
        and not governance["changes_analytical_scripts"]
        and not governance["changes_model_results"]
        and not governance["changes_frozen_policy"]
        and not governance["changes_dashboard_data"]
    )

    return [
        {
            "check": "Canonical Python runner exists",
            "status": "PASS" if runner.is_file() else "FAIL",
            "observed": config["canonical_runner"],
            "requirement": "One reviewer-facing Python entry point",
            "details": "The runner is independent of PowerShell and Bash.",
        },
        {
            "check": "Committed command surface is complete",
            "status": ("PASS" if commands == set(config["commands"]) else "FAIL"),
            "observed": sorted(commands),
            "requirement": sorted(config["commands"]),
            "details": "Quality, validation, pipeline, UI, and notebook paths are explicit.",
        },
        {
            "check": "Default pipeline resolves every committed stage",
            "status": "PASS" if len(pipeline) >= 40 else "FAIL",
            "observed": len(pipeline),
            "requirement": ">= 40 ordered stages",
            "details": "The dry run preserves complete Version 2 orchestration.",
        },
        {
            "check": "SQL equivalence gates downstream modeling",
            "status": (
                "PASS"
                if sql_stage_order == sorted(sql_stage_order)
                and len(set(sql_stage_order)) == len(sql_stage_order)
                else "FAIL"
            ),
            "observed": [position + 1 for position in sql_stage_order],
            "requirement": (
                "PostgreSQL builder -> source parity -> SQL equivalence "
                "-> model analysis"
            ),
            "details": (
                "The independent SQL claim is verified before any downstream "
                "model stage."
            ),
        },
        {
            "check": "Commands avoid shell-specific execution syntax",
            "status": "PASS" if not shell_specific else "FAIL",
            "observed": shell_specific,
            "requirement": "No shell operators or shell executables",
            "details": "Subprocess argument lists work across operating systems.",
        },
        {
            "check": "Supporting notebook command matches manifest",
            "status": (
                "PASS"
                if len(notebook_steps()) - 1 == config["expected_supporting_notebooks"]
                else "FAIL"
            ),
            "observed": len(notebook_steps()) - 1,
            "requirement": config["expected_supporting_notebooks"],
            "details": "Notebook paths are resolved under notebooks/ portably.",
        },
        {
            "check": "Checkpoint history is archived",
            "status": (
                "PASS"
                if len(checkpoint_files)
                == config["checkpoint_history"]["expected_count"]
                else "FAIL"
            ),
            "observed": len(checkpoint_files),
            "requirement": config["checkpoint_history"]["expected_count"],
            "details": "Development journals remain available but leave the scripts root.",
        },
        {
            "check": "Archived PowerShell runners resolve repository root",
            "status": ("PASS" if not root_resolution_failures else "FAIL"),
            "observed": root_resolution_failures,
            "requirement": "0 one-level root calculations",
            "details": "Archived scripts remain runnable after moving one level deeper.",
        },
        {
            "check": "Scripts root exposes only the finished interface",
            "status": (
                "PASS" if root_files == sorted(config["root_script_files"]) else "FAIL"
            ),
            "observed": root_files,
            "requirement": sorted(config["root_script_files"]),
            "details": "Reviewers see the runner, guide, and compatibility wrapper.",
        },
        {
            "check": "Windows wrapper delegates to Python",
            "status": (
                "PASS"
                if "run_project.py" in wrapper_text
                and '"pipeline"' in wrapper_text
                and "generate_reference_data.py" not in wrapper_text
                else "FAIL"
            ),
            "observed": "Delegates" if "run_project.py" in wrapper_text else "Missing",
            "requirement": "No duplicate pipeline definition",
            "details": "Windows and cross-platform orchestration cannot drift.",
        },
        {
            "check": "GitHub Actions resolves the portable pipeline",
            "status": (
                "PASS"
                if "python scripts/run_project.py --dry-run pipeline" in workflow_text
                else "FAIL"
            ),
            "observed": platform.system(),
            "requirement": "Ubuntu CI dry run",
            "details": "The command surface is exercised outside Windows.",
        },
        {
            "check": "No stale root checkpoint references remain",
            "status": "PASS" if not stale_references else "FAIL",
            "observed": stale_references,
            "requirement": "0 stale references",
            "details": "Documentation and tests use the archive or canonical runner.",
        },
        {
            "check": "Checkpoint is orchestration-only",
            "status": "PASS" if governance_ok else "FAIL",
            "observed": governance,
            "requirement": "No analytical or operational result changes",
            "details": "Models, data, policy, and dashboard outputs remain frozen.",
        },
    ]


def save_outputs(
    project_root: Path,
    config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> Path:
    """Save aggregate runner-validation evidence."""

    output_directory = project_root / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    with (output_directory / "cross_platform_validation.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "operating_system": platform.system(),
        "python_executable": sys.executable,
        "commands": sorted(command_names()),
        "default_pipeline_steps": len(default_pipeline_steps()),
        "checks_passed": sum(check["status"] == "PASS" for check in checks),
        "checks_total": len(checks),
    }
    (output_directory / "cross_platform_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_directory


def print_results(
    checks: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """Print reviewer-readable runner validation."""

    print("\nCROSS-PLATFORM RUNNER SUMMARY")
    print(f"Operating system: {platform.system()}")
    print(f"Python executable: {sys.executable}")
    print(f"Available commands: {len(command_names())}")
    print(f"Default pipeline stages: {len(default_pipeline_steps())}")

    print("\nCROSS-PLATFORM RUNNER VALIDATION")
    for check in checks:
        print(f"{check['status']:<4}  {check['check']:<54}  {check['observed']}")

    print(f"\nSaved runner validation outputs to: {output_directory}")


def main() -> None:
    """Run the cross-platform execution contract."""

    config = load_config(PROJECT_ROOT / "config" / "cross_platform_runner.yaml")
    checks = build_checks(PROJECT_ROOT, config)
    output_directory = save_outputs(PROJECT_ROOT, config, checks)
    print_results(checks, output_directory)

    failed = [check["check"] for check in checks if check["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Cross-platform runner validation failed: {failed}")

    print("\nCROSS-PLATFORM PROJECT RUNNER VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
