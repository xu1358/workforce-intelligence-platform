"""Cross-platform command runner for the Workforce Intelligence Platform."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    """One named subprocess invocation."""

    name: str
    command: tuple[str, ...]


def python_script(name: str, relative_path: str) -> Step:
    """Create a step that runs one repository Python script."""

    return Step(
        name=name,
        command=(sys.executable, str(PROJECT_ROOT / relative_path)),
    )


QUALITY_STEPS = (
    python_script(
        "Validate locked dependency environment",
        "src/validate_dependency_environment.py",
    ),
    Step("Check dependency compatibility", (sys.executable, "-m", "pip", "check")),
    python_script(
        "Validate Markdown and Mermaid rendering",
        "src/validate_markdown_docs.py",
    ),
    python_script(
        "Validate cross-platform project runner",
        "src/validate_cross_platform_runner.py",
    ),
    Step(
        "Compile source, dashboard, and tests",
        (sys.executable, "-m", "compileall", "src", "dashboard", "tests"),
    ),
    Step(
        "Lint Python with Ruff",
        (
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src",
            "dashboard",
            "tests",
        ),
    ),
    Step(
        "Check automated-test formatting",
        (sys.executable, "-m", "ruff", "format", "--check", "tests"),
    ),
    Step("Run automated test suite", (sys.executable, "-m", "pytest", "-q")),
)

PORTFOLIO_VALIDATION_STEPS = (
    python_script(
        "Validate portfolio notebooks",
        "src/validate_portfolio_notebooks.py",
    ),
    python_script(
        "Validate recruiter-facing README",
        "src/validate_readme_portfolio.py",
    ),
)

DATA_GENERATION_STEPS = (
    python_script("Generate reference data", "src/generate_reference_data.py"),
    python_script("Generate full workforce", "src/generate_full_workforce.py"),
    python_script(
        "Generate compensation history",
        "src/generate_compensation_history.py",
    ),
    python_script(
        "Generate performance reviews",
        "src/generate_performance_reviews.py",
    ),
    python_script(
        "Generate training records",
        "src/generate_training_records.py",
    ),
    python_script("Generate employee events", "src/generate_employee_events.py"),
    python_script(
        "Apply attrition hazard and censor histories",
        "src/generate_attrition_outcomes.py",
    ),
    python_script(
        "Validate Version 2 attrition data",
        "src/validate_v2_attrition_data.py",
    ),
    python_script("Generate candidates", "src/generate_candidates.py"),
    python_script(
        "Generate job requisitions",
        "src/generate_job_requisitions.py",
    ),
    python_script("Generate applications", "src/generate_applications.py"),
)

VERSION_2_MODELING_STEPS = (
    python_script(
        "Build multi-snapshot temporal datasets",
        "src/build_multi_snapshot_retention_dataset.py",
    ),
    python_script(
        "Diagnose feature redundancy and stability",
        "src/diagnose_feature_redundancy.py",
    ),
    python_script(
        "Create temporal model splits",
        "src/create_model_splits.py",
    ),
    python_script(
        "Compare Version 2 retention models",
        "src/compare_retention_models_v2.py",
    ),
    python_script(
        "Calibrate selected retention model",
        "src/calibrate_retention_model.py",
    ),
    python_script(
        "Evaluate retention ranking",
        "src/evaluate_retention_ranking.py",
    ),
    python_script(
        "Analyze model fairness and subgroups",
        "src/analyze_model_fairness.py",
    ),
    python_script(
        "Calculate retention economics",
        "src/calculate_retention_economics.py",
    ),
    python_script(
        "Optimize and evaluate retention policy",
        "src/optimize_retention_policy.py",
    ),
)

POSTGRES_STEPS = (
    python_script("Test PostgreSQL connection", "src/test_database_connection.py"),
    python_script("Create PostgreSQL schema", "src/create_database_schema.py"),
    python_script(
        "Validate PostgreSQL schema",
        "src/validate_database_schema.py",
    ),
    python_script("Load CSV data into PostgreSQL", "src/load_postgresql_data.py"),
    python_script(
        "Validate PostgreSQL data load",
        "src/validate_database_load.py",
    ),
)

LEGACY_VERSION_1_STEPS = (
    python_script(
        "Build legacy retention modeling dataset",
        "src/build_retention_dataset.py",
    ),
    python_script(
        "Train legacy baseline retention model",
        "src/train_baseline_retention_model.py",
    ),
    python_script(
        "Compare legacy retention models",
        "src/compare_retention_models.py",
    ),
    python_script(
        "Analyze legacy selected retention model",
        "src/analyze_retention_model.py",
    ),
    python_script(
        "Build legacy dashboard data layer",
        "src/build_dashboard_data.py",
    ),
    python_script(
        "Validate legacy final project outputs",
        "src/validate_project_outputs.py",
    ),
)

CURRENT_DELIVERABLE_STEPS = (
    python_script(
        "Build current-state Version 2 dashboard data",
        "src/build_dashboard_current_state.py",
    ),
    python_script(
        "Analyze manufacturing workforce stability",
        "src/analyze_workforce_stability.py",
    ),
    python_script(
        "Explain retention model and test stability",
        "src/analyze_retention_explanations.py",
    ),
    python_script(
        "Analyze employee survival and time to exit",
        "src/analyze_employee_survival.py",
    ),
    python_script(
        "Audit retention policy allocation equity",
        "src/audit_retention_policy_equity.py",
    ),
    *PORTFOLIO_VALIDATION_STEPS,
)

EXTERNAL_BENCHMARK_STEPS = (
    python_script(
        "Download and verify IBM benchmark data",
        "src/download_ibm_attrition_benchmark.py",
    ),
    python_script(
        "Run isolated IBM attrition benchmark",
        "src/benchmark_ibm_attrition.py",
    ),
)


def display_command(command: Sequence[str]) -> str:
    """Return a readable command without shell-specific quoting."""

    return " ".join(str(part) for part in command)


def run_steps(steps: Sequence[Step], dry_run: bool = False) -> None:
    """Run steps in order and stop immediately on failure."""

    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("MPLBACKEND", "Agg")

    for position, step in enumerate(steps, start=1):
        print()
        print("=" * 70)
        print(f"[{position}/{len(steps)}] {step.name}")
        print("=" * 70)
        print(f"$ {display_command(step.command)}")

        if dry_run:
            continue

        result = subprocess.run(
            step.command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Step failed with exit code {result.returncode}: {step.name}"
            )


def build_pipeline_steps(args: argparse.Namespace) -> list[Step]:
    """Build the requested end-to-end pipeline without running it."""

    steps: list[Step] = []

    if not args.skip_quality:
        steps.extend(QUALITY_STEPS)
    if not args.skip_data_generation:
        steps.extend(DATA_GENERATION_STEPS)

    steps.extend(VERSION_2_MODELING_STEPS)

    if not args.skip_postgres:
        steps.extend(POSTGRES_STEPS)
    if args.include_legacy_v1:
        steps.extend(LEGACY_VERSION_1_STEPS)

    steps.extend(CURRENT_DELIVERABLE_STEPS)

    if not args.skip_external_benchmark:
        steps.extend(EXTERNAL_BENCHMARK_STEPS)

    return steps


def notebook_steps() -> list[Step]:
    """Build portable execution steps for reviewer-visible notebooks."""

    manifest_path = PROJECT_ROOT / "config" / "portfolio_notebooks.yaml"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    steps: list[Step] = []
    for notebook in manifest["executed_supporting_notebooks"]:
        notebook_path = (Path("notebooks") / notebook).as_posix()
        steps.append(
            Step(
                name=f"Execute {notebook_path}",
                command=(
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=900",
                    "--ExecutePreprocessor.kernel_name=python3",
                    notebook_path,
                ),
            )
        )

    steps.append(PORTFOLIO_VALIDATION_STEPS[0])
    return steps


def organize_checkpoint_runners(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Move root checkpoint journals into their historical subdirectory."""

    scripts_directory = project_root / "scripts"
    archive_directory = scripts_directory / "checkpoints"
    archive_directory.mkdir(parents=True, exist_ok=True)

    sources = sorted(scripts_directory.glob("run_checkpoint*.ps1"))
    helper = scripts_directory / "execute_supporting_notebooks.ps1"
    if helper.exists():
        sources.append(helper)

    moved: list[Path] = []
    for source in sources:
        destination = archive_directory / source.name
        if destination.exists():
            raise FileExistsError(
                f"Cannot organize {source}; destination already exists: "
                f"{destination}"
            )
        shutil.move(str(source), str(destination))
        moved.append(destination)

    return moved


def run_dashboard(dry_run: bool = False) -> None:
    """Launch the Streamlit dashboard through the active Python interpreter."""

    step = Step(
        name="Launch Streamlit dashboard",
        command=(
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "dashboard" / "app.py"),
        ),
    )
    run_steps([step], dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    """Create the cross-platform command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run quality checks, portfolio validation, the complete pipeline, "
            "notebooks, or the dashboard on Windows, macOS, and Linux."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved steps without executing them.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("quality", help="Run fast engineering quality gates.")
    subparsers.add_parser(
        "validate",
        help="Run quality gates plus notebook and README validation.",
    )

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Run the complete Version 2 project pipeline.",
    )
    pipeline.add_argument("--skip-quality", action="store_true")
    pipeline.add_argument("--skip-data-generation", action="store_true")
    pipeline.add_argument("--skip-postgres", action="store_true")
    pipeline.add_argument("--include-legacy-v1", action="store_true")
    pipeline.add_argument("--skip-external-benchmark", action="store_true")

    subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard.")
    subparsers.add_parser(
        "notebooks",
        help="Execute and validate reviewer-visible supporting notebooks.",
    )
    subparsers.add_parser(
        "organize-checkpoints",
        help="Move checkpoint PowerShell journals into scripts/checkpoints/.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.monotonic()

    print("Workforce Intelligence Platform")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Command: {args.command}")
    if args.dry_run:
        print("Mode: dry run")

    try:
        if args.command == "quality":
            run_steps(QUALITY_STEPS, dry_run=args.dry_run)
        elif args.command == "validate":
            run_steps(
                (*QUALITY_STEPS, *PORTFOLIO_VALIDATION_STEPS),
                dry_run=args.dry_run,
            )
        elif args.command == "pipeline":
            run_steps(build_pipeline_steps(args), dry_run=args.dry_run)
        elif args.command == "dashboard":
            run_dashboard(dry_run=args.dry_run)
        elif args.command == "notebooks":
            run_steps(notebook_steps(), dry_run=args.dry_run)
        elif args.command == "organize-checkpoints":
            if args.dry_run:
                candidates = sorted(
                    (PROJECT_ROOT / "scripts").glob("run_checkpoint*.ps1")
                )
                print(f"Checkpoint runners to move: {len(candidates)}")
            else:
                moved = organize_checkpoint_runners()
                print(f"Checkpoint/helper scripts moved: {len(moved)}")
                for path in moved:
                    print(f"- {path.relative_to(PROJECT_ROOT).as_posix()}")
        else:
            parser.error(f"Unsupported command: {args.command}")
    except (OSError, RuntimeError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print()
    print("=" * 70)
    print(f"{args.command.upper()} COMPLETED SUCCESSFULLY")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
