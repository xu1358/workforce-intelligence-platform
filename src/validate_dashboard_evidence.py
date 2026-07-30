"""Validate that reviewer-facing dashboard evidence matches Version 2."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import re
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "dashboard_evidence.yaml"
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")


def load_config(path: Path) -> dict[str, Any]:
    """Load the dashboard evidence contract."""

    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)

    if not isinstance(values, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping.")

    return values


def extract_streamlit_tabs(path: Path) -> list[str]:
    """Read the literal tab labels from the Streamlit application."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    candidates: list[list[str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "tabs" or not node.args:
            continue

        try:
            value = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue

        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            candidates.append(value)

    if len(candidates) != 1:
        raise ValueError(
            "Expected one literal st.tabs declaration; "
            f"found {len(candidates)}."
        )

    return candidates[0]


def reviewer_markdown_paths(project_root: Path) -> list[Path]:
    """Return Markdown surfaces that can expose dashboard image links."""

    paths = [project_root / "README.md"]
    paths.extend((project_root / "docs").rglob("*.md"))
    paths.append(project_root / "notebooks" / "README.md")
    return sorted(path for path in paths if path.is_file())


def dashboard_screenshot_references(project_root: Path) -> list[str]:
    """Find Markdown image links that target docs/screenshots."""

    references: list[str] = []

    for path in reviewer_markdown_paths(project_root):
        text = path.read_text(encoding="utf-8")
        for target in MARKDOWN_IMAGE_PATTERN.findall(text):
            normalized = target.replace("\\", "/")
            if "screenshots/" in normalized:
                references.append(
                    f"{path.relative_to(project_root).as_posix()} -> {target}"
                )

    return sorted(references)


def screenshot_files(directory: Path) -> list[Path]:
    """Return committed-style image files below the screenshot directory."""

    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def make_check(
    name: str,
    passed: bool,
    observed: Any,
    requirement: Any,
    details: str,
) -> dict[str, Any]:
    """Create one printable validation row."""

    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "requirement": requirement,
        "details": details,
    }


def build_checks(
    project_root: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build dashboard evidence validation checks."""

    dashboard_path = project_root / config["dashboard_application"]
    document_path = project_root / config["dashboard_documentation"]
    screenshot_directory = project_root / config["screenshot_directory"]
    legacy_paths = [
        project_root / relative_path
        for relative_path in config["legacy_v1_screenshots"]
    ]

    tabs = extract_streamlit_tabs(dashboard_path)
    images = screenshot_files(screenshot_directory)
    image_names = [
        path.relative_to(project_root).as_posix()
        for path in images
    ]
    surviving_legacy = [
        path.relative_to(project_root).as_posix()
        for path in legacy_paths
        if path.exists()
    ]
    duplicate_names = [
        path.name
        for path in images
        if any(
            marker in path.name
            for marker in config["forbidden_duplicate_filename_markers"]
        )
    ]
    references = dashboard_screenshot_references(project_root)
    document = document_path.read_text(encoding="utf-8")
    stale_values = [
        value
        for value in config["forbidden_current_dashboard_values"]
        if value in document
    ]
    missing_current_values = [
        value
        for value in config["required_current_values"]
        if value not in document
    ]
    missing_tab_sections = [
        tab for tab in tabs if f"### {tab}" not in document
    ]
    workflow = (
        project_root / config["ci_workflow"]
    ).read_text(encoding="utf-8")
    policy = config["evidence_policy"]
    governance = config["governance"]
    governance_ok = (
        not governance["changes_dashboard_application"]
        and not governance["changes_dashboard_data"]
        and not governance["changes_model_results"]
        and not governance["changes_frozen_policy"]
        and governance["removes_legacy_visual_evidence_only"]
    )

    return [
        make_check(
            "Live dashboard exposes the six Version 2 tabs",
            tabs == config["expected_tabs"],
            tabs,
            config["expected_tabs"],
            "Workforce Stability is part of the reviewer-visible interface.",
        ),
        make_check(
            "Legacy Version 1 screenshots are absent",
            not surviving_legacy,
            surviving_legacy,
            [],
            "The current branch no longer carries contradictory V1 captures.",
        ),
        make_check(
            "Screenshot directory contains no orphan images",
            not images,
            image_names,
            [],
            "Static screenshots are omitted until an automated refresh contract exists.",
        ),
        make_check(
            "Duplicate-download filenames are absent",
            not duplicate_names,
            duplicate_names,
            [],
            "Names containing markers such as (1) cannot return.",
        ),
        make_check(
            "Reviewer Markdown contains no screenshot links",
            not references,
            references,
            [],
            "No document links to removed or unverified captures.",
        ),
        make_check(
            "Dashboard guide documents every live tab",
            not missing_tab_sections,
            missing_tab_sections,
            [],
            "The text guide covers the exact application navigation.",
        ),
        make_check(
            "Dashboard guide uses current Version 2 figures",
            not missing_current_values and not stale_values,
            {
                "missing_current_values": missing_current_values,
                "stale_values": stale_values,
            },
            {
                "required": config["required_current_values"],
                "forbidden": config["forbidden_current_dashboard_values"],
            },
            "Current evidence is consistent with the Version 2 README.",
        ),
        make_check(
            "Static evidence policy is explicit",
            (
                not policy["committed_static_screenshots"]
                and policy["live_application_is_authoritative"]
                and policy["version_1_preserved_by_tag"] == "v1.0-portfolio"
                and "Static screenshots are intentionally not committed" in document
            ),
            policy,
            "Live V2 app authoritative; V1 available only through its tag",
            "Removing screenshots is an intentional freshness control.",
        ),
        make_check(
            "GitHub Actions validates dashboard evidence",
            "python src/validate_dashboard_evidence.py" in workflow,
            config["ci_workflow"],
            "python src/validate_dashboard_evidence.py",
            "A pull request cannot silently restore the stale gallery.",
        ),
        make_check(
            "Checkpoint changes visual evidence only",
            governance_ok,
            governance,
            "No application, data, model, or policy changes",
            "Checkpoint 67 removes stale presentation artifacts without recalculation.",
        ),
    ]


def print_checks(checks: list[dict[str, Any]]) -> None:
    """Print a compact validation table."""

    print("\nDASHBOARD EVIDENCE VALIDATION")
    for check in checks:
        print(
            f"{check['status']:<4}  "
            f"{check['check']:<52}  "
            f"{check['observed']}"
        )


def save_outputs(
    project_root: Path,
    config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> Path:
    """Save aggregate dashboard evidence validation outputs."""

    output_directory = project_root / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "dashboard_evidence_validation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "checks": len(checks),
        "passed": sum(check["status"] == "PASS" for check in checks),
        "failed": sum(check["status"] == "FAIL" for check in checks),
        "committed_static_screenshots": False,
        "authoritative_dashboard": config["dashboard_application"],
        "version_1_tag": config["evidence_policy"]["version_1_preserved_by_tag"],
    }
    (output_directory / "dashboard_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_directory


def main() -> None:
    """Run the committed dashboard evidence contract."""

    config = load_config(CONFIG_PATH)
    checks = build_checks(PROJECT_ROOT, config)
    print_checks(checks)
    output_directory = save_outputs(PROJECT_ROOT, config, checks)

    failed = [check for check in checks if check["status"] == "FAIL"]
    if failed:
        names = ", ".join(check["check"] for check in failed)
        raise ValueError(f"Dashboard evidence validation failed: {names}")

    print(f"\nSaved dashboard evidence validation to: {output_directory}")
    print("\nDASHBOARD EVIDENCE VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
