"""Validate the recruiter-facing README and its evidence contract."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "readme_portfolio.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "readme_portfolio_validation.csv"

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def normalize_whitespace(text: str) -> str:
    """Collapse Markdown line wrapping for phrase comparisons."""

    unquoted_lines = [
        line.lstrip().removeprefix("> ").strip() for line in text.splitlines()
    ]

    return " ".join("\n".join(unquoted_lines).split())


def load_contract(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the committed README validation contract."""

    if not path.exists():
        raise FileNotFoundError(f"Missing README contract: {path}")

    with path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)

    if not isinstance(contract, dict):
        raise ValueError("README contract must contain a YAML mapping.")

    return contract


def local_markdown_links(markdown: str) -> list[str]:
    """Return local file targets from Markdown links and images."""

    local: list[str] = []

    for raw_target in MARKDOWN_LINK.findall(markdown):
        target = raw_target.strip().split(maxsplit=1)[0]
        target = target.split("#", maxsplit=1)[0]

        if not target:
            continue

        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue

        local.append(target.replace("\\", "/"))

    return local


def unresolved_links(
    markdown: str,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    """Return local Markdown targets that do not resolve to files."""

    return sorted(
        {
            target
            for target in local_markdown_links(markdown)
            if not (project_root / target).is_file()
        }
    )


def append_check(
    checks: list[dict[str, str]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    """Append one normalized README validation result."""

    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": str(observed),
            "requirement": requirement,
            "details": details,
        }
    )


def validate_readme(
    project_root: Path = PROJECT_ROOT,
    contract: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Validate README structure, evidence tokens, links, and safeguards."""

    if contract is None:
        contract = load_contract(project_root / "config" / "readme_portfolio.yaml")

    readme_path = project_root / contract["readme_path"]
    evidence_path = project_root / contract["evidence_map_path"]

    if not readme_path.exists():
        raise FileNotFoundError(f"Missing README: {readme_path}")

    readme = readme_path.read_text(encoding="utf-8")
    lower_readme = readme.lower()
    normalized_lower_readme = normalize_whitespace(lower_readme)
    checks: list[dict[str, str]] = []

    lines = readme.splitlines()
    minimum = int(contract["line_limits"]["minimum"])
    maximum = int(contract["line_limits"]["maximum"])
    append_check(
        checks,
        "README length is reviewer-appropriate",
        minimum <= len(lines) <= maximum,
        len(lines),
        f"{minimum}–{maximum} lines",
        "The entry point should be substantive without becoming a duplicate manual.",
    )

    missing_headings = [
        heading for heading in contract["required_headings"] if heading not in readme
    ]
    append_check(
        checks,
        "Required narrative sections are present",
        not missing_headings,
        missing_headings,
        "No missing headings",
        "The README must cover decision, evidence, use, and limitations.",
    )

    missing_phrases = [
        phrase
        for phrase in contract["required_phrases"]
        if normalize_whitespace(phrase.lower()) not in normalized_lower_readme
    ]
    append_check(
        checks,
        "Synthetic and governance safeguards are explicit",
        not missing_phrases,
        missing_phrases,
        "No missing safeguards",
        "A reviewer must not mistake the project for real employee evidence.",
    )

    missing_metrics = [
        f"{name}={value}"
        for name, value in contract["headline_metrics"].items()
        if str(value) not in readme
    ]
    append_check(
        checks,
        "Headline metrics match the committed contract",
        not missing_metrics,
        missing_metrics,
        "No missing metric tokens",
        "Displayed results must remain tied to the deterministic Version 2 run.",
    )

    stale_claims = [
        claim
        for claim in contract["forbidden_stale_claims"]
        if claim.lower() in lower_readme
    ]
    append_check(
        checks,
        "Stale Version 1 claims are absent",
        not stale_claims,
        stale_claims,
        "No forbidden claims",
        "The README cannot mix obsolete thresholds and metrics with Version 2.",
    )

    missing_required_links = [
        target for target in contract["required_links"] if target not in readme
    ]
    append_check(
        checks,
        "Required reviewer links are included",
        not missing_required_links,
        missing_required_links,
        "No missing required links",
        "Reviewers need direct paths to notebooks, methods, and runners.",
    )

    broken_links = unresolved_links(readme, project_root)
    append_check(
        checks,
        "All local Markdown links resolve",
        not broken_links,
        broken_links,
        "No unresolved local targets",
        "Repository navigation must work on GitHub and in a local clone.",
    )

    missing_commands = [
        command for command in contract["required_commands"] if command not in readme
    ]
    append_check(
        checks,
        "Setup and reproduction commands are complete",
        not missing_commands,
        missing_commands,
        "No missing commands",
        "A new reviewer must be able to install, validate, and run the project.",
    )

    fence_count = sum(line.strip().startswith("```") for line in lines)
    append_check(
        checks,
        "Markdown code fences are balanced",
        fence_count > 0 and fence_count % 2 == 0,
        fence_count,
        "Positive even count",
        "Broken fences can hide most of the README on GitHub.",
    )

    machine_paths = [
        marker
        for marker in ["c:\\users\\", "/workspace/", "/home/"]
        if marker in lower_readme
    ]
    append_check(
        checks,
        "README contains no machine-specific paths",
        not machine_paths,
        machine_paths,
        "No local absolute paths",
        "Commands and links must remain portable.",
    )

    append_check(
        checks,
        "Evidence map exists",
        evidence_path.is_file(),
        evidence_path.relative_to(project_root),
        "Committed evidence map",
        "Headline claims need an auditable source map.",
    )

    return checks


def write_checks(
    checks: list[dict[str, str]],
    path: Path = OUTPUT_PATH,
) -> None:
    """Write generated README validation evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "check",
                "status",
                "observed",
                "requirement",
                "details",
            ],
        )
        writer.writeheader()
        writer.writerows(checks)


def print_checks(checks: list[dict[str, str]]) -> None:
    """Print a compact terminal validation summary."""

    print("\nREADME PORTFOLIO VALIDATION")
    print("-" * 72)

    for check in checks:
        print(f"{check['status']:<4}  {check['check']:<48}  {check['observed']}")

    passed = sum(check["status"] == "PASS" for check in checks)
    print("-" * 72)
    print(f"Checks passed: {passed}/{len(checks)}")


def main() -> None:
    """Validate the README and fail if any contract check fails."""

    checks = validate_readme()
    write_checks(checks)
    print_checks(checks)

    failures = [check for check in checks if check["status"] != "PASS"]

    if failures:
        names = ", ".join(check["check"] for check in failures)
        raise ValueError(f"README portfolio validation failed: {names}")

    print(f"\nSaved validation: {OUTPUT_PATH}")
    print("\nRECRUITER-FACING README VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
