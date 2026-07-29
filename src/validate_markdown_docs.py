"""Validate reviewer-facing Markdown fences and Mermaid rendering contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any

import yaml


OPENING_FENCE = re.compile(
    r"^(?P<indent> {0,3})(?P<marker>`{3,}|~{3,})(?P<info>.*)$"
)


@dataclass(frozen=True)
class FenceBlock:
    """One balanced fenced block found in a Markdown document."""

    opening_line: int
    closing_line: int
    marker: str
    language: str
    content: str


@dataclass(frozen=True)
class FenceIssue:
    """One unclosed fenced block."""

    opening_line: int
    marker: str
    language: str


@dataclass(frozen=True)
class DocumentAudit:
    """Fence and Mermaid counts for one Markdown file."""

    file: str
    fenced_blocks: int
    mermaid_blocks: int
    unclosed_fences: int
    status: str
    details: str


def load_config(path: Path) -> dict[str, Any]:
    """Load the committed Markdown-integrity contract."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def closing_fence_matches(line: str, marker: str) -> bool:
    """Return whether a line closes the active CommonMark fence."""

    character = re.escape(marker[0])
    minimum_length = len(marker)
    pattern = rf"^ {{0,3}}{character}{{{minimum_length},}}[ \t]*$"
    return re.fullmatch(pattern, line) is not None


def parse_fenced_blocks(text: str) -> tuple[list[FenceBlock], list[FenceIssue]]:
    """Parse balanced and unclosed backtick or tilde fences."""

    blocks: list[FenceBlock] = []
    issues: list[FenceIssue] = []
    active: dict[str, Any] | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        if active is not None:
            marker = active["marker"]
            if closing_fence_matches(line, marker):
                blocks.append(
                    FenceBlock(
                        opening_line=active["opening_line"],
                        closing_line=line_number,
                        marker=marker,
                        language=active["language"],
                        content="\n".join(active["content"]),
                    )
                )
                active = None
            else:
                active["content"].append(line)
            continue

        match = OPENING_FENCE.fullmatch(line)
        if match is None:
            continue

        marker = match.group("marker")
        info = match.group("info").strip()
        if marker.startswith("`") and "`" in info:
            continue

        language = info.split(maxsplit=1)[0].casefold() if info else ""
        active = {
            "opening_line": line_number,
            "marker": marker,
            "language": language,
            "content": [],
        }

    if active is not None:
        issues.append(
            FenceIssue(
                opening_line=active["opening_line"],
                marker=active["marker"],
                language=active["language"],
            )
        )

    return blocks, issues


def discover_markdown_files(
    project_root: Path,
    scan_targets: list[str],
) -> list[Path]:
    """Resolve configured Markdown files and directories without duplicates."""

    discovered: set[Path] = set()

    for relative_target in scan_targets:
        target = project_root / relative_target
        if target.is_file():
            if target.suffix.casefold() != ".md":
                raise ValueError(f"Configured file is not Markdown: {relative_target}")
            discovered.add(target)
            continue

        if target.is_dir():
            discovered.update(
                path for path in target.rglob("*.md") if path.is_file()
            )
            continue

        raise FileNotFoundError(f"Markdown scan target does not exist: {target}")

    return sorted(discovered)


def relative_name(path: Path, project_root: Path) -> str:
    """Return a portable repository-relative path."""

    return path.relative_to(project_root).as_posix()


def audit_documents(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[list[DocumentAudit], dict[str, list[FenceBlock]]]:
    """Audit every configured Markdown document."""

    paths = discover_markdown_files(project_root, config["scan_targets"])
    audits: list[DocumentAudit] = []
    blocks_by_file: dict[str, list[FenceBlock]] = {}

    for path in paths:
        name = relative_name(path, project_root)
        text = path.read_text(encoding="utf-8")
        blocks, issues = parse_fenced_blocks(text)
        blocks_by_file[name] = blocks

        details = "All fenced blocks are balanced."
        if issues:
            locations = ", ".join(str(issue.opening_line) for issue in issues)
            details = f"Unclosed fence opened at line(s): {locations}."

        audits.append(
            DocumentAudit(
                file=name,
                fenced_blocks=len(blocks),
                mermaid_blocks=sum(
                    block.language == "mermaid" for block in blocks
                ),
                unclosed_fences=len(issues),
                status="PASS" if not issues else "FAIL",
                details=details,
            )
        )

    return audits, blocks_by_file


def validate_mermaid_contract(
    blocks_by_file: dict[str, list[FenceBlock]],
    contract: dict[str, Any],
) -> tuple[bool, str]:
    """Validate the committed data-model Mermaid ER diagram."""

    file_name = contract["file"]
    language = contract["language"].casefold()
    mermaid_blocks = [
        block
        for block in blocks_by_file.get(file_name, [])
        if block.language == language
    ]

    if len(mermaid_blocks) != 1:
        return False, f"Expected 1 Mermaid block in {file_name}; found {len(mermaid_blocks)}."

    content = mermaid_blocks[0].content.strip()
    if not content.startswith(contract["diagram_type"]):
        return False, f"Mermaid block does not start with {contract['diagram_type']}."

    missing = [
        relationship
        for relationship in contract["required_relationships"]
        if relationship not in content
    ]
    if missing:
        return False, f"Missing relationships: {missing}"

    return True, (
        f"Closed lines {mermaid_blocks[0].opening_line}-"
        f"{mermaid_blocks[0].closing_line}; "
        f"{len(contract['required_relationships'])} required relationships present."
    )


def build_checks(
    project_root: Path,
    config: dict[str, Any],
    audits: list[DocumentAudit],
    blocks_by_file: dict[str, list[FenceBlock]],
) -> list[dict[str, Any]]:
    """Build reviewer-readable validation checks."""

    audited_names = {audit.file for audit in audits}
    regression_documents = set(config["regression_documents"])
    failed_documents = [
        audit.file for audit in audits if audit.unclosed_fences > 0
    ]
    mermaid_ok, mermaid_details = validate_mermaid_contract(
        blocks_by_file,
        config["mermaid_contract"],
    )
    governance = config["governance"]
    governance_ok = (
        governance["documentation_only"]
        and not governance["changes_model_results"]
        and not governance["changes_frozen_policy"]
        and not governance["changes_dashboard_data"]
    )

    checks = [
        {
            "check": "Configured Markdown population is sufficiently broad",
            "status": (
                "PASS"
                if len(audits) >= config["minimum_markdown_files"]
                else "FAIL"
            ),
            "observed": len(audits),
            "requirement": f">= {config['minimum_markdown_files']} files",
            "details": "README, documentation, and notebook index are audited.",
        },
        {
            "check": "All reported regression documents are audited",
            "status": (
                "PASS"
                if regression_documents.issubset(audited_names)
                else "FAIL"
            ),
            "observed": sorted(regression_documents - audited_names),
            "requirement": "No missing regression documents",
            "details": "Every document named in the review concern is protected.",
        },
        {
            "check": "Every Markdown fence is explicitly closed",
            "status": "PASS" if not failed_documents else "FAIL",
            "observed": failed_documents,
            "requirement": "0 documents with unclosed fences",
            "details": "Backtick and tilde fences follow CommonMark closing rules.",
        },
        {
            "check": "Data-model Mermaid ER diagram is complete",
            "status": "PASS" if mermaid_ok else "FAIL",
            "observed": mermaid_details,
            "requirement": "One closed erDiagram with required relationships",
            "details": "GitHub can render the ER diagram instead of treating it as prose.",
        },
        {
            "check": "Checkpoint remains documentation-only",
            "status": "PASS" if governance_ok else "FAIL",
            "observed": governance,
            "requirement": "No analytical or operational result changes",
            "details": "The repair cannot modify model, policy, or dashboard outputs.",
        },
    ]

    return checks


def save_outputs(
    project_root: Path,
    config: dict[str, Any],
    audits: list[DocumentAudit],
    checks: list[dict[str, Any]],
) -> Path:
    """Save aggregate validation evidence."""

    output_directory = project_root / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    audit_path = output_directory / "markdown_document_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(audits[0])))
        writer.writeheader()
        writer.writerows(asdict(audit) for audit in audits)

    check_path = output_directory / "markdown_integrity_validation.csv"
    with check_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "markdown_files_audited": len(audits),
        "fenced_blocks_audited": sum(audit.fenced_blocks for audit in audits),
        "mermaid_blocks_audited": sum(audit.mermaid_blocks for audit in audits),
        "documents_with_unclosed_fences": sum(
            audit.unclosed_fences > 0 for audit in audits
        ),
        "checks_passed": sum(check["status"] == "PASS" for check in checks),
        "checks_total": len(checks),
    }
    (output_directory / "markdown_integrity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_directory


def print_results(
    audits: list[DocumentAudit],
    checks: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """Print concise validation evidence."""

    print("\nMARKDOWN RENDERING SUMMARY")
    print(f"Markdown files audited: {len(audits)}")
    print(f"Fenced blocks audited: {sum(audit.fenced_blocks for audit in audits)}")
    print(f"Mermaid blocks audited: {sum(audit.mermaid_blocks for audit in audits)}")
    print(
        "Documents with unclosed fences: "
        f"{sum(audit.unclosed_fences > 0 for audit in audits)}"
    )

    print("\nMARKDOWN INTEGRITY VALIDATION")
    for check in checks:
        print(
            f"{check['status']:<4}  {check['check']:<52}  "
            f"{check['observed']}"
        )

    print(f"\nSaved Markdown validation outputs to: {output_directory}")


def main() -> None:
    """Run the committed Markdown-integrity validation."""

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config" / "markdown_integrity.yaml")
    audits, blocks_by_file = audit_documents(project_root, config)
    checks = build_checks(project_root, config, audits, blocks_by_file)
    output_directory = save_outputs(project_root, config, audits, checks)
    print_results(audits, checks, output_directory)

    failed = [check["check"] for check in checks if check["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Markdown integrity validation failed: {failed}")

    print("\nMARKDOWN RENDERING INTEGRITY VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
