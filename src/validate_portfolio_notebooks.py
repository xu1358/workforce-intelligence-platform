"""Validate the curated, reviewer-facing notebook sequence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "portfolio_notebooks.yaml"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "portfolio_notebook_validation.csv"
)


def load_manifest(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the committed portfolio-notebook manifest."""

    if not path.exists():
        raise FileNotFoundError(f"Missing portfolio manifest: {path}")

    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    if not isinstance(manifest, dict):
        raise ValueError("Portfolio manifest must contain a YAML mapping.")

    return manifest


def load_notebook(path: Path) -> dict[str, Any]:
    """Load one notebook as JSON."""

    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    if not isinstance(notebook, dict):
        raise ValueError(f"Notebook must contain a JSON object: {path}")

    return notebook


def markdown_text(notebook: dict[str, Any]) -> str:
    """Combine all markdown cells into one searchable string."""

    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )


def notebook_text(notebook: dict[str, Any]) -> str:
    """Return source and text-output content for portability checks."""

    fragments: list[str] = []

    for cell in notebook.get("cells", []):
        fragments.append("".join(cell.get("source", [])))

        for output in cell.get("outputs", []):
            fragments.append("".join(output.get("text", [])))

            data = output.get("data", {})
            for value in data.values():
                if isinstance(value, list):
                    fragments.append("".join(str(item) for item in value))
                elif isinstance(value, str):
                    fragments.append(value)

    return "\n".join(fragments)


def first_markdown_heading(notebook: dict[str, Any]) -> str:
    """Return the first level-one markdown heading."""

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue

        for line in "".join(cell.get("source", [])).splitlines():
            if line.startswith("# "):
                return line.strip()

    return ""


def code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all code cells."""

    return [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def append_check(
    checks: list[dict[str, str]],
    check: str,
    passed: bool,
    observed: Any,
    requirement: str,
    details: str,
) -> None:
    """Append one normalized validation result."""

    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": str(observed),
            "requirement": requirement,
            "details": details,
        }
    )


def validate_portfolio(
    project_root: Path = PROJECT_ROOT,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Validate the portfolio manifest, notebooks, and index."""

    if manifest is None:
        manifest = load_manifest(
            project_root / "config" / "portfolio_notebooks.yaml"
        )

    checks: list[dict[str, str]] = []
    entries = manifest.get("notebooks", [])
    expected_count = int(manifest["expected_portfolio_notebooks"])
    portfolio_directory = project_root / manifest["portfolio_directory"]
    maximum_size = int(manifest["maximum_notebook_size_bytes"])
    minimum_outputs = int(manifest["minimum_code_outputs_per_notebook"])

    append_check(
        checks,
        "Portfolio notebook count",
        len(entries) == expected_count,
        len(entries),
        str(expected_count),
        "The reviewer path must stay intentionally concise.",
    )

    orders = [int(entry["order"]) for entry in entries]
    append_check(
        checks,
        "Portfolio order is complete",
        orders == list(range(1, expected_count + 1)),
        orders,
        str(list(range(1, expected_count + 1))),
        "Notebook order must be explicit and gap free.",
    )

    files = [str(entry["file"]) for entry in entries]
    titles = [str(entry["title"]) for entry in entries]
    append_check(
        checks,
        "Portfolio files and titles are unique",
        len(files) == len(set(files)) and len(titles) == len(set(titles)),
        f"{len(set(files))} files; {len(set(titles))} titles",
        f"{expected_count} unique files and titles",
        "Each reviewer step must have one stable identity.",
    )

    for entry in entries:
        order = int(entry["order"])
        title = str(entry["title"])
        path = portfolio_directory / str(entry["file"])
        label = f"Notebook {order}"

        exists = path.is_file()
        append_check(
            checks,
            f"{label} exists",
            exists,
            path.relative_to(project_root) if exists else "Missing",
            "Committed notebook file",
            title,
        )

        if not exists:
            continue

        try:
            notebook = load_notebook(path)
        except (json.JSONDecodeError, ValueError) as exc:
            append_check(
                checks,
                f"{label} JSON is valid",
                False,
                type(exc).__name__,
                "Valid notebook JSON",
                str(exc),
            )
            continue

        append_check(
            checks,
            f"{label} uses notebook format 4",
            notebook.get("nbformat") == 4,
            notebook.get("nbformat"),
            "4",
            "The committed notebooks use the current major notebook format.",
        )

        heading = first_markdown_heading(notebook)
        append_check(
            checks,
            f"{label} title matches manifest",
            heading == f"# {title}",
            heading,
            f"# {title}",
            "Displayed and configured titles must agree.",
        )

        markdown = markdown_text(notebook)
        missing_headings = [
            heading
            for heading in entry["required_headings"]
            if heading not in markdown
        ]
        append_check(
            checks,
            f"{label} contains required sections",
            not missing_headings,
            missing_headings,
            "No missing headings",
            "Every notebook follows its committed narrative contract.",
        )

        cells = code_cells(notebook)
        unexecuted = [
            index
            for index, cell in enumerate(cells, start=1)
            if cell.get("execution_count") is None
        ]
        append_check(
            checks,
            f"{label} code cells are executed",
            bool(cells) and not unexecuted,
            f"{len(cells)} cells; unexecuted={unexecuted}",
            "At least one code cell and none unexecuted",
            "GitHub viewers must see reproducible saved evidence.",
        )

        error_outputs = [
            output.get("ename", "Unknown error")
            for cell in cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        append_check(
            checks,
            f"{label} contains no error outputs",
            not error_outputs,
            error_outputs,
            "No error outputs",
            "A portfolio notebook cannot display a failed execution.",
        )

        output_count = sum(len(cell.get("outputs", [])) for cell in cells)
        append_check(
            checks,
            f"{label} embeds sufficient outputs",
            output_count >= minimum_outputs,
            output_count,
            f">= {minimum_outputs}",
            "The curated notebooks must be readable without local execution.",
        )

        kernel_name = (
            notebook.get("metadata", {})
            .get("kernelspec", {})
            .get("name")
        )
        append_check(
            checks,
            f"{label} uses portable kernel metadata",
            kernel_name == "python3",
            kernel_name,
            "python3",
            "A generic kernel name avoids a machine-specific environment label.",
        )

        lower_markdown = markdown.lower()
        append_check(
            checks,
            f"{label} discloses synthetic data",
            "synthetic" in lower_markdown,
            "Present" if "synthetic" in lower_markdown else "Missing",
            "Present",
            "No notebook may imply that the workforce records are real people.",
        )

        navigation_present = (
            "Portfolio navigation" in markdown
            and (
                "Previous:" in markdown
                or "Next:" in markdown
            )
        )
        append_check(
            checks,
            f"{label} includes portfolio navigation",
            navigation_present,
            navigation_present,
            "True",
            "Reviewers must be able to follow the four-notebook sequence.",
        )

        full_text = notebook_text(notebook)
        lower_text = full_text.lower()
        machine_paths = [
            marker
            for marker in ["c:\\users\\", "/workspace/", "/home/"]
            if marker in lower_text
        ]
        append_check(
            checks,
            f"{label} has no machine-specific paths",
            not machine_paths,
            machine_paths,
            "No absolute development paths",
            "Notebook sources and outputs must remain portable.",
        )

        employee_identifier_present = "employee_id" in lower_text
        append_check(
            checks,
            f"{label} presents aggregate evidence only",
            not employee_identifier_present,
            employee_identifier_present,
            "False",
            "The curated layer excludes employee-level review identifiers.",
        )

        size = path.stat().st_size
        append_check(
            checks,
            f"{label} stays within size limit",
            size <= maximum_size,
            size,
            f"<= {maximum_size}",
            "Portfolio notebooks should load reliably on GitHub.",
        )

    index_path = project_root / manifest["index_file"]
    index_text = (
        index_path.read_text(encoding="utf-8")
        if index_path.exists()
        else ""
    )
    missing_links = [
        f"portfolio/{entry['file']}"
        for entry in entries
        if f"portfolio/{entry['file']}" not in index_text
    ]
    append_check(
        checks,
        "Notebook index links every portfolio file",
        index_path.exists() and not missing_links,
        missing_links,
        "No missing links",
        "The notebook directory must provide one clear reviewer entry point.",
    )

    supporting = sorted(
        path
        for path in (project_root / "notebooks").glob("[0-9][0-9]_*.ipynb")
        if path.is_file()
    )
    minimum_supporting = int(manifest["minimum_supporting_notebooks"])
    append_check(
        checks,
        "Supporting notebook history is retained",
        len(supporting) >= minimum_supporting,
        len(supporting),
        f">= {minimum_supporting}",
        "Curation must not destroy the detailed checkpoint audit trail.",
    )

    supporting_directory = project_root / "notebooks"
    executed_supporting_files = [
        str(filename)
        for filename in manifest["executed_supporting_notebooks"]
    ]
    supporting_minimum_outputs = int(
        manifest["minimum_supporting_code_outputs_per_notebook"]
    )
    supporting_paths = [
        supporting_directory / filename
        for filename in executed_supporting_files
    ]
    missing_supporting = [
        path.name
        for path in supporting_paths
        if not path.is_file()
    ]
    append_check(
        checks,
        "Reviewer-visible supporting notebooks exist",
        not missing_supporting,
        missing_supporting,
        "No missing notebooks",
        "Version 2 technical evidence must remain directly reviewable.",
    )

    loaded_supporting: dict[str, dict[str, Any]] = {}
    invalid_supporting: dict[str, str] = {}
    for path in supporting_paths:
        if not path.is_file():
            continue

        try:
            loaded_supporting[path.name] = load_notebook(path)
        except (json.JSONDecodeError, ValueError) as exc:
            invalid_supporting[path.name] = type(exc).__name__

    append_check(
        checks,
        "Reviewer-visible supporting notebook JSON is valid",
        not invalid_supporting,
        invalid_supporting,
        "No invalid notebook JSON",
        "Notebook evidence must remain readable by GitHub and Jupyter.",
    )

    unexecuted_supporting: dict[str, list[int]] = {}
    insufficient_supporting_outputs: dict[str, int] = {}
    supporting_errors: dict[str, list[str]] = {}
    nonportable_supporting: dict[str, list[str]] = {}
    nonportable_supporting_kernels: dict[str, Any] = {}
    oversized_supporting: dict[str, int] = {}

    for filename, notebook in loaded_supporting.items():
        cells = code_cells(notebook)
        unexecuted = [
            index
            for index, cell in enumerate(cells, start=1)
            if cell.get("execution_count") is None
        ]
        if not cells or unexecuted:
            unexecuted_supporting[filename] = unexecuted

        output_count = sum(
            len(cell.get("outputs", []))
            for cell in cells
        )
        if output_count < supporting_minimum_outputs:
            insufficient_supporting_outputs[filename] = output_count

        errors = [
            str(output.get("ename", "Unknown error"))
            for cell in cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if errors:
            supporting_errors[filename] = errors

        lower_text = notebook_text(notebook).lower()
        machine_paths = [
            marker
            for marker in ["c:\\users\\", "/workspace/", "/home/"]
            if marker in lower_text
        ]
        if machine_paths:
            nonportable_supporting[filename] = machine_paths

        kernel_name = (
            notebook.get("metadata", {})
            .get("kernelspec", {})
            .get("name")
        )
        if kernel_name != "python3":
            nonportable_supporting_kernels[filename] = kernel_name

        size = (supporting_directory / filename).stat().st_size
        if size > maximum_size:
            oversized_supporting[filename] = size

    append_check(
        checks,
        "Reviewer-visible supporting code cells are executed",
        not unexecuted_supporting,
        unexecuted_supporting,
        "No unexecuted code cells",
        "GitHub reviewers must see the saved Version 2 evidence.",
    )
    append_check(
        checks,
        "Reviewer-visible supporting outputs are embedded",
        not insufficient_supporting_outputs,
        insufficient_supporting_outputs,
        f">= {supporting_minimum_outputs} output per notebook",
        "Supporting notebooks must be useful without local execution.",
    )
    append_check(
        checks,
        "Reviewer-visible supporting outputs contain no errors",
        not supporting_errors,
        supporting_errors,
        "No error outputs",
        "Committed technical evidence cannot display failed execution.",
    )
    append_check(
        checks,
        "Reviewer-visible supporting notebooks are portable",
        not nonportable_supporting,
        nonportable_supporting,
        "No absolute development paths",
        "Saved sources and outputs must not disclose a development machine.",
    )
    append_check(
        checks,
        "Reviewer-visible supporting kernels are portable",
        not nonportable_supporting_kernels,
        nonportable_supporting_kernels,
        "All kernels named python3",
        "Generic kernel metadata keeps notebooks usable across machines.",
    )
    append_check(
        checks,
        "Reviewer-visible supporting notebooks stay within size limits",
        not oversized_supporting,
        oversized_supporting,
        f"All <= {maximum_size}",
        "Executed supporting notebooks should load reliably on GitHub.",
    )

    return checks


def write_checks(
    checks: list[dict[str, str]],
    path: Path = OUTPUT_PATH,
) -> None:
    """Write validation results to a generated CSV artifact."""

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
    """Print a compact terminal summary."""

    print("\nPORTFOLIO NOTEBOOK VALIDATION")
    print("-" * 72)

    for check in checks:
        print(
            f"{check['status']:<4}  "
            f"{check['check']:<48}  "
            f"{check['observed']}"
        )

    passed = sum(check["status"] == "PASS" for check in checks)
    print("-" * 72)
    print(f"Checks passed: {passed}/{len(checks)}")


def main() -> None:
    """Validate the curated sequence and reviewer-visible supporting evidence."""

    checks = validate_portfolio()
    write_checks(checks)
    print_checks(checks)

    failures = [
        check
        for check in checks
        if check["status"] != "PASS"
    ]

    if failures:
        names = ", ".join(check["check"] for check in failures)
        raise ValueError(
            "Portfolio notebook validation failed: "
            f"{names}"
        )

    print(f"\nSaved validation: {OUTPUT_PATH}")
    print("\nPORTFOLIO NOTEBOOKS VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
