"""Regression tests for reviewer-facing result documentation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def load_contract(project_root: Path) -> dict[str, Any]:
    """Load the documentation-results contract."""

    path = project_root / "config" / "documented_results.yaml"

    with path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)

    assert isinstance(contract, dict)
    assert contract["documents"]
    return contract


def document_text(project_root: Path, document: dict[str, Any]) -> str:
    """Read one contracted documentation page."""

    path = project_root / document["path"]
    assert path.is_file()
    return path.read_text(encoding="utf-8")


def local_links(markdown: str) -> list[str]:
    """Extract local Markdown link targets."""

    links: list[str] = []

    for raw_target in MARKDOWN_LINK.findall(markdown):
        target = raw_target.strip().split(maxsplit=1)[0]
        target = target.split("#", maxsplit=1)[0]

        if target and not target.startswith(("http://", "https://", "mailto:", "#")):
            links.append(target.replace("\\", "/"))

    return links


def test_result_documents_have_required_sections(project_root: Path) -> None:
    """Every reviewer-facing page should lead with results and retain evidence."""

    contract = load_contract(project_root)

    for document in contract["documents"]:
        text = document_text(project_root, document)

        for heading in contract["required_headings"]:
            assert heading in text, f"{document['path']} is missing {heading}"


def test_result_documents_contain_committed_evidence(project_root: Path) -> None:
    """Headline metrics and interpretation boundaries must remain visible."""

    contract = load_contract(project_root)

    for document in contract["documents"]:
        text = document_text(project_root, document)
        normalized_lower_text = " ".join(text.lower().split())

        for value in document["required_values"]:
            assert value in text, f"{document['path']} is missing result {value}"

        for phrase in document["required_phrases"]:
            normalized_phrase = " ".join(phrase.lower().split())
            assert normalized_phrase in normalized_lower_text, (
                f"{document['path']} is missing interpretation: {phrase}"
            )


def test_result_document_code_fences_are_balanced(project_root: Path) -> None:
    """An unfinished code block must not break GitHub rendering again."""

    contract = load_contract(project_root)

    for document in contract["documents"]:
        text = document_text(project_root, document)
        fence_count = sum(line.strip().startswith("```") for line in text.splitlines())

        assert fence_count % 2 == 0, f"{document['path']} has an open code fence"


def test_result_document_local_links_resolve(project_root: Path) -> None:
    """Notebook and implementation links should resolve inside the repository."""

    contract = load_contract(project_root)

    for document in contract["documents"]:
        document_path = project_root / document["path"]
        text = document_path.read_text(encoding="utf-8")
        unresolved = [
            target
            for target in local_links(text)
            if not (document_path.parent / target).resolve().is_file()
        ]

        assert unresolved == [], (
            f"{document['path']} has unresolved links: {sorted(unresolved)}"
        )
