"""Tests that committed notebooks do not expose local development paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

from validate_portfolio_notebooks import notebook_text


MACHINE_PATH_PATTERNS = {
    "Windows user directory": re.compile(
        r"[A-Za-z]:\\Users\\",
        re.IGNORECASE,
    ),
    "macOS user directory": re.compile(
        r"/Users/[^/\s]+/",
        re.IGNORECASE,
    ),
    "Linux home directory": re.compile(
        r"/home/[^/\s]+/",
        re.IGNORECASE,
    ),
    "hosted workspace directory": re.compile(
        r"/workspace/",
        re.IGNORECASE,
    ),
}


def test_committed_notebooks_do_not_expose_local_paths(
    project_root: Path,
) -> None:
    """No committed notebook may disclose a developer's local directory."""
    exposed_paths: dict[str, list[str]] = {}
    notebook_root = project_root / "notebooks"

    for path in sorted(notebook_root.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue

        notebook = json.loads(path.read_text(encoding="utf-8"))
        text = notebook_text(notebook)

        matches = [
            label
            for label, pattern in MACHINE_PATH_PATTERNS.items()
            if pattern.search(text)
        ]

        if matches:
            relative_path = path.relative_to(project_root).as_posix()
            exposed_paths[relative_path] = matches

    assert not exposed_paths, f"Machine-specific notebook paths found: {exposed_paths}"
