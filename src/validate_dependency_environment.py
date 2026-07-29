"""Validate the committed Python dependency and runtime contract."""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "dependency_reproducibility.yaml"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "python-quality.yml"


@dataclass(frozen=True)
class ParsedRequirement:
    """One top-level requirement declaration."""

    requirement: Requirement
    hash_count: int


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Required configuration was not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return payload


def sha256_file(path: Path) -> str:
    """Return a deterministic SHA-256 fingerprint."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_requirements(path: Path) -> dict[str, ParsedRequirement]:
    """Parse top-level requirement lines and their artifact hashes."""

    if not path.exists():
        raise FileNotFoundError(f"Requirement file was not found: {path}")

    parsed: dict[str, ParsedRequirement] = {}
    current_name: str | None = None
    current_hash_count = 0

    def finish_current() -> None:
        nonlocal current_name, current_hash_count
        if current_name is None:
            return

        previous = parsed[current_name]
        parsed[current_name] = ParsedRequirement(
            requirement=previous.requirement,
            hash_count=current_hash_count,
        )
        current_name = None
        current_hash_count = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if raw_line[:1].isspace():
            if stripped.startswith("--hash=sha256:"):
                current_hash_count += 1
            continue

        finish_current()
        declaration = stripped.removesuffix("\\").strip()
        requirement = Requirement(declaration)
        name = canonicalize_name(requirement.name)

        if name in parsed:
            raise ValueError(f"Duplicate requirement {requirement.name!r} in {path}")

        parsed[name] = ParsedRequirement(requirement=requirement, hash_count=0)
        current_name = name

    finish_current()
    return parsed


def exact_version(requirement: Requirement) -> str | None:
    """Return the exact pinned version or None for a floating declaration."""

    specifiers = list(requirement.specifier)
    if requirement.url is not None or len(specifiers) != 1:
        return None

    specifier = specifiers[0]
    if specifier.operator != "==" or "*" in specifier.version:
        return None

    return specifier.version


def active_requirement(requirement: Requirement) -> bool:
    """Return whether a universal-lock marker applies to this runtime."""

    if requirement.marker is None:
        return True
    return requirement.marker.evaluate(environment=default_environment())


def installed_versions() -> dict[str, str]:
    """Return installed distributions using normalized names."""

    installed: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[canonicalize_name(name)] = distribution.version
    return installed


def add_check(
    checks: list[dict[str, str]],
    name: str,
    passed: bool,
    observed: str,
    requirement: str,
    details: str,
) -> None:
    """Append one validation result."""

    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "requirement": requirement,
            "details": details,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of mappings as a CSV file."""

    if not rows:
        raise ValueError(f"Cannot write an empty validation artifact: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a compact fixed-width table."""

    print("")
    print(title)
    widths = {
        column: max(
            len(column),
            *(len(str(row.get(column, ""))) for row in rows),
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(
                str(row.get(column, "")).ljust(widths[column]) for column in columns
            )
        )


def validate_environment() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build environment rows and validation checks."""

    config = load_yaml(CONFIG_PATH)
    direct_path = PROJECT_ROOT / config["files"]["direct_requirements"]
    lock_path = PROJECT_ROOT / config["files"]["lock_file"]

    direct = parse_requirements(direct_path)
    locked = parse_requirements(lock_path)
    installed = installed_versions()
    checks: list[dict[str, str]] = []

    direct_floating = sorted(
        item.requirement.name
        for item in direct.values()
        if exact_version(item.requirement) is None
    )
    lock_floating = sorted(
        item.requirement.name
        for item in locked.values()
        if exact_version(item.requirement) is None
    )
    lock_missing_hashes = sorted(
        item.requirement.name for item in locked.values() if item.hash_count == 0
    )

    expected_direct_count = int(config["expected_counts"]["direct_dependencies"])
    expected_lock_count = int(config["expected_counts"]["locked_distributions"])

    add_check(
        checks,
        "Direct dependency count is exact",
        len(direct) == expected_direct_count,
        str(len(direct)),
        str(expected_direct_count),
        "The human-maintained dependency surface is explicit.",
    )
    add_check(
        checks,
        "Every direct dependency is exactly pinned",
        not direct_floating,
        str(direct_floating),
        "No bare names, ranges, wildcards, URLs, or VCS references",
        "requirements.txt cannot silently resolve a new direct version.",
    )
    add_check(
        checks,
        "Transitive lock count is exact",
        len(locked) == expected_lock_count,
        str(len(locked)),
        str(expected_lock_count),
        "The universal Python 3.12 resolution is committed.",
    )
    add_check(
        checks,
        "Every locked distribution is exactly pinned",
        not lock_floating,
        str(lock_floating),
        "Every declaration uses one exact == version",
        "Transitive dependencies cannot float during installation.",
    )
    add_check(
        checks,
        "Every locked distribution has artifact hashes",
        not lock_missing_hashes,
        str(lock_missing_hashes),
        "At least one SHA-256 artifact hash per distribution",
        "pip can reject package files outside the committed lock.",
    )

    direct_lock_mismatches: list[str] = []
    for name, item in direct.items():
        locked_item = locked.get(name)
        direct_version = exact_version(item.requirement)
        locked_version = (
            exact_version(locked_item.requirement) if locked_item is not None else None
        )
        if direct_version != locked_version:
            direct_lock_mismatches.append(
                f"{item.requirement.name}: direct={direct_version}, lock={locked_version}"
            )

    add_check(
        checks,
        "Direct pins reconcile to the transitive lock",
        not direct_lock_mismatches,
        str(direct_lock_mismatches),
        "Every direct name and version appears in the lock",
        "The lock must be regenerated when a direct dependency changes.",
    )

    core_mismatches: list[str] = []
    for configured_name, configured_version in config["core_dependencies"].items():
        item = direct.get(canonicalize_name(configured_name))
        actual_version = exact_version(item.requirement) if item else None
        if actual_version != str(configured_version):
            core_mismatches.append(
                f"{configured_name}: configured={configured_version}, "
                f"direct={actual_version}"
            )

    add_check(
        checks,
        "Core dependency versions match the contract",
        not core_mismatches,
        str(core_mismatches),
        "Configured core versions equal requirements.txt",
        "The reviewer-facing compatibility boundary is machine checked.",
    )

    actual_direct_hash = sha256_file(direct_path)
    actual_lock_hash = sha256_file(lock_path)
    expected_direct_hash = config["files"]["direct_requirements_sha256"]
    expected_lock_hash = config["files"]["lock_file_sha256"]

    add_check(
        checks,
        "Direct requirements fingerprint matches",
        actual_direct_hash == expected_direct_hash,
        actual_direct_hash,
        expected_direct_hash,
        "The direct dependency manifest cannot change unnoticed.",
    )
    add_check(
        checks,
        "Lock-file fingerprint matches",
        actual_lock_hash == expected_lock_hash,
        actual_lock_hash,
        expected_lock_hash,
        "The complete resolved environment cannot change unnoticed.",
    )

    expected_python = str(config["python"]["supported_major_minor"])
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    add_check(
        checks,
        "Python runtime is supported",
        actual_python == expected_python,
        sys.version.split()[0],
        f"{expected_python}.x",
        "Local validation and CI use the same Python language generation.",
    )

    expected_pip = str(config["bootstrap"]["pip_version"])
    actual_pip = installed.get("pip")
    add_check(
        checks,
        "pip bootstrap version is exact",
        actual_pip == expected_pip,
        str(actual_pip),
        expected_pip,
        "The resolver itself is fixed before installing the hashed lock.",
    )

    expected_build_tools = {
        "setuptools": str(config["bootstrap"]["setuptools_version"]),
        "wheel": str(config["bootstrap"]["wheel_version"]),
    }
    build_tool_mismatches = [
        f"{name}: expected={version}, installed={installed.get(name)}"
        for name, version in expected_build_tools.items()
        if installed.get(name) != version
    ]
    add_check(
        checks,
        "Source-build tools are exact",
        not build_tool_mismatches,
        str(build_tool_mismatches),
        "Pinned setuptools and wheel",
        "Source distributions cannot select a new build toolchain.",
    )

    environment_rows: list[dict[str, Any]] = []
    active_locked = {
        name: item
        for name, item in locked.items()
        if active_requirement(item.requirement)
    }
    drift: list[str] = []
    for name, item in sorted(active_locked.items()):
        expected_version = exact_version(item.requirement)
        installed_version = installed.get(name)
        status = "MATCH" if installed_version == expected_version else "DRIFT"
        if status == "DRIFT":
            drift.append(
                f"{item.requirement.name}: expected={expected_version}, "
                f"installed={installed_version}"
            )
        environment_rows.append(
            {
                "distribution": item.requirement.name,
                "expected_version": expected_version,
                "installed_version": installed_version or "MISSING",
                "direct_dependency": name in direct,
                "artifact_hashes": item.hash_count,
                "status": status,
            }
        )

    add_check(
        checks,
        "Installed environment matches the active lock",
        not drift,
        f"{len(drift)} mismatches",
        "0 mismatches",
        "Every platform-applicable distribution has the committed version.",
    )

    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    ci_version = str(config["python"]["ci_version"])
    required_workflow_fragments = [
        f'python-version: "{ci_version}"',
        (
            f'python -m pip install "pip=={expected_pip}" '
            f'"setuptools=={expected_build_tools["setuptools"]}" '
            f'"wheel=={expected_build_tools["wheel"]}"'
        ),
        (
            "python -m pip install --no-build-isolation "
            "--require-hashes -r requirements-lock.txt"
        ),
        "python -m pip check",
        "python src/validate_dependency_environment.py",
    ]
    missing_workflow_fragments = [
        fragment for fragment in required_workflow_fragments if fragment not in workflow
    ]
    add_check(
        checks,
        "GitHub Actions enforces the locked environment",
        not missing_workflow_fragments,
        str(missing_workflow_fragments),
        "Pinned Python, pip, hashed install, pip check, and environment validation",
        "A future CI run cannot silently select newer package versions.",
    )

    return environment_rows, checks


def main() -> None:
    """Run dependency reproducibility validation."""

    print("")
    print("==================================================")
    print("Validate dependency reproducibility")
    print("==================================================")

    config = load_yaml(CONFIG_PATH)
    environment_rows, checks = validate_environment()

    output_directory = PROJECT_ROOT / config["outputs"]["directory"]
    write_csv(output_directory / "dependency_environment.csv", environment_rows)
    write_csv(
        output_directory / "dependency_reproducibility_validation.csv",
        checks,
    )

    core_rows = []
    installed = installed_versions()
    for name, version in config["core_dependencies"].items():
        core_rows.append(
            {
                "dependency": name,
                "locked_version": version,
                "installed_version": installed.get(canonicalize_name(name), "MISSING"),
            }
        )

    print_table(
        "CORE DEPENDENCY VERSIONS",
        core_rows,
        ["dependency", "locked_version", "installed_version"],
    )
    print_table(
        "DEPENDENCY REPRODUCIBILITY VALIDATION",
        checks,
        ["check", "status", "observed", "requirement"],
    )

    failures = [check["check"] for check in checks if check["status"] != "PASS"]
    if failures:
        raise ValueError(
            "Dependency reproducibility validation failed: " + ", ".join(failures)
        )

    print("")
    print(
        f"Direct dependencies pinned: {config['expected_counts']['direct_dependencies']}"
    )
    print(f"Locked distributions: {config['expected_counts']['locked_distributions']}")
    print("Installed-version mismatches: 0")
    print(f"Saved validation: {output_directory}")
    print("")
    print("DEPENDENCY REPRODUCIBILITY VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
