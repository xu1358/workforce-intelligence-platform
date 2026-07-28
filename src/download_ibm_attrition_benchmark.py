"""Download and verify the pinned IBM attrition benchmark dataset."""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "ibm_attrition_benchmark.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the benchmark configuration."""

    if not path.exists():
        raise FileNotFoundError(f"Missing IBM benchmark config: {path}")

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("IBM benchmark config must contain a YAML mapping.")

    return config


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> str:
    """Verify a file against its committed SHA-256 digest."""

    observed = file_sha256(path)

    if observed != expected_sha256:
        raise ValueError(
            "IBM benchmark checksum mismatch. "
            f"Expected {expected_sha256}, observed {observed}."
        )

    return observed


def validate_source_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate the downloaded dataset's fixed schema and target counts."""

    schema = config["schema"]
    expected_shape = (
        int(schema["expected_rows"]),
        int(schema["expected_columns"]),
    )

    if frame.shape != expected_shape:
        raise ValueError(
            "IBM benchmark shape mismatch. "
            f"Expected {expected_shape}, observed {frame.shape}."
        )

    expected_columns = list(schema["expected_columns_in_order"])
    if frame.columns.tolist() != expected_columns:
        raise ValueError("IBM benchmark columns do not match the pinned schema.")

    target = str(schema["target_column"])
    counts = frame[target].value_counts(dropna=False).to_dict()
    expected_counts = {
        str(schema["positive_label"]): int(schema["expected_positive_cases"]),
        str(schema["negative_label"]): int(schema["expected_negative_cases"]),
    }

    if counts != expected_counts:
        raise ValueError(
            "IBM benchmark target counts changed. "
            f"Expected {expected_counts}, observed {counts}."
        )

    if int(frame.isna().sum().sum()) != 0:
        raise ValueError("IBM benchmark unexpectedly contains missing values.")

    identifier = str(schema["identifier_column"])
    if not frame[identifier].is_unique:
        raise ValueError("IBM benchmark employee identifiers are not unique.")

    for column, expected_value in schema["constant_columns"].items():
        observed = frame[column].drop_duplicates().tolist()
        if observed != [expected_value]:
            raise ValueError(f"IBM benchmark constant changed for {column}: {observed}")


def download_dataset(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, str, bool]:
    """Download the pinned file when needed and return validation metadata."""

    source = config["source"]
    destination = project_root / str(source["local_path"])
    expected_sha256 = str(source["sha256"])
    downloaded = False

    if destination.exists():
        try:
            observed = verify_checksum(destination, expected_sha256)
        except ValueError:
            observed = ""

        if observed == expected_sha256:
            frame = pd.read_csv(destination)
            validate_source_frame(frame, config)
            return destination, observed, downloaded

    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        str(source["raw_url"]),
        headers={"User-Agent": "workforce-intelligence-platform/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix="ibm_attrition_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)

    try:
        observed = verify_checksum(temporary_path, expected_sha256)
        frame = pd.read_csv(temporary_path)
        validate_source_frame(frame, config)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    downloaded = True
    return destination, observed, downloaded


def main() -> None:
    """Download and verify the external benchmark dataset."""

    config = load_config()
    path, checksum, downloaded = download_dataset(config)
    frame = pd.read_csv(path)
    source = config["source"]

    print("\nIBM EXTERNAL BENCHMARK SOURCE")
    print(f"Dataset: {source['dataset_name']}")
    print(f"Fictional data: {source['fictional_data']}")
    print(f"Source commit: {source['source_commit']}")
    print(f"License: {source['license_name']}")
    print(f"Downloaded now: {downloaded}")
    print(f"Saved path: {path}")
    print(f"SHA-256: {checksum}")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {len(frame.columns)}")
    print("\nIBM BENCHMARK DATASET VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    main()
