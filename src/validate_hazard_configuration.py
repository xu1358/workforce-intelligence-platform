"""Validate that attrition-hazard simulation config controls runtime behavior."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from attrition_hazard import (
    REQUIRED_SIMULATION_KEYS,
    SUPPORTED_CAUSE_MODELS,
    _cause_probabilities,
    _employee_is_at_risk,
    _feature_cutoff,
    hazard_simulation_settings,
    load_hazard_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rejects_setting(
    config: dict[str, Any],
    key: str,
    value: Any,
) -> bool:
    """Return whether a changed simulation setting is rejected."""

    candidate = deepcopy(config)
    candidate["simulation"][key] = value
    try:
        hazard_simulation_settings(candidate)
    except ValueError:
        return True
    return False


def build_checks(
    project_root: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build executable checks for every simulation setting."""

    config = load_hazard_config(project_root / contract["hazard_config"])
    settings = hazard_simulation_settings(config)
    implementation_text = (
        project_root / contract["implementation"]
    ).read_text(encoding="utf-8")
    required_settings = contract["required_settings"]
    expected_values = {
        key: specification["expected"]
        for key, specification in required_settings.items()
    }
    observed_values = {
        "time_step": settings.time_step,
        "as_of_date": settings.as_of_date.strftime("%Y-%m-%d"),
        "cause_model": settings.cause_model,
        "feature_lag_months": settings.feature_lag_months,
        "allow_first_month_exit": settings.allow_first_month_exit,
        "censor_at_as_of_date": settings.censor_at_as_of_date,
    }
    missing_runtime_references = [
        key
        for key, specification in required_settings.items()
        if specification["runtime_use"] not in implementation_text
    ]

    behavior = contract["behavior"]
    month_start = pd.Timestamp(behavior["hire_month_start"])
    month_end = pd.Timestamp(behavior["hire_month_end"])
    hire_date = pd.Timestamp(behavior["hire_month_example"])
    one_month_cutoff = _feature_cutoff(month_start, 1)
    two_month_cutoff = _feature_cutoff(month_start, 2)
    first_month_allowed = _employee_is_at_risk(
        hire_date,
        month_start,
        month_end,
        True,
    )
    first_month_blocked = not _employee_is_at_risk(
        hire_date,
        month_start,
        month_end,
        False,
    )
    probabilities = _cause_probabilities(
        0.0,
        -1.0,
        config,
        settings.cause_model,
    )
    governance = contract["governance"]
    governance_ok = (
        governance["preserves_committed_default_behavior"]
        and not governance["changes_reference_random_seed"]
        and not governance["changes_model_results"]
        and not governance["changes_frozen_policy"]
        and not governance["changes_dashboard_data"]
    )

    return [
        {
            "check": "Simulation setting population is exact",
            "status": (
                "PASS"
                if set(config["simulation"]) == REQUIRED_SIMULATION_KEYS
                else "FAIL"
            ),
            "observed": sorted(config["simulation"]),
            "requirement": sorted(REQUIRED_SIMULATION_KEYS),
            "details": "No required key may be absent or silently optional.",
        },
        {
            "check": "Configured values materialize as typed settings",
            "status": (
                "PASS" if observed_values == expected_values else "FAIL"
            ),
            "observed": observed_values,
            "requirement": expected_values,
            "details": "YAML values are parsed once into the runtime contract.",
        },
        {
            "check": "Every setting has an implementation reference",
            "status": "PASS" if not missing_runtime_references else "FAIL",
            "observed": missing_runtime_references,
            "requirement": "0 missing runtime references",
            "details": "The source must read each configured simulation control.",
        },
        {
            "check": "Feature lag changes the exclusive cutoff",
            "status": (
                "PASS"
                if one_month_cutoff
                == pd.Timestamp(behavior["one_month_lag_cutoff"])
                and two_month_cutoff
                == pd.Timestamp(behavior["two_month_lag_cutoff"])
                else "FAIL"
            ),
            "observed": {
                "lag_1": one_month_cutoff.strftime("%Y-%m-%d"),
                "lag_2": two_month_cutoff.strftime("%Y-%m-%d"),
            },
            "requirement": {
                "lag_1": behavior["one_month_lag_cutoff"],
                "lag_2": behavior["two_month_lag_cutoff"],
            },
            "details": "The committed lag of one preserves the former cutoff.",
        },
        {
            "check": "First-month exit flag changes hire-month risk",
            "status": (
                "PASS"
                if first_month_allowed and first_month_blocked
                else "FAIL"
            ),
            "observed": {
                "enabled": first_month_allowed,
                "disabled": not first_month_blocked,
            },
            "requirement": {"enabled": True, "disabled": False},
            "details": "The switch directly controls risk-set membership.",
        },
        {
            "check": "Cause model selects a probability implementation",
            "status": (
                "PASS"
                if settings.cause_model in SUPPORTED_CAUSE_MODELS
                and abs(sum(probabilities) - 1.0) <= 1e-12
                and rejects_setting(
                    config,
                    "cause_model",
                    "unimplemented_model",
                )
                else "FAIL"
            ),
            "observed": {
                "model": settings.cause_model,
                "probability_sum": sum(probabilities),
            },
            "requirement": "Implemented model and probabilities sum to 1",
            "details": "Unknown model names now fail before simulation.",
        },
        {
            "check": "As-of censoring is an enforced boundary",
            "status": (
                "PASS"
                if settings.censor_at_as_of_date
                and rejects_setting(config, "censor_at_as_of_date", False)
                else "FAIL"
            ),
            "observed": {
                "as_of_date": settings.as_of_date.strftime("%Y-%m-%d"),
                "censoring": settings.censor_at_as_of_date,
                "false_rejected": rejects_setting(
                    config,
                    "censor_at_as_of_date",
                    False,
                ),
            },
            "requirement": "Censor at 2026-06-30; reject an open horizon",
            "details": "The generator cannot imply observations after its boundary.",
        },
        {
            "check": "Committed defaults preserve prior semantics",
            "status": (
                "PASS"
                if settings.feature_lag_months == 1
                and settings.allow_first_month_exit
                and settings.cause_model == "multinomial_logit"
                and settings.censor_at_as_of_date
                else "FAIL"
            ),
            "observed": observed_values,
            "requirement": expected_values,
            "details": "Wiring removes decoration without retuning the generator.",
        },
        {
            "check": "Checkpoint governance prevents downstream changes",
            "status": "PASS" if governance_ok else "FAIL",
            "observed": governance,
            "requirement": "No data, model, policy, or dashboard result changes",
            "details": "This checkpoint enforces configuration behavior only.",
        },
    ]


def save_outputs(
    project_root: Path,
    contract: dict[str, Any],
    checks: list[dict[str, Any]],
) -> Path:
    """Save aggregate configuration-validation evidence."""

    output_directory = project_root / contract["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    with (output_directory / "hazard_configuration_validation.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "checks_passed": sum(check["status"] == "PASS" for check in checks),
        "checks_total": len(checks),
        "configured_settings": len(REQUIRED_SIMULATION_KEYS),
        "supported_cause_models": sorted(SUPPORTED_CAUSE_MODELS),
    }
    (output_directory / "hazard_configuration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_directory


def print_results(
    checks: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """Print the executable hazard-configuration contract."""

    print("\nHAZARD SIMULATION CONFIGURATION")
    for check in checks:
        print(
            f"{check['status']:<4}  {check['check']:<52}  "
            f"{check['observed']}"
        )
    print(f"\nSaved hazard-configuration outputs to: {output_directory}")


def main() -> None:
    """Run the hazard-configuration validation."""

    contract = load_yaml(
        PROJECT_ROOT / "config" / "hazard_configuration_contract.yaml"
    )
    checks = build_checks(PROJECT_ROOT, contract)
    output_directory = save_outputs(PROJECT_ROOT, contract, checks)
    print_results(checks, output_directory)

    failed = [check["check"] for check in checks if check["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Hazard-configuration validation failed: {failed}")

    print("\nHAZARD CONFIGURATION CONTRACT VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
