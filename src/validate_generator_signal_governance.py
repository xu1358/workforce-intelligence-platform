"""Validate separation between generator signal checks and model evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_checks(
    project_root: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build generator-signal governance checks."""

    hazard_config = load_yaml(project_root / contract["hazard_config"])
    validator_text = (
        project_root / contract["data_validator"]
    ).read_text(encoding="utf-8")
    reviewer_text = (
        project_root / contract["reviewer_document"]
    ).read_text(encoding="utf-8")
    public_text = "\n".join(
        (project_root / path).read_text(encoding="utf-8")
        for path in contract["public_documents"]
    )

    diagnostic_contract = contract["diagnostic_contract"]
    target_section = diagnostic_contract["config_section"]
    target_key = diagnostic_contract["config_key"]
    configured = hazard_config[target_section][target_key]
    expected_bounds = (
        float(diagnostic_contract["minimum"]),
        float(diagnostic_contract["maximum"]),
    )
    configured_bounds = (
        float(configured["minimum"]),
        float(configured["maximum"]),
    )
    accepted = float(diagnostic_contract["accepted_observed"])
    initial = float(diagnostic_contract["initial_observed_approximate"])
    later = contract["later_model_evidence"]

    public_contract = contract["public_documentation"]
    normalized_public_text = " ".join(public_text.split())
    missing_public_phrases = [
        phrase
        for phrase in public_contract["required_phrases"]
        if phrase.lower() not in normalized_public_text.lower()
    ]
    present_private_markers = [
        phrase
        for phrase in public_contract[
            "private_markers_prohibited_in_repository"
        ]
        if phrase.lower() in public_text.lower()
    ]

    governance = contract["governance"]
    governance_ok = (
        governance["documentation_and_naming_only"]
        and not governance["changes_synthetic_records"]
        and not governance["reruns_generator"]
        and not governance["changes_model_results"]
        and not governance["reopens_final_test"]
        and not governance["changes_frozen_policy"]
        and not governance["changes_dashboard_data"]
    )

    return [
        {
            "check": "Generator acceptance naming is explicit",
            "status": (
                "PASS"
                if target_section == "generator_acceptance_targets"
                and target_key == "observable_signal_diagnostic"
                and "calibration_targets" not in hazard_config
                and "model_roc_auc"
                not in hazard_config.get(target_section, {})
                else "FAIL"
            ),
            "observed": f"{target_section}.{target_key}",
            "requirement": (
                "Generator diagnostic; no model-performance target naming"
            ),
            "details": "The configuration cannot be mistaken for model selection.",
        },
        {
            "check": "Diagnostic metric and bounds reconcile",
            "status": (
                "PASS"
                if configured["metric"] == diagnostic_contract["metric"]
                and configured_bounds == expected_bounds
                and expected_bounds[0] <= accepted <= expected_bounds[1]
                else "FAIL"
            ),
            "observed": {
                "metric": configured["metric"],
                "bounds": configured_bounds,
                "accepted": accepted,
            },
            "requirement": {
                "metric": diagnostic_contract["metric"],
                "bounds": expected_bounds,
            },
            "details": "The lower and upper bounds define a moderate-signal band.",
        },
        {
            "check": "Initial failure and accepted diagnostic are disclosed",
            "status": (
                "PASS"
                if initial < expected_bounds[0]
                and expected_bounds[0] <= accepted <= expected_bounds[1]
                and "0.54" in reviewer_text
                and "0.6298" in reviewer_text
                else "FAIL"
            ),
            "observed": {"initial": initial, "accepted": accepted},
            "requirement": "Disclose both generator iterations",
            "details": "The project does not hide that the generator was calibrated.",
        },
        {
            "check": "Signal check precedes formal model development",
            "status": (
                "PASS"
                if diagnostic_contract[
                    "performed_before_temporal_model_development"
                ]
                and not diagnostic_contract["final_test_target_accessed"]
                and not diagnostic_contract["selects_final_model"]
                else "FAIL"
            ),
            "observed": {
                "before_modeling": diagnostic_contract[
                    "performed_before_temporal_model_development"
                ],
                "test_access": diagnostic_contract[
                    "final_test_target_accessed"
                ],
                "selects_model": diagnostic_contract["selects_final_model"],
            },
            "requirement": "Generator validation only; no test access",
            "details": "Chronology separates data design from classifier selection.",
        },
        {
            "check": "Once-only final test demonstrates no target enforcement",
            "status": (
                "PASS"
                if float(later["final_test_roc_auc"]) < expected_bounds[0]
                and later["final_test_evaluated_once"]
                and not later["post_test_retuning_performed"]
                else "FAIL"
            ),
            "observed": {
                "final_test_roc_auc": later["final_test_roc_auc"],
                "evaluated_once": later["final_test_evaluated_once"],
                "retuned": later["post_test_retuning_performed"],
            },
            "requirement": "0.606113 retained without post-test retuning",
            "details": "The final model was not forced to pass the generator band.",
        },
        {
            "check": "Data validator uses generator-diagnostic terminology",
            "status": (
                "PASS"
                if 'targets["observable_signal_diagnostic"]'
                in validator_text
                and "not a final-model performance target"
                in validator_text
                and "model_roc_auc" not in validator_text
                else "FAIL"
            ),
            "observed": contract["data_validator"],
            "requirement": "No stale model-target implementation language",
            "details": "Executable output states the diagnostic's limited purpose.",
        },
        {
            "check": "Public methodology explanation is complete",
            "status": (
                "PASS" if not missing_public_phrases else "FAIL"
            ),
            "observed": missing_public_phrases,
            "requirement": "0 missing public evidence phrases",
            "details": "The repository explains the method without a personal script.",
        },
        {
            "check": "Private answer markers are absent from repository",
            "status": (
                "PASS" if not present_private_markers else "FAIL"
            ),
            "observed": present_private_markers,
            "requirement": "0 private first-person script markers",
            "details": "Only public methodology is committed to Git.",
        },
        {
            "check": "Checkpoint leaves analytical results frozen",
            "status": "PASS" if governance_ok else "FAIL",
            "observed": governance,
            "requirement": "Documentation and configuration naming only",
            "details": "No data generation, fitting, test access, or policy change.",
        },
    ]


def save_outputs(
    project_root: Path,
    contract: dict[str, Any],
    checks: list[dict[str, Any]],
) -> Path:
    """Save aggregate governance-validation evidence."""

    output_directory = project_root / contract["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    with (output_directory / "generator_signal_validation.csv").open(
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
        "generator_diagnostic_roc_auc": contract["diagnostic_contract"][
            "accepted_observed"
        ],
        "final_test_roc_auc": contract["later_model_evidence"][
            "final_test_roc_auc"
        ],
        "post_test_retuning": contract["later_model_evidence"][
            "post_test_retuning_performed"
        ],
    }
    (output_directory / "generator_signal_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_directory


def print_results(
    checks: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """Print reviewer-readable governance results."""

    print("\nGENERATOR SIGNAL GOVERNANCE")
    for check in checks:
        print(
            f"{check['status']:<4}  {check['check']:<57}  "
            f"{check['observed']}"
        )
    print(f"\nSaved generator-signal outputs to: {output_directory}")


def main() -> None:
    """Run the generator-signal governance contract."""

    contract = load_yaml(
        PROJECT_ROOT / "config" / "generator_signal_governance.yaml"
    )
    checks = build_checks(PROJECT_ROOT, contract)
    output_directory = save_outputs(PROJECT_ROOT, contract, checks)
    print_results(checks, output_directory)

    failed = [check["check"] for check in checks if check["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Generator-signal governance failed: {failed}")

    print("\nGENERATOR SIGNAL GOVERNANCE VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
