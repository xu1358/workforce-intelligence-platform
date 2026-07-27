"""Build transparent retention-cost scenarios without selecting a policy.

Checkpoint 45 converts avoidable employee departures into financial and
operational planning units. It intentionally does not load model
probabilities, inspect the reserved test target, rank employees, or choose
an operating threshold. Checkpoint 46 will use these scenario inputs for
expected-value policy analysis.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CONFIG_PATH = (
    PROJECT_ROOT / "config" / "retention_cost_assumptions.yaml"
)
TEMPORAL_DATA_PATH = (
    PROCESSED_DIR / "retention_multi_snapshot.csv"
)
CURRENT_DATA_PATH = (
    PROCESSED_DIR / "current_active_scoring_population.csv"
)
ASSIGNMENT_PATH = (
    PROCESSED_DIR / "model_split_assignments.csv"
)

SCENARIO_OUTPUT_PATH = (
    PROCESSED_DIR / "retention_cost_scenarios.csv"
)
POPULATION_OUTPUT_PATH = (
    PROCESSED_DIR / "retention_cost_population_summary.csv"
)
ASSUMPTION_OUTPUT_PATH = (
    PROCESSED_DIR / "retention_cost_assumption_table.csv"
)
VALIDATION_OUTPUT_PATH = (
    PROCESSED_DIR / "retention_cost_validation.csv"
)

FEATURE_COLUMNS = [
    "employee_id",
    "snapshot_date",
    "organizational_level",
    "department_name",
    "employment_type",
    "job_level",
    "base_salary",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one required YAML mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")

    return loaded


def require_files(paths: list[Path]) -> None:
    """Raise one readable error when required inputs are missing."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Checkpoint 45 input files:\n"
            + "\n".join(missing)
        )


def load_populations(
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, int, int]:
    """Load validation and current features without any outcome column."""

    require_files(
        [
            TEMPORAL_DATA_PATH,
            CURRENT_DATA_PATH,
            ASSIGNMENT_PATH,
        ]
    )

    # Only these explicitly named feature columns are read. The attrition
    # target is never loaded into memory by this checkpoint.
    temporal = pd.read_csv(
        TEMPORAL_DATA_PATH,
        usecols=FEATURE_COLUMNS,
    )
    current = pd.read_csv(
        CURRENT_DATA_PATH,
        usecols=FEATURE_COLUMNS,
    )
    assignments = pd.read_csv(
        ASSIGNMENT_PATH,
        usecols=[
            "employee_id",
            "snapshot_date",
            "primary_split",
            "reserved_test",
        ],
    )

    test_rows = int(assignments["primary_split"].eq("test").sum())
    validation_assignments = assignments.loc[
        assignments["primary_split"].eq("validation"),
        ["employee_id", "snapshot_date"],
    ].copy()

    validation = temporal.merge(
        validation_assignments,
        on=["employee_id", "snapshot_date"],
        how="inner",
        validate="one_to_one",
    )

    validation_snapshot = str(
        policy["population"]["validation_snapshot"]
    )
    current_snapshot = str(
        policy["population"]["current_snapshot"]
    )
    validation = validation.loc[
        validation["snapshot_date"].astype(str).eq(
            validation_snapshot
        )
    ].copy()
    current = current.loc[
        current["snapshot_date"].astype(str).eq(current_snapshot)
    ].copy()

    eligible_levels = set(
        policy["population"]["eligible_organizational_levels"]
    )
    validation = validation.loc[
        validation["organizational_level"].isin(eligible_levels)
    ].copy()
    current = current.loc[
        current["organizational_level"].isin(eligible_levels)
    ].copy()

    validation["population"] = "Validation"
    current["population"] = "Current active"

    populations = pd.concat(
        [validation, current],
        ignore_index=True,
    )
    populations["snapshot_date"] = populations[
        "snapshot_date"
    ].astype(str)
    populations["base_salary"] = pd.to_numeric(
        populations["base_salary"],
        errors="raise",
    )

    return populations, test_rows, 0


def build_assumption_table(
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Create a human-readable table of committed assumptions."""

    rows: list[dict[str, Any]] = []

    for scenario in policy["replacement_impact_scenarios"]:
        coverage_hours = (
            float(scenario["vacancy_days"])
            * float(scenario["coverage_hours_per_vacancy_day"])
        )
        rows.append(
            {
                "assumption_family": "Replacement impact",
                "scenario": str(scenario["name"]),
                "salary_multiplier": float(
                    scenario["salary_multiplier"]
                ),
                "intervention_cost_usd": np.nan,
                "success_probability": np.nan,
                "vacancy_days": int(scenario["vacancy_days"]),
                "time_to_productivity_days": int(
                    scenario["time_to_productivity_days"]
                ),
                "training_hours_per_replacement": float(
                    scenario["training_hours_per_replacement"]
                ),
                "coverage_hours_per_replacement": coverage_hours,
            }
        )

    for scenario in policy["intervention_cost_scenarios"]:
        rows.append(
            {
                "assumption_family": "Intervention cost",
                "scenario": str(scenario["name"]),
                "salary_multiplier": np.nan,
                "intervention_cost_usd": float(
                    scenario["cost_per_employee_usd"]
                ),
                "success_probability": np.nan,
                "vacancy_days": np.nan,
                "time_to_productivity_days": np.nan,
                "training_hours_per_replacement": np.nan,
                "coverage_hours_per_replacement": np.nan,
            }
        )

    for scenario in policy[
        "intervention_effectiveness_scenarios"
    ]:
        rows.append(
            {
                "assumption_family": "Intervention effectiveness",
                "scenario": str(scenario["name"]),
                "salary_multiplier": np.nan,
                "intervention_cost_usd": np.nan,
                "success_probability": float(
                    scenario["success_probability"]
                ),
                "vacancy_days": np.nan,
                "time_to_productivity_days": np.nan,
                "training_hours_per_replacement": np.nan,
                "coverage_hours_per_replacement": np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_scenario_table(
    populations: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate every cost-assumption combination by population."""

    replacement_scenarios = policy[
        "replacement_impact_scenarios"
    ]
    intervention_scenarios = policy[
        "intervention_cost_scenarios"
    ]
    effectiveness_scenarios = policy[
        "intervention_effectiveness_scenarios"
    ]

    rows: list[dict[str, Any]] = []

    for population_name, population in populations.groupby(
        "population",
        sort=False,
    ):
        salary = population["base_salary"].to_numpy(dtype=float)

        for replacement, intervention, effectiveness in product(
            replacement_scenarios,
            intervention_scenarios,
            effectiveness_scenarios,
        ):
            multiplier = float(replacement["salary_multiplier"])
            intervention_cost = float(
                intervention["cost_per_employee_usd"]
            )
            success_probability = float(
                effectiveness["success_probability"]
            )

            replacement_cost = salary * multiplier
            maximum_avoidable_cost = (
                replacement_cost * success_probability
            )
            break_even_probability = np.divide(
                intervention_cost,
                maximum_avoidable_cost,
                out=np.full(
                    maximum_avoidable_cost.shape,
                    np.nan,
                    dtype=float,
                ),
                where=maximum_avoidable_cost > 0,
            )
            coverage_hours = (
                float(replacement["vacancy_days"])
                * float(
                    replacement[
                        "coverage_hours_per_vacancy_day"
                    ]
                )
            )

            scenario_id = "-".join(
                [
                    str(replacement["name"]).lower(),
                    str(intervention["name"]).lower(),
                    str(effectiveness["name"]).lower(),
                ]
            )

            rows.append(
                {
                    "population": population_name,
                    "snapshot_date": str(
                        population["snapshot_date"].iloc[0]
                    ),
                    "scenario_id": scenario_id,
                    "replacement_impact_scenario": str(
                        replacement["name"]
                    ),
                    "intervention_cost_scenario": str(
                        intervention["name"]
                    ),
                    "effectiveness_scenario": str(
                        effectiveness["name"]
                    ),
                    "population_rows": len(population),
                    "average_base_salary": float(salary.mean()),
                    "median_base_salary": float(
                        np.median(salary)
                    ),
                    "replacement_cost_salary_multiplier": multiplier,
                    "average_replacement_cost_usd": float(
                        replacement_cost.mean()
                    ),
                    "median_replacement_cost_usd": float(
                        np.median(replacement_cost)
                    ),
                    "intervention_cost_per_employee_usd": (
                        intervention_cost
                    ),
                    "intervention_success_probability": (
                        success_probability
                    ),
                    "average_maximum_avoidable_cost_usd": float(
                        maximum_avoidable_cost.mean()
                    ),
                    "median_maximum_avoidable_cost_usd": float(
                        np.median(maximum_avoidable_cost)
                    ),
                    "average_break_even_attrition_probability": (
                        float(np.nanmean(break_even_probability))
                    ),
                    "median_break_even_attrition_probability": (
                        float(np.nanmedian(break_even_probability))
                    ),
                    "share_break_even_at_or_below_100_percent": (
                        float(
                            np.mean(
                                break_even_probability <= 1.0
                            )
                        )
                    ),
                    "vacancy_days_per_prevented_departure": int(
                        replacement["vacancy_days"]
                    ),
                    "time_to_productivity_days_per_replacement": (
                        int(
                            replacement[
                                "time_to_productivity_days"
                            ]
                        )
                    ),
                    "training_hours_per_replacement": float(
                        replacement[
                            "training_hours_per_replacement"
                        ]
                    ),
                    "coverage_hours_per_prevented_departure": (
                        coverage_hours
                    ),
                    "model_probabilities_used": False,
                    "operating_policy_selected": False,
                    "reserved_test_outcomes_used": False,
                }
            )

    scenarios = pd.DataFrame(rows)
    return scenarios.sort_values(
        [
            "population",
            "replacement_impact_scenario",
            "intervention_cost_scenario",
            "effectiveness_scenario",
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_population_summary(
    populations: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Summarize salaries and reference-scenario replacement costs."""

    reference = policy["reference_scenario"]
    replacement = next(
        item
        for item in policy["replacement_impact_scenarios"]
        if item["name"] == reference["replacement_impact"]
    )
    multiplier = float(replacement["salary_multiplier"])

    summary = (
        populations.groupby(
            ["population", "snapshot_date", "department_name"],
            as_index=False,
            observed=True,
        )
        .agg(
            employees=("employee_id", "size"),
            average_base_salary=("base_salary", "mean"),
            median_base_salary=("base_salary", "median"),
        )
    )
    summary["reference_replacement_cost_multiplier"] = multiplier
    summary["average_reference_replacement_cost_usd"] = (
        summary["average_base_salary"] * multiplier
    )
    summary["median_reference_replacement_cost_usd"] = (
        summary["median_base_salary"] * multiplier
    )
    return summary


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    requirement: Any,
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


def validate_outputs(
    populations: pd.DataFrame,
    scenarios: pd.DataFrame,
    assumptions: pd.DataFrame,
    policy: dict[str, Any],
    reserved_test_rows: int,
    reserved_test_targets_used: int,
) -> pd.DataFrame:
    """Validate formulas, scope, scenarios, and policy boundaries."""

    checks: list[dict[str, Any]] = []
    validation_policy = policy["validation"]
    tolerance = float(validation_policy["numeric_tolerance"])

    replacement = policy["replacement_impact_scenarios"]
    intervention = policy["intervention_cost_scenarios"]
    effectiveness = policy[
        "intervention_effectiveness_scenarios"
    ]

    replacement_names = [item["name"] for item in replacement]
    intervention_names = [item["name"] for item in intervention]
    effectiveness_names = [item["name"] for item in effectiveness]

    add_check(
        checks,
        "Replacement scenarios are complete and unique",
        (
            len(replacement)
            == int(
                validation_policy[
                    "expected_replacement_scenarios"
                ]
            )
            and len(set(replacement_names)) == len(replacement_names)
        ),
        replacement_names,
        (
            f"{validation_policy['expected_replacement_scenarios']} "
            "unique scenarios"
        ),
        "Low, base, and high replacement impacts must all be explicit.",
    )
    add_check(
        checks,
        "Intervention cost scenarios are complete and unique",
        (
            len(intervention)
            == int(
                validation_policy[
                    "expected_intervention_cost_scenarios"
                ]
            )
            and len(set(intervention_names)) == len(intervention_names)
        ),
        intervention_names,
        (
            f"{validation_policy['expected_intervention_cost_scenarios']} "
            "unique scenarios"
        ),
        "The model must compare more than one intervention intensity.",
    )
    add_check(
        checks,
        "Effectiveness scenarios are complete and unique",
        (
            len(effectiveness)
            == int(
                validation_policy[
                    "expected_effectiveness_scenarios"
                ]
            )
            and len(set(effectiveness_names))
            == len(effectiveness_names)
        ),
        effectiveness_names,
        (
            f"{validation_policy['expected_effectiveness_scenarios']} "
            "unique scenarios"
        ),
        "Uncertain intervention success must be handled by scenarios.",
    )

    multipliers = [
        float(item["salary_multiplier"])
        for item in replacement
    ]
    expected_minimum = float(
        validation_policy["minimum_salary_multiplier"]
    )
    expected_maximum = float(
        validation_policy["maximum_salary_multiplier"]
    )
    add_check(
        checks,
        "Replacement-cost range spans 50% to 150% of salary",
        (
            abs(min(multipliers) - expected_minimum) <= tolerance
            and abs(max(multipliers) - expected_maximum)
            <= tolerance
        ),
        f"{min(multipliers):.2f}–{max(multipliers):.2f}",
        f"{expected_minimum:.2f}–{expected_maximum:.2f}",
        "The committed sensitivity range matches the revision plan.",
    )

    costs = [
        float(item["cost_per_employee_usd"])
        for item in intervention
    ]
    add_check(
        checks,
        "Intervention costs are positive and ordered",
        all(cost > 0 for cost in costs)
        and costs == sorted(costs),
        costs,
        "Positive increasing costs",
        "Every intervention has an explicit per-employee dollar cost.",
    )

    probabilities = [
        float(item["success_probability"])
        for item in effectiveness
    ]
    add_check(
        checks,
        "Effectiveness assumptions are valid probabilities",
        all(0 < value <= 1 for value in probabilities)
        and probabilities == sorted(probabilities),
        probabilities,
        "Increasing values inside (0, 1]",
        "Effectiveness is uncertain and cannot exceed certainty.",
    )

    replacement_ordered = all(
        replacement[index][field]
        <= replacement[index + 1][field]
        for index in range(len(replacement) - 1)
        for field in [
            "salary_multiplier",
            "vacancy_days",
            "time_to_productivity_days",
            "training_hours_per_replacement",
            "coverage_hours_per_vacancy_day",
        ]
    )
    add_check(
        checks,
        "Operational assumptions increase by impact scenario",
        replacement_ordered,
        replacement_names,
        "Nondecreasing Low to Base to High",
        "More severe scenarios should not assume smaller disruption.",
    )

    population_counts = (
        populations.groupby("population").size().to_dict()
    )
    minimum_rows = int(validation_policy["minimum_population_rows"])
    add_check(
        checks,
        "Economic populations are sufficiently large",
        all(count >= minimum_rows for count in population_counts.values()),
        population_counts,
        f"Each population >= {minimum_rows}",
        "Cost summaries need useful validation and current populations.",
    )

    expected_snapshots = {
        "Validation": str(
            policy["population"]["validation_snapshot"]
        ),
        "Current active": str(
            policy["population"]["current_snapshot"]
        ),
    }
    observed_snapshots = (
        populations.groupby("population")["snapshot_date"]
        .unique()
        .apply(lambda values: sorted(values.tolist()))
        .to_dict()
    )
    snapshots_match = all(
        observed_snapshots.get(name) == [snapshot]
        for name, snapshot in expected_snapshots.items()
    )
    add_check(
        checks,
        "Only configured economic snapshots are used",
        snapshots_match,
        observed_snapshots,
        expected_snapshots,
        "Validation economics precede current operational planning.",
    )

    duplicates = int(
        populations.duplicated(
            ["population", "employee_id", "snapshot_date"]
        ).sum()
    )
    add_check(
        checks,
        "Population employee-snapshot keys are unique",
        duplicates == 0,
        duplicates,
        "0 duplicates",
        "One employee contributes once to each economic population.",
    )

    eligible_levels = set(
        policy["population"]["eligible_organizational_levels"]
    )
    observed_levels = set(populations["organizational_level"])
    add_check(
        checks,
        "Cost population matches model-eligible levels",
        observed_levels == eligible_levels,
        sorted(observed_levels),
        sorted(eligible_levels),
        "Protected hierarchy levels remain outside intervention modeling.",
    )

    invalid_salaries = int(
        (
            populations["base_salary"].isna()
            | ~np.isfinite(populations["base_salary"])
            | populations["base_salary"].le(0)
        ).sum()
    )
    add_check(
        checks,
        "Base salaries are complete, finite, and positive",
        invalid_salaries == 0,
        invalid_salaries,
        "0 invalid salaries",
        "Salary-dependent replacement costs require valid salaries.",
    )

    expected_combinations = int(
        validation_policy["expected_combined_scenarios"]
    )
    scenario_counts = (
        scenarios.groupby("population")["scenario_id"]
        .nunique()
        .to_dict()
    )
    add_check(
        checks,
        "Full scenario cross-product is present",
        all(
            count == expected_combinations
            for count in scenario_counts.values()
        ),
        scenario_counts,
        f"{expected_combinations} per population",
        "No favorable or unfavorable assumption combination is omitted.",
    )

    replacement_difference = (
        scenarios["average_replacement_cost_usd"]
        - (
            scenarios["average_base_salary"]
            * scenarios[
                "replacement_cost_salary_multiplier"
            ]
        )
    ).abs().max()
    add_check(
        checks,
        "Replacement-cost formula reconciles",
        float(replacement_difference) <= tolerance,
        float(replacement_difference),
        f"<= {tolerance}",
        "Average replacement cost equals salary times multiplier.",
    )

    avoidable_difference = (
        scenarios["average_maximum_avoidable_cost_usd"]
        - (
            scenarios["average_replacement_cost_usd"]
            * scenarios[
                "intervention_success_probability"
            ]
        )
    ).abs().max()
    add_check(
        checks,
        "Maximum avoidable-cost formula reconciles",
        float(avoidable_difference) <= tolerance,
        float(avoidable_difference),
        f"<= {tolerance}",
        "The value conditional on departure reflects intervention success.",
    )

    break_even_expected = (
        scenarios["intervention_cost_per_employee_usd"]
        / scenarios["median_maximum_avoidable_cost_usd"]
    )
    break_even_difference = (
        scenarios["median_break_even_attrition_probability"]
        - break_even_expected
    ).abs().max()
    add_check(
        checks,
        "Median break-even formula reconciles",
        float(break_even_difference) <= tolerance,
        float(break_even_difference),
        f"<= {tolerance}",
        "Break-even risk is cost divided by maximum avoidable value.",
    )

    operational_difference = assumptions.loc[
        assumptions["assumption_family"].eq("Replacement impact"),
        "coverage_hours_per_replacement",
    ].to_numpy(dtype=float) - (
        assumptions.loc[
            assumptions["assumption_family"].eq(
                "Replacement impact"
            ),
            "vacancy_days",
        ].to_numpy(dtype=float)
        * np.array(
            [
                float(item["coverage_hours_per_vacancy_day"])
                for item in replacement
            ]
        )
    )
    add_check(
        checks,
        "Coverage-hour formula reconciles",
        float(np.abs(operational_difference).max()) <= tolerance,
        float(np.abs(operational_difference).max()),
        f"<= {tolerance}",
        "Coverage hours equal vacancy days times daily coverage hours.",
    )

    target_column_loaded = (
        "attrition_next_12m" in populations.columns
    )
    add_check(
        checks,
        "Reserved test outcomes remain unused",
        (
            reserved_test_rows > 0
            and reserved_test_targets_used == 0
            and not target_column_loaded
        ),
        (
            f"{reserved_test_rows} reserved rows; "
            f"{reserved_test_targets_used} target values used"
        ),
        "Reserved rows > 0 and target usage = 0",
        "Only explicitly named non-target feature columns are loaded.",
    )

    prediction_columns = {
        "attrition_probability",
        "predicted_probability",
        "risk_score",
    }
    predictions_used = bool(
        prediction_columns.intersection(populations.columns)
    ) or bool(scenarios["model_probabilities_used"].any())
    add_check(
        checks,
        "Model probabilities are not used",
        not predictions_used,
        predictions_used,
        False,
        "Checkpoint 45 defines economics before policy optimization.",
    )

    policy_selected = bool(
        scenarios["operating_policy_selected"].any()
    ) or bool(
        policy["policy_boundary"]["select_operating_threshold"]
    )
    add_check(
        checks,
        "No operating threshold or employee ranking is selected",
        (
            not policy_selected
            and not bool(policy["policy_boundary"]["rank_employees"])
        ),
        policy_selected,
        False,
        "Checkpoint 46 will compare policies using explicit costs.",
    )

    operational_separate = bool(
        policy["economic_definition"][
            "operational_units_are_separate"
        ]
    )
    add_check(
        checks,
        "Operational units remain separate from dollars",
        operational_separate,
        operational_separate,
        True,
        "Unpriced disruption measures are not silently double counted.",
    )

    validation = pd.DataFrame(checks)
    failed = validation.loc[validation["status"].eq("FAIL")]
    if not failed.empty:
        raise ValueError(
            "Checkpoint 45 validation failed:\n"
            + failed.to_string(index=False)
        )

    return validation


def select_reference_rows(
    scenarios: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Return the configured reference scenario for reporting."""

    reference = policy["reference_scenario"]
    return scenarios.loc[
        scenarios["replacement_impact_scenario"].eq(
            reference["replacement_impact"]
        )
        & scenarios["intervention_cost_scenario"].eq(
            reference["intervention_cost"]
        )
        & scenarios["effectiveness_scenario"].eq(
            reference["intervention_effectiveness"]
        )
    ].copy()


def print_outputs(
    assumptions: pd.DataFrame,
    scenarios: pd.DataFrame,
    validation: pd.DataFrame,
    policy: dict[str, Any],
) -> None:
    """Print concise, beginner-readable checkpoint results."""

    reference = select_reference_rows(scenarios, policy)
    replacement_assumptions = assumptions.loc[
        assumptions["assumption_family"].eq("Replacement impact")
    ].copy()

    reference_columns = [
        "population",
        "snapshot_date",
        "population_rows",
        "average_base_salary",
        "median_replacement_cost_usd",
        "intervention_cost_per_employee_usd",
        "intervention_success_probability",
        "median_maximum_avoidable_cost_usd",
        "median_break_even_attrition_probability",
    ]
    operational_columns = [
        "scenario",
        "salary_multiplier",
        "vacancy_days",
        "time_to_productivity_days",
        "training_hours_per_replacement",
        "coverage_hours_per_replacement",
    ]

    print("\nRETENTION COST ASSUMPTIONS")
    print(assumptions.to_string(index=False))

    print("\nREFERENCE ECONOMIC SCENARIO")
    print(reference[reference_columns].to_string(index=False))

    print("\nOPERATIONAL UNITS PER PREVENTED DEPARTURE")
    print(
        replacement_assumptions[operational_columns].to_string(
            index=False
        )
    )

    print("\nCOST MODEL VALIDATION")
    print(validation.to_string(index=False))

    print(f"\nSaved cost scenarios: {SCENARIO_OUTPUT_PATH}")
    print(
        "Reserved test outcomes used: 0; "
        "operating policies selected: 0"
    )
    print(
        "\nRETENTION COST MODEL COMPLETED SUCCESSFULLY"
    )


def main() -> None:
    """Run Checkpoint 45 cost-model construction and validation."""

    policy = load_yaml(CONFIG_PATH)
    populations, test_rows, test_targets_used = load_populations(
        policy
    )
    assumptions = build_assumption_table(policy)
    scenarios = build_scenario_table(populations, policy)
    population_summary = build_population_summary(
        populations,
        policy,
    )
    validation = validate_outputs(
        populations,
        scenarios,
        assumptions,
        policy,
        test_rows,
        test_targets_used,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    assumptions.to_csv(ASSUMPTION_OUTPUT_PATH, index=False)
    scenarios.to_csv(SCENARIO_OUTPUT_PATH, index=False)
    population_summary.to_csv(
        POPULATION_OUTPUT_PATH,
        index=False,
    )
    validation.to_csv(VALIDATION_OUTPUT_PATH, index=False)

    print_outputs(
        assumptions,
        scenarios,
        validation,
        policy,
    )


if __name__ == "__main__":
    main()
