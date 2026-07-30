# Workforce Intelligence Platform — Version 2 Revision Roadmap

## Purpose

Version 1 of the Workforce Intelligence Platform was completed after Checkpoint
30 and remains preserved with the Git tag `v1.0-portfolio`.

Version 2 strengthened the project's analytical validity, operational
usefulness, engineering quality, and portfolio presentation. Development took
place on the `revision-v2` branch, and the validated Version 2 release is now
published on `master`. The Version 1 tag remains recoverable independently of
the current default branch.

## Revision Principles

The Version 2 work will follow these principles:

1. Correct the synthetic data-generating process before rebuilding models.
2. Prevent temporal and evaluation leakage.
3. Separate model development decisions from final test evaluation.
4. Evaluate ranking, calibration, uncertainty, and subgroup performance.
5. Translate predictions into constrained operational decisions.
6. Separate historical model evaluation from current workforce scoring.
7. Clearly identify synthetic assumptions and avoid causal claims.
8. Preserve reproducibility through automated tests and documentation.

## Phase 1 — Preserve and Diagnose Version 1

- **Checkpoint 31:** Preserve the Version 1 baseline and create the Version 2 branch.
- **Checkpoint 32:** Audit the current termination data-generating process.
- **Checkpoint 33:** Add substantive exploratory data analysis.
- **Checkpoint 34:** Quantify department-history attribution bias.

## Phase 2 — Rebuild the Synthetic Workforce Process

- **Checkpoint 35:** Design a realistic time-varying attrition hazard.
- **Checkpoint 36:** Implement hazard-based termination generation.
- **Checkpoint 37:** Regenerate and validate the Version 2 data.
- **Checkpoint 38:** Build multi-snapshot temporal modeling datasets.

## Phase 3 — Rebuild Modeling and Evaluation

- **Checkpoint 39:** Correct redundant feature encodings and diagnose collinearity.
- **Checkpoint 40:** Introduce separate training, validation, and test periods.
- **Checkpoint 41:** Rebuild model comparison with cross-validation and uncertainty estimates.
- **Checkpoint 42:** Add probability-calibration analysis.
- **Checkpoint 43:** Add ranking metrics and model-evaluation visualizations.
- **Checkpoint 44:** Add fairness and subgroup diagnostics.

## Phase 4 — Build the Operational Decision System

- **Checkpoint 45:** Define the retention cost model.
- **Checkpoint 46:** Replace fixed thresholding with expected-value optimization and build the top-700 intervention scenario.
- **Checkpoint 47:** Separate historical evaluation from current workforce scoring in the dashboard.
- **Checkpoint 48:** Reframe the project around manufacturing workforce stability and backfill planning.

### Execution-number note

The expected-value optimization and top-700 scenario were completed together
in Checkpoint 46. The remaining checkpoint numbers were shifted forward by one
so the roadmap matches the implemented Git history.

## Phase 5 — Improve Engineering and Portfolio Quality

- **Checkpoint 49:** Add automated tests.
- **Checkpoint 50:** Add code-quality tools and GitHub Actions.
- **Checkpoint 51:** Consolidate the notebooks into a curated portfolio sequence.
- **Checkpoint 52:** Rewrite the README around findings and operational decisions.
- **Checkpoint 53:** Add the isolated IBM employee-attrition methodological
  benchmark with pinned provenance, repeated out-of-fold evaluation,
  calibration, subgroup diagnostics, and explicit non-comparability controls.
- **Checkpoint 54:** Add exact aggregate linear-SHAP explanations and
  employee-grouped explanation-stability checks without changing the final
  test or frozen policy.
- **Checkpoint 55:** Add censoring-aware Kaplan–Meier retention curves,
  baseline-at-hire Cox hazard associations, proportional-hazards diagnostics,
  and held-out concordance checks without changing the primary model or
  policy.
- **Checkpoint 56:** Audit salary-driven allocation in the frozen expected-value
  policy, compare salary-neutral and salary-capped sensitivities, and preserve
  the no-post-test-retuning boundary.
- **Checkpoint 57:** Pin Python and pip in CI, commit exact direct dependencies,
  generate a SHA-256-verified transitive lock, and validate installed-version
  drift locally and remotely.
- **Checkpoint 58:** Repair reviewer-facing Markdown fences, restore GitHub
  Mermaid rendering, and add local and CI regression validation.
- **Checkpoint 59:** Replace the checkpoint-oriented scripts surface with one
  cross-platform Python runner and archive the PowerShell development history.
- **Checkpoint 60:** Convert every attrition-hazard simulation setting into an
  executable, tested runtime contract and repair archived-runner root
  resolution without changing generated or downstream results.
- **Checkpoint 61:** Separate the synthetic-generator signal acceptance check
  from model performance and document the design chronology without reopening
  any analytical result.
- **Checkpoint 62:** Make PostgreSQL the primary Version 2 temporal-data source,
  add least-privilege analytical views, and enforce database/CSV and frozen
  output parity before downstream modeling.
- **Checkpoint 63:** Execute the independent multi-snapshot SQL in a read-only
  transaction and require complete row-, column-, null-, value-, and
  fingerprint-equivalence with the Python temporal builder before modeling.
- **Checkpoint 64:** Separate probability-ranking subgroup diagnostics from
  deployed-policy fairness and audit the exact frozen expected-value
  selections with the shared Checkpoint 46 implementation.
- **Checkpoint 65:** Quantify temporal base-rate drift, document
  prior-probability shift risk and calibration transport, and preserve the
  no-final-test-recalibration boundary.
- **Checkpoint 66:** Replace Version 1-only exploratory evidence with
  training-safe Version 2 EDA, re-test performance-history missingness, and
  audit department continuity plus dated location reconstruction.
- **Checkpoint 67:** Remove the nine stale Version 1 dashboard screenshots,
  document the six-tab Version 2 interface, and enforce screenshot freshness
  in local and GitHub quality gates.

## Optional Advanced Extensions

- **Future option:** Deploy a public demonstration dashboard if a hosted demo
  is later required.

## Primary Version 2 Decision Question

> Given a limited intervention budget, which active employees should receive a retention intervention, and what is the expected operational and financial value?

The initial scenario will evaluate a capacity of 700 interventions and compare the model-based policy with simpler selection strategies.

## External Benchmark

The primary project will remain a synthetic, temporal workforce-intelligence platform.

A separate benchmark track uses the fictional employee-attrition dataset
distributed through IBM's archived `employee-attrition-aif360` repository.
The source is pinned and checksum verified. The benchmark is not merged with
the primary synthetic database or described as real IBM employee data.

## Version 2 Completion Standard

The revised project should:

- Use a documented, hazard-based synthetic termination process.
- Maintain strict temporal feature and target boundaries.
- Preserve an untouched final test period.
- Report model uncertainty, ranking performance, and calibration.
- Include subgroup and fairness diagnostics.
- Convert risk scores into an explicit intervention policy.
- Separate historical evaluation from current scoring.
- Include repeatable automated tests.
- Clearly disclose assumptions, limitations, and ethical considerations.
- Remain understandable and reproducible for technical reviewers.
