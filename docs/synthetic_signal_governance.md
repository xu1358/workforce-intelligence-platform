# Synthetic-Generator Signal Governance

## Purpose

The Version 2 generator contains a development-only ROC-AUC acceptance band of
`0.62–0.75`. This document records the boundary between synthetic
data-generating-process calibration and later model evaluation.

The acceptance band verifies that observable histories contain moderate,
recoverable signal while rejecting an unrealistically easy synthetic outcome.
It does not select the final classifier, use the reserved target, or establish
real-world predictive validity.

## What Actually Happened

| Stage | Purpose | ROC-AUC evidence | Used the reserved test target? |
| --- | --- | ---: | --- |
| First generator diagnostic | Detect whether observable histories carried recoverable signal | Approximately 0.54 | No |
| Accepted generator diagnostic | Approve a moderate, imperfect synthetic relationship | 0.6298 | No |
| 2024 model validation | Compare three fixed candidate classifiers | Selected Logistic Regression: 0.6211 | No |
| 2025 once-only final test | Estimate future-time generalization after decisions were frozen | 0.6061 | Yes, once |

The values answer different questions. The generator diagnostic asks whether
the synthetic mechanism creates a usable analytical problem. The final test
asks whether the selected modeling pipeline generalizes to a later period.

## Why a Lower and an Upper Bound Exist

### Lower bound: reject an empty analytical problem

The first generator produced approximately `0.54` ROC-AUC. That indicated the
observable histories had little relationship with the simulated exits. A
portfolio classifier built on that process would mostly demonstrate that its
inputs were disconnected from its target.

The `0.62` lower bound required nontrivial but modest observable signal. It was
a synthetic-generator acceptance rule, not an expected production score.

### Upper bound: reject an unrealistically easy problem

The `0.75` ceiling is just as important as the lower bound. It prevents the
generator from creating outcomes that are too directly determined by a small
set of visible features.

Additional safeguards limit single-feature correlation and large-group rate
ratios. The accepted diagnostic of `0.6298` sits close to the lower boundary,
not near the upper ceiling.

## What Was Adjusted

The generator calibration changed transparent assumptions in the synthetic
hazard:

- observable performance, compensation, promotion, training, and
  manager-change effects were strengthened;
- unobserved frailty and period noise were reduced but retained; and
- baseline exit probabilities were adjusted to keep the annual outcome rate
  within its separate acceptance range.

These are assumptions about a fictional data-generating process. They are not
estimated causal effects and are not presented as facts about a real employer.

## What Was Not Done

The project did not:

- use the 2025 final-test target to design the generator;
- choose the final classifier using the generator diagnostic;
- require the final model to exceed `0.62`;
- repeatedly tune model hyperparameters against the test period;
- retune after the final ROC-AUC reached `0.6061`; or
- claim that the acceptance band is an external industry benchmark.

The model comparison later used PR-AUC as its primary metric. Logistic
Regression was selected because it had the highest observed validation PR-AUC,
competitive grouped robustness, and the lowest complexity—not because it met
the generator’s ROC-AUC band.

## Why the Final-Test Result Is Important Evidence

The once-only final test ROC-AUC is `0.6061`, below the generator diagnostic’s
`0.62` lower bound. The project reports that result with no retuning and does
not reopen generator design, model selection, calibration, or policy
optimization.

If the purpose had been to manufacture a final score above `0.62`, this result
would have triggered another adjustment. Keeping it unchanged demonstrates the
separation between:

1. constructing a nontrivial synthetic learning problem; and
2. honestly evaluating a frozen modeling pipeline.

## Terminology

Prefer:

- “development-only generator acceptance band”;
- “moderate-signal design check”;
- “synthetic data-generating-process calibration”; and
- “not a final-model performance target.”

The `0.62–0.75` range is not described as a real-world benchmark. The project
does not cite real employer evidence supporting that range. It is an explicit
design choice for this synthetic portfolio.

## Reproduction and Evidence

The implementation is in:

- `config/attrition_hazard_config.yaml`;
- `src/validate_v2_attrition_data.py`;
- `config/generator_signal_governance.yaml`; and
- `src/validate_generator_signal_governance.py`.

The original generator diagnostic evidence is produced under:

```text
data/processed/v2_validation/
```

The final-test evidence is produced under:

```text
data/processed/retention_final_test_model_metrics.csv
```

Run the governance checkpoint on Windows with:

```powershell
.\scripts\checkpoints\run_checkpoint61.ps1
```

No synthetic records, models, policies, or dashboard outputs are regenerated
by this checkpoint.
