# Version 2 Department and Location History Audit

## Purpose

This report re-tests Notebook 19's history-attribution question on the
authoritative Version 2 source records and temporal panel.

The generator uses two different rules:

- department is assigned at hire and remains invariant; and
- location may change through dated `Transfer` events.

The audit verifies both rules instead of carrying the Version 1 conclusion
forward without evidence.

## Results Snapshot

| Check | Version 2 result |
| --- | ---: |
| Source employees | 10,000 |
| Hire events with parsed department | 10,000 |
| Hire-to-current department mismatches | 0 |
| Transfer events | 486 |
| Employees with transfer events | 486 |
| Transfer events explicitly representing location | 486 |
| Transfer events explicitly representing department | 0 |
| Temporal-panel employees | 7,745 |
| Employees with multiple panel departments | 0 |
| Historical rows whose reconstructed location differs from current location | 86 |
| Employees represented by those rows | 60 |

The zero department mismatch is not evidence that department history was
successfully reconstructed. It reflects the Version 2 generator contract:
department does not change after hire.

### Transfer semantics

Every one of the 486 `Transfer` events states that `old_value` and `new_value`
represent `location_id`. Treating these records as department changes would
create false history.

The audit parses all 10,000 hire-event notes and finds zero differences between
hire department and the employee source-table department.

### Historical location reconstruction

Location is different. Current location differs from hire location for 107
employees. The temporal builder uses dated transfer events to reconstruct
historical location rather than assigning every historical row to the final
location.

| Snapshot | Historical rows different from current location |
| --- | ---: |
| 2023 | 31 |
| 2024 | 34 |
| 2025 | 21 |
| **Total** | **86** |

These 86 rows cover 60 employees. Without historical reconstruction, those rows
would be attributed to a location that the employee had not yet reached at the
snapshot date.

## What Changed From Version 1

Notebook 19 was a useful Version 1 design audit, but it did not validate the
Version 2 temporal dataset. Checkpoint 66 adds an executable Version 2
successor that:

1. reads the Version 2 source employees and employee events;
2. parses hire departments and locations;
3. classifies transfer-event semantics;
4. checks department continuity across all temporal snapshots;
5. verifies historical location reconstruction; and
6. saves only aggregate, target-free evidence.

The result is structurally similar to Version 1—department remains immutable—
but it is now proven on the data used by the current project.

## Interpretation

The correct claim is:

> Version 2 does not require department-history reconstruction because the
> synthetic generator defines department as time-invariant. Location is
> time-varying, and the temporal builder reconstructs it from dated transfer
> events.

This is a data-model limitation as well as a leakage control. The project cannot
evaluate real department transfers because synthetic department mobility does
not exist. A production implementation would require an effective-dated
department-assignment table or department-change events.

## Governance Boundary

This audit:

- does not access the attrition target;
- does not export employee identifiers;
- does not change source data or temporal features;
- does not change the selected model;
- does not change policy selections; and
- does not make causal department comparisons.

## Evidence and Reproduction

Run the complete checkpoint:

```powershell
.\scripts\checkpoints\run_checkpoint66.ps1
```

Run only the history audit:

```powershell
python src\audit_v2_department_history.py
```

Reviewer-facing evidence:

- [Notebook 38: Version 2 Department History](../notebooks/38_v2_department_history.ipynb)
- configuration: `config/v2_department_history.yaml`
- implementation: `src/audit_v2_department_history.py`
- aggregate outputs: `data/processed/v2_department_history/`

Historical context:

- [Version 1 history report](department_history_bias.md)
- [Version 1 Notebook 19](../notebooks/19_department_history_bias.ipynb)
