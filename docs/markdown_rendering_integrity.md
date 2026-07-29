# Markdown Rendering Integrity

## Purpose

Checkpoint 58 repairs reviewer-facing Markdown files that ended inside open
fenced code blocks. An unclosed fence can cause GitHub to display the remainder
of a document as code and prevents Mermaid diagrams from rendering.

This checkpoint also adds an automated repository-wide contract so a later
documentation edit cannot silently reintroduce the problem.

## Initial Finding

The seven documents named in the review concern were inspected directly.

Four contained a genuinely unclosed fence:

- `docs/attrition_hazard_design.md`
- `docs/data_model.md`
- `docs/end_to_end_validation.md`
- `docs/portfolio_presentation.md`

Three were already balanced after the earlier results-documentation update:

- `docs/department_history_bias.md`
- `docs/exploratory_analysis.md`
- `docs/generator_audit.md`

All seven remain listed in the regression configuration so future changes to
any of them are tested.

## Repair

Explicit closing delimiters were added to the four open blocks:

| Document | Block type | Repair |
| --- | --- | --- |
| `attrition_hazard_design.md` | Text formula | Close after the annualized-hazard calculation |
| `data_model.md` | Mermaid | Close after the final entity relationship |
| `end_to_end_validation.md` | PowerShell | Close after the end-to-end command |
| `portfolio_presentation.md` | Text pipeline | Close after the Streamlit dashboard stage |

The `data_model.md` repair is especially important. GitHub recognizes Mermaid
only when the diagram is inside a complete `mermaid` fence. The validator also
checks that the block begins with `erDiagram` and retains nine required
relationships.

## Automated Contract

`src/validate_markdown_docs.py` scans:

- The root `README.md`
- Every Markdown file under `docs/`
- `notebooks/README.md`

The parser follows the relevant CommonMark fence rules:

- Backtick and tilde fences are recognized.
- Up to three leading spaces are allowed.
- A closing marker must use the same character as the opener.
- A closing marker must be at least as long as the opener.
- An unfinished block at end of file is reported with its opening line.

The validator saves aggregate evidence to
`data/processed/markdown_integrity/`.

## Local Validation

Run the complete checkpoint:

```powershell
.\scripts\checkpoints\run_checkpoint58.ps1
```

Run only the Markdown validator:

```powershell
python src\validate_markdown_docs.py
```

## Continuous Integration

The GitHub **Python quality** workflow runs the Markdown validator before the
test suite. Pytest independently repeats the contract, including negative
tests that prove the parser catches missing and invalid closing markers.

Therefore, a pull request or push that introduces another unclosed fence will
fail visibly.

## Scope and Governance

Checkpoint 58 changes documentation rendering and quality controls only. It
does not:

- Regenerate synthetic data
- Retrain or recalibrate a model
- Access the reserved final-test target
- Change the frozen retention policy
- Change the dashboard population or current review plan
- Recalculate any financial or fairness result

All analytical conclusions and committed result numbers remain unchanged.
