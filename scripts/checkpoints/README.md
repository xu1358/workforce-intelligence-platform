# Historical Checkpoint Runners

This folder preserves the Windows PowerShell runners used while the project
was developed checkpoint by checkpoint.

They are retained as an implementation journal and reproducibility record.
They are not the primary way to run the finished project.

For Windows, macOS, or Linux, use:

```bash
python scripts/run_project.py --help
```

The canonical commands and pipeline options are documented in
[`scripts/README.md`](../README.md).

Every archived PowerShell runner resolves the repository root by moving from
this directory to `scripts/` and then to the repository root. This keeps the
historical commands runnable after the Checkpoint 59 reorganization.

The SQL-equivalence journal is `run_checkpoint63.ps1`. It runs the
canonical cross-platform `postgres` command, executes the multi-snapshot SQL,
compares all SQL results with the Python builder, and then runs the complete
quality suite. The analytical implementation remains in the reviewer-facing
Python and SQL files.

Checkpoint 64 audits subgroup allocation with the exact frozen expected-value
policy and executes Notebook 35.

Checkpoint 65 is `run_checkpoint65.ps1`. It quantifies temporal
base-rate drift and prior-probability shift risk, executes Notebook 36, and
then runs the complete quality suite without recalibrating on the final test.

Checkpoint 66 is `run_checkpoint66.ps1`. It replaces the missing Version 2
exploratory evidence, re-tests performance-history missingness on training
data, audits department continuity and location reconstruction, executes
Notebooks 37–38, and then runs the complete quality suite.

Checkpoint 67 is `run_checkpoint67.ps1`. It removes the exact nine stale
Version 1 dashboard screenshots, validates the six live Version 2 tabs and
current dashboard guide, and then runs the complete quality suite.
