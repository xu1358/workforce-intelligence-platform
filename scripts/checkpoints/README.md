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

The newest database-integration journal is
`run_checkpoint62.ps1`. It runs the canonical cross-platform `postgres`
command and then the complete quality suite; the analytical implementation
remains in the reviewer-facing Python and SQL files.
