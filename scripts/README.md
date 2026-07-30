# Project Runners

This directory exposes one reviewer-facing, cross-platform entry point:

```text
scripts/run_project.py
```

It uses the active Python interpreter and subprocess argument lists rather
than platform-specific shell syntax. The same commands therefore work on
Windows, macOS, and Linux.

## Main Commands

Run fast engineering checks:

```bash
python scripts/run_project.py quality
```

Run quality checks plus portfolio notebook and README validation:

```bash
python scripts/run_project.py validate
```

Run the complete Version 2 pipeline:

```bash
python scripts/run_project.py pipeline
```

Load PostgreSQL, build the Version 2 temporal datasets from analytical views,
and validate database/CSV parity:

```bash
python scripts/run_project.py postgres
```

Launch the dashboard:

```bash
python scripts/run_project.py dashboard
```

Execute reviewer-visible supporting notebooks:

```bash
python scripts/run_project.py notebooks
```

Preview resolved pipeline steps without executing them:

```bash
python scripts/run_project.py --dry-run pipeline
```

## Pipeline Options

The `pipeline` command supports:

| Option | Effect |
| --- | --- |
| `--skip-quality` | Skip compile, lint, formatting, and tests |
| `--skip-data-generation` | Reuse existing generated CSV files |
| `--skip-postgres` | Use the parity-tested CSV fallback instead of the primary PostgreSQL source |
| `--include-legacy-v1` | Rebuild deprecated Version 1 outputs |
| `--skip-external-benchmark` | Skip the isolated IBM benchmark |

The default remains the complete Version 2 pipeline. The legacy Version 1
path remains off unless explicitly requested.

## Windows Compatibility Wrapper

`run_end_to_end.ps1` remains as a small Windows compatibility wrapper. It
translates PowerShell switches into arguments for `run_project.py`; it no
longer maintains a second pipeline definition.

## Checkpoint History

`checkpoints/` contains the historical checkpoint-specific PowerShell runners.
They are retained for auditability and step-by-step reconstruction, but they
are not the primary reviewer interface.

The checkpoint files are Windows-specific because they document the original
development workflow. New project execution should use `run_project.py`.
