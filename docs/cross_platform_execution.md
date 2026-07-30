# Cross-Platform Project Execution

## Purpose

Checkpoint 59 replaces the checkpoint-oriented PowerShell surface with one
reviewer-facing runner that works on Windows, macOS, and Linux.

The historical PowerShell scripts remain available under
`scripts/checkpoints/`, but they are no longer presented as the finished
project interface.

## Canonical Entry Point

The canonical runner is:

```bash
python scripts/run_project.py
```

It uses:

- The active Python interpreter from `sys.executable`
- `pathlib` for operating-system-independent paths
- `subprocess.run` with argument lists instead of shell command strings
- Fail-fast return-code handling
- UTF-8 and noninteractive plotting defaults

The runner does not require PowerShell, Bash, Make, or platform-specific path
separators.

## Commands

| Command | Purpose |
| --- | --- |
| `quality` | Dependency validation, Markdown checks, compilation, Ruff, and pytest |
| `validate` | Quality gates plus notebook and README validation |
| `postgres` | Load PostgreSQL, build Version 2 temporal data, validate source parity, and execute SQL/Python equivalence |
| `pipeline` | Complete Version 2 generation, modeling, database, policy, and portfolio pipeline |
| `dashboard` | Launch the Streamlit dashboard |
| `notebooks` | Execute and validate the twelve supporting notebooks |
| `organize-checkpoints` | One-time migration of checkpoint journals |

List all options:

```bash
python scripts/run_project.py --help
python scripts/run_project.py pipeline --help
```

## Pipeline Controls

The default `pipeline` prepares PostgreSQL before Version 2 modeling and builds
the temporal dataset from eight analytical views. It then executes the
independent multi-snapshot SQL and requires complete equivalence with the
Python builder before modeling. The focused command is:

```bash
python scripts/run_project.py postgres
```

Optional flags allow a reviewer to narrow infrastructure work:

```bash
python scripts/run_project.py pipeline --skip-data-generation
python scripts/run_project.py pipeline --skip-postgres
python scripts/run_project.py pipeline --skip-external-benchmark
```

The legacy Version 1 model remains disabled by default. It can be reproduced
only through the explicit `--include-legacy-v1` option.

Use a dry run to inspect every resolved command without changing data:

```bash
python scripts/run_project.py --dry-run pipeline
```

## Platform Setup

Create and activate the locked environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the same verified lock on every platform:

```bash
python -m pip install "pip==26.0.1" "setuptools==83.0.0" "wheel==0.47.0"
python -m pip install --no-build-isolation --require-hashes -r requirements-lock.txt
```

The database stages require a reachable PostgreSQL instance and a local `.env`
file. A reviewer without PostgreSQL can run the explicit file-based fallback
with `--skip-postgres`; that path is parity-tested against the primary
PostgreSQL source contract.

## Windows Compatibility

`scripts/run_end_to_end.ps1` is retained as a thin compatibility wrapper. It
maps existing PowerShell switches to the Python `pipeline` command. The
pipeline definition exists only in `run_project.py`, preventing Windows and
cross-platform paths from drifting apart.

## Checkpoint Archive

The historical `run_checkpointNN.ps1` scripts document how the project was
built. They are now grouped under:

```text
scripts/checkpoints/
```

This preserves the complete development record without making a reviewer
choose among two dozen primary-looking runners.

## Validation

The runner validation verifies:

- All seven commands are exposed.
- PostgreSQL preparation occurs before the Version 2 temporal build.
- The primary temporal builder receives `--source postgresql`.
- The SQL equivalence validator runs after source parity and before modeling.
- The `--skip-postgres` path receives `--source csv`.
- Skip flags remove only their intended stage groups.
- No shell operators or shell executables are used.
- Twelve supporting notebooks come from the committed manifest.
- Twenty-eight checkpoint runners are archived outside the scripts root.
- Archived PowerShell runners resolve the repository root from their new
  two-level location.
- The Windows wrapper delegates to the Python runner.
- Documentation contains no stale root checkpoint paths.

## Scope

This checkpoint changes orchestration and repository organization only. It
does not modify:

- Synthetic data generation logic
- Feature definitions
- Model selection or calibration
- Final-test evidence
- Fairness or policy calculations
- Dashboard data
- Employee review selections

All analytical results remain unchanged.
