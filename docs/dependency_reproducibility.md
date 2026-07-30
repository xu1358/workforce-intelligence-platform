# Dependency Reproducibility

## Result

Checkpoint 57 closes the project's floating-dependency gap.

The tested Python environment now has:

- Python 3.12 as the supported runtime generation;
- Python 3.12.4 as the exact GitHub Actions interpreter;
- pip 26.0.1 as the exact CI resolver;
- setuptools 83.0.0 and wheel 0.47.0 as the source-build toolchain;
- 19 exactly pinned direct dependencies;
- 113 exactly pinned direct and transitive distributions;
- at least one SHA-256 artifact hash for every locked distribution; and
- automated comparison between the installed environment and the active lock.

The full automated test suite passes under this locked environment.

## Why This Was Necessary

The previous `requirements.txt` contained bare declarations such as:

```text
numpy
pandas
scikit-learn
Faker
plotly
streamlit
```

Those declarations asked pip to select the newest compatible versions at
installation time. Two installations performed on different dates could
therefore receive different code even when the Git commit and synthetic data
were identical.

This was inconsistent with the project's broader reproducibility controls,
including deterministic random seeds, data fingerprints, a frozen model
policy, and once-only final-test governance.

The risk is practical rather than theoretical:

- pandas 3 changes important data-frame behavior, including copy-on-write
  semantics;
- scikit-learn preprocessing and encoder interfaces have changed across
  releases;
- plotting and dashboard libraries can change serialization or rendering;
- test and lint tools can introduce new checks; and
- transitive packages can break an environment even when the top-level
  package version did not change.

## Two-Layer Dependency Design

### Direct dependency manifest

`requirements.txt` is the human-reviewed compatibility set. Every declaration
uses one exact version.

Important modeling and presentation versions include:

| Dependency | Committed version | Role |
| --- | ---: | --- |
| numpy | 2.3.5 | Numerical arrays and calculations |
| pandas | 2.2.3 | Data preparation and tabular outputs |
| scikit-learn | 1.8.0 | Preprocessing, models, calibration, and metrics |
| lifelines | 0.30.3 | Kaplan–Meier and Cox survival analysis |
| Faker | 40.36.0 | Synthetic reference data |
| matplotlib | 3.10.8 | Static analytical figures |
| plotly | 6.9.0 | Interactive visualizations |
| streamlit | 1.60.0 | Portfolio dashboard |
| pytest | 9.1.1 | Automated regression tests |
| ruff | 0.16.0 | Linting and formatting validation |

The pandas 2.2.3 pin deliberately prevents an unreviewed pandas 3 upgrade.
The scikit-learn 1.8.0 pin similarly freezes the preprocessing and encoder
behavior exercised by the current tests.

### Complete transitive lock

`requirements-lock.txt` is the machine-installable resolution. It includes
the direct packages plus dependencies such as SciPy, PyArrow, Formulaic,
Jinja2, Pillow, and their supporting libraries.

Every distribution has:

1. one exact `==` version;
2. environment markers where a dependency is platform-specific; and
3. one or more approved SHA-256 package hashes.

The lock is universal across the supported Python 3.12 Windows, Linux, and
macOS environments. Platform markers prevent an operating-system-specific
package from being installed where it does not apply.

## Fingerprints

The committed dependency contract stores fingerprints for both dependency
files:

| File | SHA-256 |
| --- | --- |
| `requirements.txt` | `c04380f6d8435ecc20b938b9fa1e380e5bc89e873b888ef1db9f43e65dbe0771` |
| `requirements-lock.txt` | `aaa6e316e31d65d121ccc6b3a3c3f3bf3bc114626ef416a1f237698b093401a8` |

`src/validate_dependency_environment.py` recalculates both fingerprints after
canonicalizing CRLF and LF line endings. The repository also pins these two
files to LF through `.gitattributes`. Therefore, a Windows checkout cannot
invalidate the contract solely because Git rewrote newline bytes. Changing
dependency declarations, versions, markers, or artifact hashes without
intentionally updating the contract still causes Checkpoint 57, the end-to-end
pipeline, and GitHub Actions to fail.

Hashes serve two different purposes:

- the file fingerprint detects a changed dependency manifest or lock; and
- each package hash tells pip which downloadable artifacts are approved.

## Reproducing the Environment

Activate the project virtual environment, then install the pinned resolver:

```powershell
python -m pip install "pip==26.0.1" "setuptools==83.0.0" "wheel==0.47.0"
```

Install the complete lock with hash enforcement:

```powershell
python -m pip install --no-build-isolation --require-hashes -r requirements-lock.txt
```

`--no-build-isolation` makes source distributions use the already pinned
setuptools and wheel versions instead of creating a temporary environment with
newer, unreviewed build tools.

Check package compatibility:

```powershell
python -m pip check
```

Validate versions, counts, hashes, fingerprints, Python, pip, and CI:

```powershell
python src\validate_dependency_environment.py
```

Run the complete local checkpoint:

```powershell
.\scripts\checkpoints\run_checkpoint57.ps1
```

## GitHub Actions Enforcement

The `Python quality` workflow now:

1. installs Python 3.12.4;
2. installs pip 26.0.1, setuptools 83.0.0, and wheel 0.47.0;
3. installs `requirements-lock.txt` with `--require-hashes`;
4. runs `pip check`;
5. runs the dependency validator;
6. compiles the project;
7. runs Ruff linting and formatting checks; and
8. runs the full automated test suite.

The pip cache key includes both `requirements.txt` and
`requirements-lock.txt`. A dependency change therefore invalidates the old
cache instead of silently reusing it.

## Intentional Dependency Update Process

A dependency update must be a reviewed engineering change. Do not edit only
one version and commit it without regeneration and validation.

The required process is:

1. Create a dedicated dependency-update branch.
2. Change the intended exact version in `requirements.txt`.
3. Regenerate `requirements-lock.txt` for Python 3.12 with artifact hashes.
4. Review the resolved transitive changes.
5. Update the two SHA-256 fingerprints in
   `config/dependency_reproducibility.yaml`.
6. Install the new lock in a clean virtual environment.
7. Run `python -m pip check`.
8. Run the dependency validator.
9. Run the full automated test suite.
10. Run affected analytical checkpoints and inspect material result changes.
11. Commit the manifest, lock, fingerprints, tests, and documentation
    together.

Automatic dependency updates are not accepted without this review. A newer
version is not automatically a safer or compatible version for this
portfolio.

## What This Checkpoint Does Not Claim

Pinned dependencies improve repeatability; they do not guarantee:

- that all upstream software is defect-free;
- that an old dependency should be used forever;
- that CPU architecture and operating-system numerical results are
  bit-for-bit identical;
- that PostgreSQL server behavior is locked by the Python environment; or
- that future dependency updates require no analytical review.

The lock records the environment that was tested. Security and compatibility
updates should still occur through the intentional dependency update process.

## Saved Validation Outputs

Checkpoint 57 writes aggregate technical evidence to:

```text
data/processed/dependency_environment/
```

The folder contains:

- `dependency_environment.csv`, with expected and installed versions; and
- `dependency_reproducibility_validation.csv`, with every contract check.

No employee, salary, attrition, or model-prediction data is used.
