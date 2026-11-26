# What the software and publication checks cover

| Check | Evidence and scope |
|---|---|
| PDF | Forced a complete LuaLaTeX/Biber/LaTeX rebuild in a disposable edition directory. The final PDF is 180 pages. The final LaTeX log reports no unresolved references or citations. Extracted text includes the Chapter 9 held-out heterogeneity section and incremental policy-gain calculation. |
| Web | The academic content validator checked all ten chapters. The static build generated and indexed 15 pages, including the corrected Chapter 9. The standalone PDF and web builds used the same Chapter 9 source versions checked by the executable-listings test. |
| Sphinx | Clean HTML build with warnings as errors. Strict rendering caught an indentation error in the Robinson estimator example during the audit. Both the locked docs-extra and rich optional-dependency environments subsequently produced a clean build. Global doctests had 49 passes and 13 skips in the locked base environment, and 50 passes and 12 skips with optional dependencies installed. |
| Notebooks | Both notebooks ran with a real kernel (22 code cells total), including a correction to working-directory resolution and an independent calculation of simulated coverage. [Notebook receipts](additional-materials.md) |
| TemporalCV | The consumer uses TemporalCV 2.2.0 from its lock. A second run prepended the local TemporalCV 2.3.0 source directory and ran the split-factory, temporal PLR and audit contract suites. All 72 tests passed. This checks compatibility of these public paths. It supplies no new causal identification assumptions. |
| Public estimators and artifacts | The final full suite, including the bootstrap-copy corrections, passed **1,154 tests** with **37 skips**. The tier 1–2 run passed **968**, with 9 skipped and 214 deselected by tier. It covers independent algebra, splits, storage, serialization, interval constructions and seeded simulation controls. The separate test of 43 monitor cases includes four zero/negative-baseline branches. Thirty-one forecast-record tests check issuance and scoring. The full registered and standalone data replays have their own numerical receipts. [Registered replays](historical-reproduction.md) · [Standalone evidence](standalone-reproduction.md) |
| Static checks | The CI mypy command passed for all 46 package source files. It exposed a reused dictionary-key variable and inadequate narrowing of a validated numeric value in the effect-forecast record. The source now describes the keys and validated values with their actual types. The complete audit's Python files passed Ruff. [Forecast implementation](../../../dml_ts/validation/effect_forecast.py) |
| External numerical control | A second execution on the 16 external ACIC development files reproduced the saved CSV byte for byte in the same environment. The ordinary estimator was also checked against an independent residual-moment calculation on every file. [Source, limitations and replay command](external-control.md) |

The [software receipt](reproduction/verification.json) records environment versions, edition digests, regression outcomes and the checks applied. An environment with EconML, R and the source caches supports more tests than the minimal package installation. Skipped optional data and dependency cases should be read with the separate real-data replays. A green unit suite alone cannot demonstrate interval calibration.

## Commands

Run from the repo root. The PDF and web commands can run in a disposable copy when publication artifacts should be kept outside the working tree. The tests use single-threaded numerical libraries where a study or Monte Carlo driver already parallelizes across independent units.

```bash
export PYTHONPATH=.
export DML_N_JOBS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --locked --extra dev pytest --no-cov -m 'tier1 or tier2'
uv run --locked --extra dev python -m mypy dml_ts/ --ignore-missing-imports --no-strict-optional --explicit-package-bases
make -B all
(cd web && npm ci && npm run build)
sphinx-build -E -b html -W --keep-going docs/sphinx /tmp/dml-sphinx-build
```

The full Python suite can use `pytest --no-cov` in the optional-dependency environment. The registry commands, source hashes, independent worker limits, cutoffs, and per-output tolerances are documented in the corresponding reproduction reports. The Git commit alone omits uncommitted work in the audit input. [Starting snapshot and fingerprints](reproduction/launch-receipt.json)
