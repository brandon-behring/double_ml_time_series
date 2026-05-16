# Current Status

Last updated: 2026-05-02

This file is the canonical project-status source. Historical reports and
roadmaps were moved to `docs/archive/superseded_2026-05-02/` during the
truth-first remediation.

## Status

The repository is being remediated as a book-grade companion repo for Double
Machine Learning pedagogy. It is not a deployable production software project.

Current verdict: partial restart.

Keep and stabilize:

- Cross-sectional FWL, Robinson, and DML teaching path.
- Temporal partially linear DML tooling with lagged controls, temporal
  cross-validation, and HAC inference.
- HAC, time-series splitters, stationarity diagnostics, synthetic DGPs, and the
  existing test suite.
- LaTeX manuscript as a draft book companion.

Deferred until after remediation gates pass:

- True Lewis-Syrgkanis recursive dynamic g-estimation.
- Public package namespace migration away from `src.*`.
- Optional EconML heterogeneity examples as runnable features.
- Blocking stationarity, cointegration, and overlap gates.
- Cleanup of tracked generated artifacts in `results/` and `main.bbl`.
- Content expansion beyond the current manuscript.

## Verified Baseline

Evidence from the 2026-05-02 audit:

- 796 tests collected before remediation.
- Tier 1 before remediation: 314 passed.
- Tier 1 + Tier 2 before remediation: 615 passed with 16 warnings.
- Black check passed on `src/`, `test/`, and `examples/`.
- Mypy check passed on `src/` with the existing permissive command.
- Five example scripts executed, with labeling/methodology fixes required.
- Existing `main.pdf` is 205 pages and produced by LuaTeX.
- Existing `main.log` had no direct LaTeX errors or direct `LaTeX Warning`
  lines, but did include 254 overfull and 12 underfull boxes.
- Current venv did not include Sphinx or optional full-stack dependencies such
  as EconML, DoubleML, FRED, XGBoost, LightGBM, DoWhy, CausalML, or linearmodels.

## Remediation Verification

Evidence from the 2026-05-02 remediation run:

- 802 tests collected after adding public-snippet contract tests.
- Tier 1: 320 passed.
- Tier 1 + Tier 2: 621 passed with 23 warnings.
- Black check passed on `src/`, `test/`, and `examples/`.
- Mypy check passed on `src/` with the documented permissive command.
- All five `examples/*.py` scripts executed successfully.
- Sphinx built successfully with `-W --keep-going` after installing the `docs`
  optional extra.
- LuaLaTeX + biber book build completed successfully with no fatal TeX errors.
  Overfull and underfull boxes remain non-blocking report items.

## Canonical References

- Current audit: `docs/audits/2026-05-02_repo_quality_methodology_audit.md`
- Audit index: `docs/audits/README.md`
- Historical status/reports: `docs/archive/superseded_2026-05-02/`
- Active plans: `docs/plans/active/`

## Working Rules

- Runtime behavior, tests, examples, build logs, and primary literature outrank
  old roadmaps or status files.
- Public snippets in README, Sphinx, examples, and chapters must either execute
  under documented dependencies or be clearly labeled as conceptual.
- The current temporal estimator is scalar temporal PLR DML, not recursive
  dynamic g-estimation.
- "Production" language means book-companion production concepts unless code is
  explicitly tested as deployable software.
