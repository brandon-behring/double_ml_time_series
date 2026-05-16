# CLAUDE.md

Repository guidance for coding agents working in this repo.

## Project Frame

This is a book-grade companion repository for a Double Machine Learning manuscript,
not production deployment software and not a stable public package. The public contract
is being rebuilt around verified runtime behavior.

Use [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) as the only current status source.
Older reports and roadmaps live under
[docs/archive/superseded_2026-05-02](docs/archive/superseded_2026-05-02/ARCHIVE_INDEX.md).

Current core:

- FWL, Robinson, and cross-sectional/i.i.d.-style `double_ml`
- `TemporalPLRDML`: scalar temporal PLR DML with lagged treatment controls,
  temporal cross-fitting, and HAC inference
- `RollingWindowDML` and `PanelDML` as companion estimators
- Time-series CV, HAC/Newey-West, synthetic macro controls, and validation helpers

Do not describe this repo as production-grade. Do not describe `TemporalPLRDML` as true
Lewis-Syrgkanis dynamic g-estimation, period-specific `theta_t`, recursive
g-estimation, CausalForestDML/BLP/policy-tree implementation, automatic stationarity
enforcement, or deployment readiness unless the code path being discussed proves it.

## Required Commands

Use the repo venv explicitly:

```bash
venv/bin/python -m pytest --collect-only -q
venv/bin/python -m pytest -m tier1 --no-cov -q
venv/bin/python -m pytest -m "tier1 or tier2" --no-cov -q
venv/bin/python -m black --check src/ test/ examples/
venv/bin/python -m mypy src/ --ignore-missing-imports --no-strict-optional --explicit-package-bases
```

Examples:

```bash
for f in examples/*.py; do venv/bin/python "$f"; done
```

Docs:

```bash
venv/bin/python -m pip install -e ".[dev,docs]"
venv/bin/python -m sphinx -b html -W --keep-going docs/sphinx docs/sphinx/_build/html
```

Book:

```bash
lualatex -shell-escape main.tex
biber main
lualatex -shell-escape main.tex
lualatex -shell-escape main.tex
```

LuaLaTeX plus biber is the intended book toolchain. Fatal TeX errors are blocking.
Overfull/underfull boxes are currently reported but not yet blocking.

## Current Baseline

Audit baseline from 2026-05-02:

- 796 tests collected before this remediation pass
- Tier 1: 314 tests
- Tier 1 + Tier 2: 615 tests
- Book PDF: 205 pages
- Sphinx requires docs dependencies
- Generated artifacts remain tracked until a separate cleanup decides whether to untrack
  or freeze them

## Source Layout

```text
src/dml/
  fwl.py                 FWL residualization baseline
  robinson.py            Robinson partially linear estimator
  double_ml.py           Cross-fitted i.i.d.-style PLR DML
  cross_fitting.py       Time-series CV helpers
  hac.py                 HAC/Newey-West inference
  temporal_plr_dml.py    TemporalPLRDML, RollingWindowDML, PanelDML

src/data/                FRED loader, OJ loader, synthetic macro controls
src/validation/          DGPs, stationarity diagnostics, validation helpers
src/production/          Research/demo utilities only
examples/                Runnable public examples
docs/sphinx/             Sphinx docs
docs/audits/             Current audit evidence
docs/archive/            Superseded reports and plans
chapters/                LaTeX manuscript
```

## Methodology Rules

- Runtime behavior, tests, live APIs, logs, and primary literature outrank docs.
- `double_ml` stays cross-sectional/i.i.d.-style in this remediation pass.
- `TemporalPLRDML` uses temporal CV and excludes rows without valid out-of-fold
  temporal predictions.
- Stationarity, cointegration, overlap, and weak treatment residual variation are
  documented risks. This milestone may warn but should not block estimation on those
  diagnostics.
- Keep imports under the staged `src.*` namespace until the documentation/API cleanup
  has passed.

## Editing Rules

- Keep changes tightly scoped to truth cleanup, API/doc contract repair, examples,
  and gates.
- Do not restore archived status docs as current sources.
- Do not add compatibility aliases for the old `DynamicDML` names.
- Avoid new methodology claims unless they are backed by executable examples or tests.
