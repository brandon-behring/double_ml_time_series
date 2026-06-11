# Double Machine Learning For Time Series

Book-grade companion code for a Double Machine Learning manuscript. This repository is
not a deployable production package and does not currently expose a stable public API.

The current verified core is:

- Cross-sectional partially linear DML via `double_ml`
- FWL and Robinson-estimator teaching implementations
- Temporal partially linear DML via `TemporalPLRDML`
- Time-series cross-validation helpers and HAC/Newey-West inference utilities
- Synthetic data generators and book examples (stationarity diagnostics via temporalcv)

The temporal estimator `TemporalPLRDML` estimates a scalar partially linear treatment
effect with lagged treatment controls, temporal cross-fitting, and HAC inference. True
Lewis-Syrgkanis recursive dynamic g-estimation with period-specific `theta_t` blips is
implemented separately as `DynamicGEstimationDML` (constant-blip linear SNMM, panel +
single-series, with a gated EconML cross-check). Heterogeneous `theta_t(X)`, causal
forests, BLP/policy-tree workflows, blocking stationarity gates, and production
deployment remain deferred work unless a specific module or example proves otherwise.

## Current Status

Use [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) as the only current project-status
source. Older reports and roadmaps are archived under
[docs/archive/superseded_2026-05-02](docs/archive/superseded_2026-05-02/ARCHIVE_INDEX.md).

Audit baseline from 2026-05-02:

- 796 tests collected before this remediation pass
- Tier 1: 314 tests
- Tier 1 + Tier 2: 615 tests
- Book PDF: 205 pages
- Sphinx docs require optional docs dependencies
- Book build had no recorded fatal TeX errors, but overfull boxes remain known issues

## Installation

```bash
cd double_ml_time_series
venv/bin/python -m pip install -e ".[dev]"
```

For docs work:

```bash
venv/bin/python -m pip install -e ".[dev,docs]"
```

Quick import check:

```bash
venv/bin/python -c "from dml_ts.dml import double_ml, TemporalPLRDML, RollingWindowDML; print('OK')"
```

## Quick Starts

### Cross-Sectional PLR DML

`double_ml` is the i.i.d.-style partially linear DML helper. It does not perform temporal
cross-validation in this remediation milestone.

```python
import numpy as np

from dml_ts import double_ml

rng = np.random.default_rng(42)
n = 500
X = rng.normal(size=(n, 5))
T = X[:, 0] + rng.normal(size=n)
Y = 2.0 * T + X[:, 1] ** 2 + rng.normal(size=n)

result = double_ml(Y, T, X, n_folds=5, model="ridge", random_state=42)

print(f"theta: {result.theta:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### Temporal PLR DML

Use `TemporalPLRDML` for ordered data when lagged treatment controls, temporal
cross-fitting, and HAC standard errors are part of the chapter claim.

```python
import numpy as np

from dml_ts import TemporalPLRDML

rng = np.random.default_rng(42)
n = 240
time_index = np.arange(n)
X = np.column_stack([rng.normal(size=n), np.sin(time_index / 12)])
T = 0.4 * X[:, 0] + rng.normal(size=n)
Y = 1.5 * T + X[:, 1] + rng.normal(size=n)

model = TemporalPLRDML(
    n_lags=2,
    model_y="ridge",
    model_t="ridge",
    n_splits=4,
    gap=2,
    hac_bandwidth=6,
    random_state=42,
)
result = model.fit(Y, T, X, time_index=time_index)

print(f"theta: {result.theta:.3f}")
print(f"HAC SE: {result.se:.3f}")
print(f"temporal CV rows dropped: {result.dropped_initial_rows}")
```

### Time-Series Cross-Validation

```python
import numpy as np

from dml_ts.dml import TimeSeriesCrossValidator

X = np.arange(100).reshape(-1, 1)
cv = TimeSeriesCrossValidator(n_splits=5, gap=3, purge_length=2)

for train_idx, test_idx in cv.split(X):
    assert train_idx[-1] + 3 + 2 <= test_idx[0]
```

### Synthetic Macro Controls

```python
from dml_ts.data import create_synthetic_fred_data

macro = create_synthetic_fred_data(
    start_date="2018-01-01",
    end_date="2020-12-31",
    frequency="M",
    seed=42,
)

X_macro = macro.data.values
print(macro.data.columns.tolist())
```

Live FRED access uses `FREDLoader.get_macro_controls(...)` and requires a configured FRED
API key.

## Repository Map

```text
dml_ts/dml/
  fwl.py                 Linear residualization baseline
  robinson.py            Robinson partially linear estimator
  double_ml.py           Cross-fitted i.i.d.-style PLR DML
  cross_fitting.py       Time-series CV helpers
  inference.py           HAC causal-layer helper (primitives via temporalcv)
  temporal_plr_dml.py    TemporalPLRDML, RollingWindowDML, PanelDML

dml_ts/data/                FRED loader, OJ loader, synthetic macro data
dml_ts/validation/          DGPs, diagnostics, validation helpers
dml_ts/production/          Research/demo pipeline utilities, not deployment guarantees
examples/                Runnable companion examples
chapters/                LaTeX manuscript chapters
docs/sphinx/             Sphinx API/user docs
docs/audits/             Audit reports and evidence
docs/archive/            Superseded reports and roadmaps
```

## Verification Commands

```bash
venv/bin/python -m pytest --collect-only -q
venv/bin/python -m pytest -m tier1 --no-cov -q
venv/bin/python -m pytest -m "tier1 or tier2" --no-cov -q
venv/bin/python -m ruff check dml_ts/ test/ examples/
venv/bin/python -m ruff format --check dml_ts/ test/ examples/
venv/bin/python -m mypy dml_ts/ --ignore-missing-imports --no-strict-optional --explicit-package-bases
```

Examples:

```bash
for f in examples/*.py; do venv/bin/python "$f"; done
```

Sphinx docs:

```bash
venv/bin/python -m sphinx -b html -W --keep-going docs/sphinx docs/sphinx/_build/html
```

Book:

```bash
lualatex -shell-escape main.tex
biber main
lualatex -shell-escape main.tex
lualatex -shell-escape main.tex
```

Fatal TeX errors are blocking. Overfull and underfull boxes are reported in this
milestone but are not yet blocking.

## Methodology Guardrails

- `double_ml` should be documented as cross-sectional/i.i.d.-style PLR DML.
- `TemporalPLRDML` should be documented as scalar temporal PLR DML, not recursive
  dynamic g-estimation.
- Temporal CV predictions are only used where true out-of-fold predictions exist; early
  uncovered rows are excluded and reported.
- Stationarity, cointegration, overlap, and weak treatment residual variation remain
  user responsibilities in this milestone. The code provides diagnostics and warnings,
  not blocking automatic enforcement.

## License

Personal research project. Not for distribution without an explicit release pass.
