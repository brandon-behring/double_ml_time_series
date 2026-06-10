# Causal & Statistical Assumptions

What each estimator assumes, where the manuscript develops it, and what the
code does (warn vs raise) when an assumption is at risk. Runtime behavior,
tests, and primary literature outrank this file if they ever disagree.

## Shared (all PLR-style estimators)

| Assumption | Meaning here | On violation |
|---|---|---|
| Partially linear outcome | `Y = theta*T + g(X) + e`, scalar constant `theta` | Not testable; chapter 3 discusses misspecification |
| Unconfoundedness given controls | `E[e | T, X] = 0` (with lags where applicable) | Not testable; sensitivity tools in `dml_ts/sensitivity/` |
| Overlap / treatment variation | residualized T retains variation | ALL estimators RAISE when residual variation ~0; TemporalPLRDML additionally WARNS when weak (the others stay silent today) |
| Nuisance learnability | `m(X)`, `l(X)` estimable at adequate rates | TemporalPLRDML WARNS on poor cross-validated nuisance R² (threshold -0.25); `double_ml`/FWL/Robinson compute R² but do not warn |

## `double_ml` (cross-sectional / i.i.d.-style)

- I.i.d. sampling; K-fold cross-fitting valid. Influence-function variance
  `Var(psi)/n` assumes no serial correlation — do not use on time series
  (that is `TemporalPLRDML`'s job).

## `TemporalPLRDML` (+ `RollingWindowDML`)

- **Sequential exogeneity** with lagged treatment controls (chapter 5). This is
  scalar PLR DML with lags as controls — NOT recursive dynamic g-estimation and
  it does not estimate period-specific effects.
- **Temporal cross-fitting**: nuisances trained strictly on the past; early rows
  without out-of-fold predictions are excluded (counts reported on the result).
- **HAC inference**: influence scores may be serially correlated; Newey-West
  long-run variance with data-driven bandwidth. SE = sqrt(Omega_hat / n)
  (issue #7 fixed the sqrt(n) understatement).
- **Stationarity, cointegration**: documented risks — diagnostics may WARN but
  do not block estimation at this milestone.
- `RollingWindowDML` additionally assumes effects vary slowly relative to the
  window; windows that fail to fit are currently skipped (see issue #10 for the
  center-alignment hazard).

## `PanelDML`

- Fixed-effects transformation removes unit/time heterogeneity; strict
  exogeneity within the demeaned equation.
- **Cluster-robust inference** (CR1 over within-cluster influence sums, issue #9
  fixed the ~cluster-size overstatement): requires >= 2 clusters with retained
  observations (RAISES otherwise); clusters dropped by CV trimming are disclosed
  via RuntimeWarning. Normal critical values — mildly anticonservative at small
  G (t_{G-1} refinement tracked in #12 discussion).

## `DynamicGEstimationDML`

- **Linear SNMM with constant (non-heterogeneous) period-specific blips**
  `theta_1..theta_m` (chapter 6); recursive Lewis-Syrgkanis g-estimation.
- Sequential conditional exogeneity at every period; joint sandwich variance
  (panel mode; the single-series mode uses a per-stack HAC sandwich); optional
  EconML `DynamicDML` cross-check is gated (tier3, nightly lane).
- Heterogeneous `theta_t(X)` is explicitly future work — do not claim it.

## Data layer

- FRED/OJ loaders document their transformations; synthetic DGPs carry known
  ground-truth effects *for validation only* — generic series simulation is
  temporalcv's domain (mechanism universal, interpretation unique).
