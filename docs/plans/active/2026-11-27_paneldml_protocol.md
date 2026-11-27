# PanelDML study protocol (frozen)

Date: 2026-11-27. Status: **frozen.** Its SHA-256 is recorded in
`docs/plans/active/2026-11-24_backlog_landing.md` (Addendum 6). It is committed and pushed
before any study run. A change after that is a new protocol with a new hash and a reason.

Source: §5 of the design gate (`docs/plans/active/2026-11-27_paneldml_design_gate.md`), adopted by
the owner on 2026-11-27, with the decisions of its §7 applied. The draft left some details open.
Each detail chosen at freezing is marked **[F]**.

## 1. Question

The question is which nuisance treatment lets `PanelDML` recover a constant effect θ = 2, with
intervals that cover. The treatments are A (basis first, then within), B (correlated random
effects with the treatment-mean correction) and C (exact first difference). The panels are
static, partially linear, with fixed effects, and the confounding is nonlinear in X. D (hybrid)
is not implemented in this round.

## 2. Data

All cells use m = 6 periods, rows unit-major (unit, then period), and true effect θ = 2. The
base generator is `_panel_data` of the second audit
(`docs/audits/2026-11-22_end_to_end_review/second_audit_checks/area1.py:939`), restated here so
that the protocol does not depend on a held file:

```
alpha_i ~ N(0, 1)                      unit effect, i = 1..N
mu_i    = 1.2 alpha_i + N(0, 0.5^2)    unit mean of X
X_it    = mu_i + N(0, 1)
e_T, e_Y ~ N(0, 1), independent
linear, quadratic:  g = X (linear) or X^2 (quadratic)
                    T = alpha_i + 0.8 g + e_T
                    Y = 2 T + alpha_i + g + e_Y
interactive:        T = alpha_i X + e_T
                    Y = 2 T + alpha_i X + e_Y
```

| Cell | Definition | N (units) | Role |
|---|---|---|---|
| linear | g = X | 80, 160 | gating (control) |
| quadratic | g = X² | 80, 160 | gating |
| interactive | as above | 80, 160 | gating |
| uncorrelated effects | quadratic, with mu_i = 1.3 z_i, z_i ~ N(0, 1) independent of alpha_i (the same variance as 1.2 alpha_i + N(0, 0.5²)) **[F]** | 160 | gating |
| two-way | quadratic, plus lambda_t ~ N(0, 1) per period, added with coefficient 1 to X, T and Y; fixed effects `twoway` **[F]** | 80, 160 | gating |
| unbalanced | quadratic; each unit-period dropped with probability 0.2, independently of everything (missing completely at random) **[F]** | 160 | gating |
| few units | quadratic | 20 | reported, not gating |
| feedback | quadratic, generated period by period with T_it = alpha_i + 0.8 X_it² + 0.3 Y_i,t−1 + e_T and Y_i,−1 = 0 **[F]** | 160 | violation, reported apart |
| absorbed | quadratic, but T_it = alpha_i + 0.8 mu_i² + e_i, one draw per unit (constant within unit) **[F]** | 80 | every arm must raise |

Fixed effects are `individual` in every cell except two-way. **[F]**

## 3. Arms

Inputs pass the input boundary in every arm. Folds are `whole_unit` with 4 folds, the setting
of the review rows this study is compared with (`rep_panel_native`) **[F]**. Inference is
`unit_cluster` (C&P eq 5.6). In the two-way cell `two_way_cluster` is also reported from the
same fits; it does not gate.

| Arm | What it is | Learners |
|---|---|---|
| Z | `main`'s `PanelDML` at the head that runs the study: demeaned X, the known defect of #84. A negative control | default forest; boosting(40) |
| S | the snapshot's `PanelDML` (`refs/snapshots/backlog-2026-11-24`): `whole_unit`, `unit_cluster` | default forest; boosting(40) |
| A | basis B(X_it) on levels, then the within (or two-way) transform with Y and T; a linear fit | ridge; lasso |
| B | (X_it, X̄_i) on levels (in the two-way cell, period means too, F&P 4.5.4); treatment residual V̂ = (D − D̄) − (m̂ − m̂̄) (C&P 5.1.1) | lasso on the dictionary; boosting(40); forest(100) |
| C | ΔY and ΔD learned on (X_it, X_i,t−1); in the two-way cell, period dummies added to the learners' inputs | lasso on the dictionary; boosting(40); forest(100) |
| T | true nuisance: the generator's conditional means of Y and T given X and the unit effect, through the transform of each of A, B and C, the same folds and score | none (3 arms: within, CRE, FD) |

That is 15 arm-learner pairs. **[F]** The learners:

- **dictionary:** `PolynomialFeatures(degree=3, include_bias=False)` on the standardized
  learner inputs.
- **ridge:** `RidgeCV(alphas=np.logspace(-3, 3, 13))` on the dictionary.
- **lasso:** `LassoCV(cv=5, n_alphas=50, max_iter=10_000)` on the dictionary.
- **boosting(40):** `GradientBoostingRegressor(n_estimators=40, max_depth=3)`.
- **forest(100):** `RandomForestRegressor(n_estimators=100, min_samples_leaf=5)`.
- **default forest:** the estimator's own default learner at the head that runs the study.

Each learner's `random_state` comes from the replication's seed (§5).

## 4. Pass rule

- An arm **covers** in a cell if its nominal 95% interval covers θ in at least 92.5% of
  replications, and **fails** below 90%. Coverage between 90% and 92.5% does not cover and is
  reported as marginal. At R = 1,000 the Monte Carlo SE of a 95% coverage is 0.69 points.
- **Bias:** |mean(θ̂) − 2| ≤ 0.2 × SD(θ̂) in the cell. A bias of 0.2 SD alone lowers a 95%
  interval's coverage to 94.5%.
- **A candidate passes** if, with at least one of its learners, it covers and meets the bias
  rule in every gating cell at every size listed, and every arm raises `ValueError` in the
  absorbed cell. **[F: "with at least one of its learners"; the release then names that learner
  class.]**
- **The study is valid** only if the true-nuisance arms cover and meet the bias rule in every
  gating cell, and Z fails the quadratic cell (coverage below 90%) with both learners. If either
  check fails, the study is broken: it is reported and no verdict is drawn.
- A replication that raises counts against coverage as a non-covering interval. Raises are
  reported by cell, arm and message. **[F]**

## 5. Replications and seeds

- 1,000 replications per simulated cell and size. There are 12 cell-sizes, plus the absorbed
  cell, which runs 20 replications per arm: each must raise. **[F: 20]**
- The seed for replication r of cell c (in the order of the table in §2) and size N is
  `np.random.SeedSequence(20261127, spawn_key=(c, N, r))`. Its first child draws the data, so
  every arm sees the same data in a replication. Its second child gives the learners'
  `random_state` and the fold shuffle. **[F]**
- The draws are stored gzip-compressed, one row per (cell, N, r, arm, learner), with θ̂, SE,
  interval, covered and the error message if one was raised. A second run must give a
  byte-identical file: its SHA-256 is compared.

## 6. Cost and the memory cap

- 12 cell-sizes × 1,000 replications × 15 arm-learner pairs = 180,000 fits, plus 300 fits in
  the absorbed cell.
- Timed single fits (design gate §5, locked environment, one thread): the default forest
  0.70-0.76 s, boosting(40) 0.02-0.03 s, forest(100) 0.12-0.20 s, RidgeCV 0.001 s. If every fit
  took 0.76 s, the study would be about 38 CPU hours.
- **Under the cap:** at most 16 worker processes (one thread each) in one
  `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0` scope. A run starts only with
  at least 60 GiB available, and no other heavy job of this repository runs at the same time.
  The worst case is 38/16 ≈ 2.4 hours, within the four-hour budget. At about 2 GiB per process,
  16 workers use about 32 GiB.
- Before the full run, a pilot of 20 replications per cell, size and arm is timed. If it projects
  more than four hours, the run stops and is reported. It is not trimmed silently.

## 7. What is reported

For every cell, size, arm and learner: bias, SD, bias/SD, RMSE, coverage with its Monte Carlo
SE, the mean SE over SD, and the count of raises. Also reported: each candidate's verdict under
§4, the validity checks, the feedback and few-units cells apart, the two-way-cluster coverage,
and the SHA-256 of both runs' draws.

## 8. After the study

The decision is the owner's, as the design gate's §6.5 says. If a candidate passes, it is
released with its learner class named, the nuisance argument required, `whole_unit` folds and
`unit_cluster` by default. If none passes, the result is reported and the owner chooses
between a restricted, documented scope and holding `PanelDML`.
