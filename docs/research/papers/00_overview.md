# Double Machine Learning for time series and panel data — Overview

Extended overview, scope statement, and glossary for the agent-index.

**Last updated:** 2026-05-16.

## What this synthesis covers

Primary sources for Double Machine Learning (DML) applied to time-series and panel data, organized into 7 sub-areas:

- **A — FWL theorem, Robinson estimator, partial-out theory** — `01_theory_partialing.md`
- **B — Neyman orthogonality and semiparametric DML theory** — `02_theory_orthogonality.md`
- **C — Cross-sectional DML estimators and HTE** — `03_estimator_xs.md`
- **D — Temporal & dynamic DML** (user-emphasized) — `04_estimator_temporal.md`
- **E — Time-series inference machinery** — `05_inference_ts.md`
- **F — Real-world replications & applied DML** (user-emphasized) — `06_replication_applied.md`
- **G — Implementations & software** — `07_tooling.md`

## Glossary

Canonical terminology, aliases, one-line definitions, and pointers into the synthesis files.

- **DML** — *Double Machine Learning (a.k.a. Debiased ML, DDML)*. ML-nuisance-controlled causal estimation using Neyman-orthogonal scores and cross-fitting. Canonical reference: Chernozhukov et al. 2018 in `02_theory_orthogonality.md` § B1.
- **FWL** — *Frisch–Waugh–Lovell theorem*. Equivalence between regressing Y on X with controls Z, and regressing residual-Y-after-Z on residual-X-after-Z. Canonical reference: Frisch & Waugh 1933; Lovell 1963 in `01_theory_partialing.md` § A1.
- **PLR** — *Partially Linear Regression*. DML model Y = θ·D + g(X) + ε with D the scalar treatment, g a nonparametric nuisance function. The simplest DML setup.
- **IRM** — *Interactive Regression Model*. DML model with binary treatment; estimates ATE / ATTE using AIPW-style score functions.
- **IIVM** — *Interactive IV Model*. DML model with binary treatment and binary instrument; estimates LATE under monotonicity.
- **Neyman orthogonality** — *First-order insensitivity to nuisance parameters*. Property of a moment condition where its Gateaux derivative wrt the nuisance function vanishes at the true parameter — the condition that makes DML's cross-fitting work.
- **Cross-fitting** — *Sample splitting + averaging*. K-fold split: train nuisance on K-1 folds, score on the held-out fold, rotate; averages give the DML estimate. Eliminates bias from over-fitting the nuisance in-sample.
- **HTE / CATE** — *Heterogeneous (or Conditional Average) Treatment Effect*. Treatment effect that varies with covariates: τ(x) = E[Y(1)−Y(0) | X=x]. Estimated by causal forests, R-learner, X-learner, etc. See file C.
- **R-learner** — *Robinson-style two-step CATE learner*. Nie & Wager 2021 (Biometrika): estimate Y- and D-residuals via cross-fitting, then minimize R-loss for CATE. See `03_estimator_xs.md` § C1.
- **X-learner** — *Künzel et al. 2019 meta-learner*. Targets unbalanced treatment-group sizes by training separate base learners and combining via propensity. See `03_estimator_xs.md` § C1.
- **Causal forest / GRF** — *Random-forest-based CATE estimator*. Wager–Athey 2018 (causal forest) + Athey–Tibshirani–Wager 2019 (GRF). See `03_estimator_xs.md` § C1.
- **Dragonnet** — *Shi–Blei–Veitch 2019 neural-net causal estimator*. Shared-trunk network with outcome and propensity heads + targeted regularization. See `03_estimator_xs.md` § C1.
- **g-estimation** — *Robins-family estimator for time-varying treatments*. Solves structural nested mean models via residual moments; the temporal-DML estimator is its orthogonal cross-fitted version. See `04_estimator_temporal.md` § D1.
- **SNMM** — *Structural Nested Mean Model*. Robins' model class for time-varying treatments with time-varying confounders that are themselves affected by past treatment.
- **MSM** — *Marginal Structural Model*. Robins–Hernán model with IPTW for time-varying treatments. See `04_estimator_temporal.md` § D1.
- **Lewis-Syrgkanis** — *Lewis & Syrgkanis 2020 dynamic-DML paper*. Founding reference for DML applied to dynamic treatment effects with ML nuisance. The book's `TemporalPLRDML` cites this as canonical.
- **TemporalPLRDML** — *Repo-specific name for scalar temporal-PLR DML*. The estimator implemented in `src/dml/temporal_plr.py` — Lewis-Syrgkanis-style with lagged treatment controls and HAC inference.
- **HAC** — *Heteroskedasticity- and Autocorrelation-Consistent (covariance)*. Standard-error estimator robust to both forms of dependence. Canonical: Newey & West 1987. See `05_inference_ts.md` § E1.
- **Newey–West** — *HAC SE estimator*. Newey & West 1987 Econometrica — Bartlett-kernel sum of sample autocovariances.
- **Driscoll–Kraay** — *Panel-data HAC robust to cross-sectional dependence*. Driscoll & Kraay 1998 ReStat — Newey-West applied to cross-section averages of moment conditions.
- **Fixed-b asymptotics** — *Kiefer–Vogelsang 2005*. Asymptotic theory holding bandwidth proportional to sample size; gives more accurate finite-sample HAC inference.
- **Purged k-fold CV** — *de Prado 2018 backtest-safe CV*. Removes training labels overlapping test labels; embargoes a buffer period.
- **KPSS / ADF / PP** — *Stationarity / unit-root tests*. KPSS (Kwiatkowski et al. 1992) tests trend-stationarity; ADF (Dickey-Fuller 1979) and PP (Phillips-Perron 1988) test for unit roots. See `05_inference_ts.md` § E1.
- **401(k) eligibility study** — *Canonical applied-DML benchmark*. Chernozhukov–Hansen 2004 with 401(k) eligibility as IV; Poterba–Venti–Wise 1995/1996 for the original crowd-out debate. See `06_replication_applied.md` § F1.
- **LaLonde / Dehejia–Wahba** — *Founding training-program reanalysis benchmark*. LaLonde 1986 AER + Dehejia–Wahba 1999 JASA propensity-score reanalysis. See `06_replication_applied.md` § F1.
- **FRED** — *Federal Reserve Economic Data*. Public macro-data API (fred.stlouisfed.org). Used in the book's macro-controls chapter.
- **DoubleML** — *R/Python DML implementation*. Bach–Chernozhukov–Kurz–Spindler — R (JSS 2024) + Python (JMLR 2022). See `07_tooling.md` § G1.
- **EconML** — *Microsoft Research DML library*. Microsoft / ALICE — DML + dynamic-DML + causal forests + meta-learners. See `07_tooling.md` § G1.
- **DoWhy** — *PyWhy graphical-causal-model library*. Microsoft / PyWhy — Model–Identify–Estimate–Refute pipeline. See `07_tooling.md` § G1.
- **CausalML** — *Uber causal-ML Python package*. Meta-learners + uplift trees + Dragonnet + CEVAE. See `07_tooling.md` § G1.
- **grf** — *R generalized-random-forests package*. Canonical implementation of GRF and causal forests. See `07_tooling.md` § G1.

## Out of scope

- Synthetic control / synthetic DiD (deserves its own dossier).
- General time-series forecasting (ARIMA, transformers for TS).
- DiD with staggered adoption (own active literature).
- Causal discovery / DAG learning.
- Mediation analysis and surrogate endpoints under DML.
- Off-policy evaluation / contextual bandits.
- LLM-based causal reasoning.
- Production deployment / ML-Ops for causal models.
- Survival / duration outcomes under DML.

See the research plan for the full out-of-scope list with rationale.