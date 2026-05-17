# Research Plan: Double Machine Learning for time series and panel data

This research builds a dual-audience dossier (human-readable + LLM-agent-grounding)
of primary sources for Double Machine Learning (DML) applied to time-series and
panel data. Coverage: the FWL / Robinson partial-out roots, the Neyman-orthogonality
theory, cross-sectional DML estimators, temporal & dynamic DML estimators
(recursive g-estimation, TemporalPLR, rolling-window, panel DML), time-series
inference machinery (HAC, time-series CV, blocked CV, stationarity), real-world
replications (401(k), LaLonde/Dehejia, FRED-style macro applications), and the
implementing software toolchain. Target: ~150–200 verified primary sources
across 7 sub-areas, with **A4 (Temporal & dynamic DML)** and
**A6 (Real-world replications & applied DML)** deliberately weighted heavier
than the others (≥25 entries each vs ≥15 elsewhere). Output consumed by the
`double_ml_time_series` book manuscript revision (cite-mining) and by future
Claude agents working in that repo.

## Sub-areas

- A1. FWL theorem, Robinson estimator, and partial-out theory
  - Source types: foundational econometrics journals (Econometrica, JASA, AnnStat), classic textbooks, lecture notes, working papers
  - Notes: IN — Frisch–Waugh–Lovell partial-out, Robinson's root-N partially linear regression, double-residual / orthogonalization viewpoint, exposition of FWL in the DML era (Chernozhukov "FWL revisited"). OUT — modern variants of FWL for generalized models (deferred to A2/A3).

- A2. Neyman orthogonality & semiparametric DML theory
  - Source types: arXiv, top stats/econ journals (Econometrica, AnnStat, JASA), AISTATS/NeurIPS/ICML for ML-flavored treatments
  - Notes: IN — Neyman orthogonality and its derivation, sample-splitting / cross-fitting, semiparametric efficiency bounds, Locally Robust Moment Conditions (Chernozhukov-Escanciano-Ichimura-Newey-Robins), influence functions for DML, asymptotic theory under high-dimensional nuisance. OUT — full semiparametric theory outside DML (Bickel-Klaassen-Ritov-Wellner) — deferred reference only.

- A3. Cross-sectional DML estimators & heterogeneous treatment effects (HTE)
  - Source types: arXiv, NeurIPS/ICML/AISTATS, Biometrika, JASA, Econometrica, Annals of Statistics
  - Notes: IN — Partially Linear Regression (PLR), Interactive Regression Model (IRM), Interactive IV Model (IIVM), R-learner, X-learner, DR-learner, T-learner; causal forests (Wager–Athey, Athey–Imbens–Wager generalized random forests); meta-learners. OUT — heavy DAG-side semantics (deferred), pure DiD / synthetic control (separate plan).

- A4. **Temporal & dynamic DML estimators — heavier weight**
  - Source types: arXiv (econometrics & ML.LG / stat.ML), Econometrica, AnnStat, NBER working papers, JMLR, NeurIPS causal-ML workshops, EconML / Microsoft Research preprints
  - Notes: IN — Lewis & Syrgkanis recursive (orthogonal) g-estimation for dynamic treatment effects; TemporalPLR / partial-out DML with lagged treatment controls; rolling-window DML and walk-forward variants; PanelDML (DML for panel data with fixed effects); period-specific θ_t estimation; sequential conditional ignorability; Robins' g-methods reinterpreted under DML; high-dimensional panel methods. OUT — pure forecasting (ARIMA / state-space / TS-transformer) — see exclusions.

- A5. Time-series inference machinery for DML
  - Source types: Econometrica, JASA, Annals of Statistics, NBER working papers, arXiv (econ.EM)
  - Notes: IN — HAC / Newey-West and modern updates (Andrews 1991, Kiefer-Vogelsang, fixed-b asymptotics), Driscoll-Kraay panel HAC, time-series cross-validation (blocked, walk-forward, purged-CV), stationarity & unit-root diagnostics (KPSS, ADF, PP), cointegration considerations for panel DML, long-run-variance estimation. OUT — pure spectral methods / wavelet inference — deferred reference only.

- A6. **Real-world replications & applied DML — heavier weight**
  - Source types: AER / QJE / J. Econometrics / J. Pol. Econ.; replication packages on Harvard Dataverse, AEA replication archive, Zenodo, GitHub; FRED / NBER macro data; SIPP/PSID administrative data
  - Notes: IN — 401(k) eligibility / participation studies (Poterba-Venti-Wise, Chernozhukov-Hansen 401(k) replication); LaLonde 1986 / Dehejia-Wahba 1999 training-program reanalysis under DML; FRED-style macroeconomic confounder control (real-time vintages, MIDAS-like considerations); panel DML applications in finance / labor / health / policy; replication studies that explicitly re-estimate published causal effects under DML. OUT — pure economic-history descriptive papers without a causal identification claim.

- A7. Implementations & software tooling
  - Source types: GitHub repos (canonical URL), JSS / JStatSoft / NeurIPS-D&B publications, official docs sites, vendor blogs (Microsoft Causality, AWS Causal, Google causal-ML)
  - Notes: IN — DoubleML (R + Python, Bach-Chernozhukov-Kurz-Spindler), EconML (Microsoft, including dynamic-treatment estimators), DoWhy (Sharma–Kiciman), CausalML (Uber), scikit-learn integrations for DML, PyTorch-based causal estimators, R `grf` (generalized random forests). OUT — pure forecasting libraries (Prophet, sktime, NeuralForecast).

## Out-of-scope

The following are deliberately excluded — each would over-scope this plan or
has methodologically distinct evaluation conventions that would dilute the
dossier:

- **Synthetic Control & Synthetic DiD** (Abadie et al.) — adjacent to panel
  causal inference but does not use ML nuisance functions; deserves its own
  research plan.
- **Difference-in-Differences with staggered adoption** — large active
  literature (Callaway-SantAnna, de Chaisemartin-d'Haultfœuille, Borusyak et
  al.) but evaluation conventions differ; defer.
- **Causal discovery / DAG learning** (PC, GES, NOTEARS) — different end-goal
  (structure learning, not effect estimation).
- **Pure time-series forecasting** (ARIMA, state-space, Prophet, neural
  forecasting, transformers for TS) — different evaluation conventions
  (MAPE / sMAPE / quantile loss), no treatment effect.
- **Mediation analysis & surrogate endpoints** under DML — small adjacent
  literature; defer.
- **Off-policy evaluation / contextual bandits** — DML appears here but
  evaluation conventions differ (regret-based); defer.
- **LLM-based causal reasoning** — methodologically distinct, agent-side
  rather than estimator-side.
- **Production deployment / ML-Ops for causal models** — different audience
  (engineering, not statistical); defer.
- **Survival / duration outcomes under DML** — specialized literature
  (Westling, Tchetgen-Tchetgen et al.); defer.
- **Pure CATE benchmarking & competition papers** — important but their
  evaluation conventions emphasize PEHE / R-loss tournaments rather than the
  primary-source theory this dossier targets.

## Claim family taxonomy

Used by `bib_ledger.yml` to classify every entry. Stable across pass-1 +
pass-2; matched to the 7 sub-areas above plus a fallback category.

- `theory_partialing` — A1; FWL/Robinson partial-out machinery, foundational
  theorems.
- `theory_orthogonality` — A2; Neyman orthogonality, semiparametric theory,
  sample-splitting / cross-fitting, influence-function / locally-robust
  theory.
- `estimator_xs` — A3; cross-sectional DML estimators (PLR / IRM / IIVM,
  meta-learners) and CATE/HTE estimators (causal forests, R-learner,
  X-learner, DR-learner).
- `estimator_temporal` — A4; temporal & dynamic DML — recursive g-estimation,
  TemporalPLR, rolling-window, PanelDML, dynamic treatments.
- `inference_ts` — A5; HAC / Newey-West / Driscoll-Kraay, time-series CV,
  stationarity diagnostics, long-run-variance estimation.
- `replication_applied` — A6; real-world replications (401(k), LaLonde,
  Dehejia), FRED macro applications, applied panel DML, replication
  packages.
- `tooling` — A7; implementations & software (DoubleML, EconML, DoWhy,
  CausalML, grf).
- `survey` — fallback for survey / review / pedagogical articles that span
  multiple sub-areas (e.g., Athey-Imbens 2019 "what economists should know
  about ML").

## Known landmark papers

These are pre-known canonical references that anchor the dossier. The
gather skill should find and verify them but should not claim discovery
credit for them. Bibkeys use the existing `bibliography.bib` style
(`{firstauthor_lowercase}{year}{slug}`) for easy cite-mining migration.

- `chernozhukov2018double`: the headline DML / debiased ML paper (The
  Econometrics Journal, 2018) — anchor for A2 and A3.
- `chernozhukov2017double`: the arXiv preprint version (1608.00060)
  predating the 2018 publication.
- `chernozhukov2022locally`: locally robust semiparametric estimation
  (Econometrica 2022) — the orthogonality theory follow-up.
- `frisch1933partial` + `lovell1963seasonal`: the FWL theorem (anchor A1).
- `robinson1988rootn`: Robinson's root-N partially linear regression
  (Econometrica 1988) — **NOT in current `bibliography.bib`** despite being
  referenced in the book's Robinson-estimator chapter; gap to surface.
- `newey1987`: Newey–West HAC (Econometrica 1987) — anchor A5.
- `lewis2020dynamic`: Lewis & Syrgkanis 2020 dynamic treatment effects
  with DML / recursive g-estimation — **NOT in current `bibliography.bib`**
  despite being referenced as the canonical reference for the
  `TemporalPLRDML` estimator in the book's CLAUDE.md; gap to surface.
- `wager2018estimation`: causal forests (JASA 2018) — anchor for the HTE
  part of A3.
- `nie2021quasi`: R-learner (Biometrika 2021) — anchor for meta-learners
  in A3.
- `bach2022doubleml`: DoubleML R/Python package paper (JSS 2022) — anchor
  for A7.
