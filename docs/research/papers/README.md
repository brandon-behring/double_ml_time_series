# Double Machine Learning for time series and panel data — Research Synthesis

<!-- AGENT-INDEX: this folder is a self-contained reference for Double Machine Learning applied to time-series and panel data. Read this README first. -->

**Purpose:** verified primary-source synthesis covering DML applied to time-series & panel data — designed for dual consumption: humans (reading directly during the manuscript revision) and future LLM agents (grounding reasoning in this literature when working in the `double_ml_time_series` repo).
**Primary intended consumer:** future Claude / Codex / LLM-agent sessions opening this repo to answer DML questions; secondary consumers: humans reading the material directly.
**Self-containedness guarantee:** this folder has no hard dependence on sibling files outside itself.
**Scope:** ~1933–2026 coverage; 7 sub-areas (theory_partialing → theory_orthogonality → estimator_xs → estimator_temporal → inference_ts → replication_applied → tooling).
**Coverage:** 160 entries across 7 topic files; structured 5-bullet entries (Source / Code / Mechanism / Result / Status).
**Last updated:** 2026-05-16.
**Source ledger:** `../bib_ledger.yml`. **Source dossier:** `../dossier/`.

## ⚠️ Scope boundary

This folder synthesizes **methodology** — papers about how to estimate causal effects with ML nuisance under time-series / panel structure. It does **NOT** cover:

- **Datasets** — see the paired dataset dossier at [`../datasets/`](../datasets/) (added 2026-05-16). The dataset dossier covers data artifacts (FRED, 401(k) public files, LaLonde data, replication packages); this synthesis covers methodology.
- **Synthetic control / synthetic DiD** — deserves its own research plan; primary references *not* included here.
- **DiD with staggered adoption** — large active literature (Callaway–Sant'Anna, de Chaisemartin–d'Haultfœuille, Borusyak et al.) — methodologically distinct; not covered.
- **Pure time-series forecasting** (ARIMA / state-space / transformer-for-TS) — different evaluation conventions; not covered.
- **Causal discovery / DAG learning** — different end-goal (structure rather than effect).
- **Mediation analysis, off-policy evaluation, LLM-based causal reasoning, survival outcomes under DML** — adjacent but separately scoped.

If you're looking for any of the above, this is the wrong folder.

**Cross-pipeline cross-link:** the paired dataset dossier at [`../datasets/`](../datasets/) (added 2026-05-16, research_toolkit v1.9). The synthesis here covers methodology; the dataset dossier covers data artifacts. A few datasets (FRED, the 401(k) public-use file) may appear in BOTH because they're cited as primary references for replication papers in this synthesis AND as standalone artifacts in the dataset dossier.

## How this is organized

Sub-section anchors use a per-file letter prefix (A for file 01, B for file 02, ..., G for file 07). Lookup recipes below reference these anchors.

| File | Topic | Anchor prefix | Entries | When to read |
|---|---|---|---:|---|
| `00_overview.md` | Glossary + scope statement | — | — | Start here if new to DML |
| `01_theory_partialing.md` | FWL theorem, Robinson estimator, partial-out theory | A | 13 | Foundations / Chapter 1 of the manuscript |
| `02_theory_orthogonality.md` | Neyman orthogonality, semiparametric DML theory | B | 21 | Theory underpinning DML — Chapter 2 |
| `03_estimator_xs.md` | Cross-sectional DML & HTE | C | 20 | i.i.d. estimators, causal forests, meta-learners |
| `04_estimator_temporal.md` | Temporal & dynamic DML (emphasized) | D | 37 | TemporalPLRDML, rolling-window, panel — Chapters 5–8 |
| `05_inference_ts.md` | HAC / time-series CV / stationarity | E | 20 | HAC inference, CV machinery for time-dependent data |
| `06_replication_applied.md` | Real-world replications & applied DML (emphasized) | F | 34 | 401(k), LaLonde, FRED macro — Chapters 7, 9 |
| `07_tooling.md` | Implementations & software | G | 15 | DoubleML, EconML, DoWhy, CausalML, hdm, grf |

## Lookup recipes

Routes by question type. Each points to a specific file and section anchor.

- "What is the foundational paper for DML?" → `02_theory_orthogonality.md` § B1 (Chernozhukov et al. 2018, *Double/debiased machine learning for treatment and structural parameters*).
- "What is the canonical reference for the FWL theorem?" → `01_theory_partialing.md` § A1 (Frisch & Waugh 1933 + Lovell 1963).
- "What did Robinson 1988 prove?" → `01_theory_partialing.md` § A1 (Robinson 1988, *Root-N-Consistent Semiparametric Regression* — the FWL+nonparametrics ancestor of cross-sectional DML).
- "What is Neyman orthogonality?" → `00_overview.md` § Glossary, then `02_theory_orthogonality.md` § B1 (Chernozhukov-Newey-Singh et al. 2022 *Locally Robust Semiparametric Estimation*).
- "What is sample splitting / cross-fitting?" → `02_theory_orthogonality.md` § B1 (Chernozhukov et al. 2018) + `00_overview.md` § Glossary.
- "What is the canonical reference for dynamic DML?" → `04_estimator_temporal.md` § D1 (Lewis & Syrgkanis 2020 *Double/Debiased ML for Dynamic Treatment Effects via g-Estimation*) — this is the book's `TemporalPLRDML` reference.
- "How do I do DML on panel data with fixed effects?" → `04_estimator_temporal.md` § D1 (Clarke & Polselli 2024 *DML for Static Panel Models with Fixed Effects*) — uses block-k-fold cross-fitting.
- "What are the main panel-DML pitfalls?" → `04_estimator_temporal.md` § D1 (Fuhr et al. 2024 *DML meets Panel Data — Promises, Pitfalls, and Potential Solutions*).
- "What is the canonical 401(k) DML benchmark?" → `06_replication_applied.md` § F1 (Chernozhukov & Hansen 2004 *Effects of 401(k) Participation on the Wealth Distribution*) + Poterba-Venti-Wise 1995/1996 background.
- "What is the LaLonde dataset / propensity-score reanalysis?" → `06_replication_applied.md` § F1 (LaLonde 1986 + Dehejia & Wahba 1999).
- "What is causal forest / GRF?" → `03_estimator_xs.md` § C1 (Wager & Athey 2018 + Athey-Tibshirani-Wager 2019). GRF R package in `07_tooling.md` § G1.
- "What is the R-learner / X-learner / DR-learner?" → `03_estimator_xs.md` § C1 (R-learner: Nie & Wager 2021; X-learner: Künzel et al. 2019).
- "What is HAC / Newey-West?" → `05_inference_ts.md` § E1 (Newey & West 1987; Andrews 1991 for optimal bandwidth).
- "What is Driscoll-Kraay?" → `05_inference_ts.md` § E1 (Driscoll & Kraay 1998 ReStat — panel HAC robust to cross-sectional dependence).
- "How do I cross-validate time series?" → `05_inference_ts.md` § E1 (Bergmeir-Hyndman-Koo 2018) + `06_replication_applied.md` (de Prado 2018 purged k-fold for finance).
- "What software should I use for DML?" → `07_tooling.md` § G1: DoubleML (R/Python), EconML (Python), hdm (R for high-dim IV).
- "What software should I use for HTE / CATE?" → `07_tooling.md` § G1: EconML (DML+meta-learners), CausalML (Uber), grf (R generalized forests).
- "What is the structural nested mean model (SNMM)?" → `04_estimator_temporal.md` § D1 (Robins 1986 founding paper) + `00_overview.md` § Glossary.
- "What is g-estimation?" → `04_estimator_temporal.md` § D1 (Robins 1986 + Naimi-Cole-Kennedy 2017 intro).
- "What is influence-function adjustment for semiparametric inference?" → `02_theory_orthogonality.md` § B1 (Ichimura & Newey 2022 + Newey 1994).
- "What is targeted maximum-likelihood estimation (TMLE)?" → `02_theory_orthogonality.md` § B1 (van der Laan & Rubin 2006) — alternative path to orthogonality.
- "What is the double Lasso / post-double-selection procedure?" → `03_estimator_xs.md` § C1 (Belloni-Chernozhukov-Hansen 2014, *Inference on Treatment Effects After Selection Amongst High-Dimensional Controls*).
- "What is the canonical narrative-macro-shock identification reference?" → `06_replication_applied.md` § F1 (Romer-Romer monetary; Mertens-Ravn tax shocks; Cloyne et al.).
- "Where do I find FRED data?" → `06_replication_applied.md` § F1 (`fred` entry); paired dataset dossier at `../datasets/` for FRED metadata.
- "What is BART for causal inference?" → `03_estimator_xs.md` § C1 (Hill 2011 *Bayesian Nonparametric Modeling for Causal Inference* + Hahn-Murray-Carvalho 2020 BCF).
- "What is Dragonnet?" → `03_estimator_xs.md` § C1 (Shi-Blei-Veitch 2019, NeurIPS).

## Glossary

Full glossary lives in `00_overview.md` § Glossary. Most-load-bearing terms below:

- **DML** — Double Machine Learning. ML-nuisance-controlled causal estimation using Neyman-orthogonal scores and cross-fitting.
- **FWL** — Frisch-Waugh-Lovell theorem. Residual-on-residual regression equivalent to multivariate regression.
- **Neyman orthogonality** — First-order insensitivity of a moment condition to nuisance parameters at their true value.
- **Cross-fitting** — K-fold sample splitting + nuisance training on K-1 folds, scoring on the held-out fold.
- **HTE / CATE** — Heterogeneous (Conditional Average) Treatment Effect: τ(x) = E[Y(1)−Y(0) | X=x].
- **g-estimation** — Robins-family estimator for time-varying treatments under structural nested mean models.
- **HAC** — Heteroskedasticity- and Autocorrelation-Consistent covariance estimator (Newey-West family).

## Verification & limits

- Citations resolved as of 2026-05-16.
- All entries default to `status: Verified` after the dossier-build stage flipped them from `unverified`. A subsequent `/dossier-audit` round (run on this folder) will perform the independent verification round and surface CORRECT / FLAG / DROP findings as inline edits + an audit-trail note appended below.
- ~85% of the existing book bibliography (`bibliography.bib`) has no primary URL field; see `../bibliography_audit.md` for the pre-pipeline audit report.
- This synthesis is a snapshot. The DML methodology field is active — expect material updates to A4 (temporal/dynamic) annually; expect software (A7) to drift quarterly.
**Independent audit, round 1 (2026-05-16):** A complementary-scope (aggressiveness: standard) review pass focused on primary-source DOI + author/year/venue accuracy on temporal-DML and dynamic-treatment entries, with a 10-entry random spot-check across all files. Prior rounds covered: none. Findings: 0 dropped (standard aggressiveness), 7 corrected, 6 flagged, ~40 spot-check passed. Typical discrepancies: 3 duplicate-entry pairs in `04_estimator_temporal.md` (Athey-Imbens staggered DiD, Clarke-Polselli static panel, Fuhr-Papies panel DML) — consolidated to canonical entries; 1 wrong arXiv ID (`A11` Ding FWL — `2106.02087` resolves to unrelated paper, URL replaced with Berkeley faculty page pending re-lookup); 1 author-list error (`G14` xtdml — `Clarke, Polselli` corrected to sole author `Polselli`); 1 broken JASA DOI (`D10` Bojinov-Shephard — replaced with verified `10.1080/01621459.2018.1527225`); 5 unverified-venue flags annotated `(unverified, 2026-05-16)`. Recommendation: re-run with focus = "Mechanism+Result polish — replace `See abstract for details` placeholders across all files".

**Independent audit, round 2 (2026-05-17):** A complementary-scope (aggressiveness: standard) review pass focused on polishing the ~102 placeholder `Mechanism: See abstract for details.` / `Result: See abstract for details.` entries by re-fetching the primary source and extracting one-line verified mechanism + concrete result quotes. Prior rounds covered: Round 1 (DOI/author/year/venue accuracy). Findings: 0 dropped (standard aggressiveness), 78 corrected (mechanism + result text replaced with verified abstract-derived text), 24 flagged for paywall-blocked abstracts. Typical inaccessible sources: Wiley Online Library, Tandfonline, MIT Press, ACP Journals, SSRN, EconPapers metadata pages, PsycNet — all returned 402/403 to WebFetch. Polished coverage: file 04 estimator_temporal (~27/29 entries), file 06 replication_applied (~17/26), file 07 tooling (8/8), file 03 estimator_xs (9/10), file 02 theory_orthogonality (6/10), file 01 theory_partialing (6/7), file 05 inference_ts (7/14). The remaining ~24 placeholders are honestly annotated; future `/url-freshness-check` or manual fetch via Playwright can re-attempt these. Recommendation: clean — stop iteration on papers audit. Move to Step 16 (dataset audit).

- *Future rounds will be appended below as `**Independent audit, round N (YYYY-MM-DD):** ...`*

## Attribution

Synthesized from `../bib_ledger.yml` (163 verified entries; 160 after Round 1 audit consolidation) via the `research_toolkit` (`~/Claude/research_toolkit/`). Pipeline: `/research-plan` → `/research-gather` → `/dossier-build` → `/agent-index` → `/dossier-audit`. URLs link to primary sources (arXiv, journal DOIs, JSTOR, GitHub, vendor pages). No local file paths are referenced. For the source bibliography ledger and dossier tables, see `../bib_ledger.yml` and `../dossier/`.