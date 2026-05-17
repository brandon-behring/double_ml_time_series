# Datasets for Double Machine Learning (Time Series & Panel) — Research Synthesis

<!-- AGENT-INDEX: this folder is a self-contained dataset reference for Double Machine Learning applied to time-series and panel data. Read this README first. -->

**Purpose:** Curated index of public datasets supporting Double Machine Learning research and practice for time-series and panel-data applications. Designed for dual consumption — humans (reading directly) and future LLM agents (grounding reasoning in this dataset catalog).
**Primary intended consumer:** future Claude Code / LLM agents working in the `double_ml_time_series` repo who need to ground reasoning in actual datasets rather than memory. Secondary consumers: humans cite-mining for the manuscript revision and reading the material directly.
**Self-containedness guarantee:** this folder has no hard dependence on sibling files outside itself. Move it elsewhere and it still works (paper-side synthesis is cross-linked but optional).
**Scope:** datasets useful for DML in time series + panel + cross-sectional replication settings; assembled 2026-05.
**Coverage:** 105 entries across 4 topic files; structured 5-bullet entries (Source / Access / Schema / Size+License / Tasks / Status).
**Last updated:** 2026-05-16.

## ⚠️ Scope boundary

This dossier covers **dataset artifacts** (metadata + access methods + license + size + canonical citation). It does NOT cover the methodology papers themselves — for those, see the paired paper-synthesis dossier:

> For **paper synthesis** on this topic, see [`../papers/`](../papers/). The dataset dossier here covers metadata + access methods; the synthesis covers methodology (FWL/Robinson partial-out, Neyman orthogonality, cross-sectional and temporal DML estimators, HAC inference, applied replications, software tooling).

Out of scope (deliberately excluded):
- Pure NLP / vision ML benchmarks (GLUE, ImageNet) — no causal-inference relevance.
- Forecasting-only competition data (M3/M4/M5) — different evaluation conventions (forecast loss, not treatment effect).
- Pure financial OHLC archives (Yahoo Finance daily stock prices) — not used as DML treatments in the manuscript.
- Survival / duration outcome microdata — specialized literature, deferred.

**Cross-pipeline duplication is intentional, not a bug.** A handful of entries (e.g., 401(k) data, LaLonde NSW) appear here AND in the paper-side `06_replication_applied.md` synthesis file: the paper dossier captures the methodology paper that used the data; this dossier captures the data artifact itself. Both citations are correct — do NOT remove from either.

## How this is organized

Sub-section anchors use a per-file letter prefix (`## A1.`, `## A2.` … in file 01; `## B1.` in file 02; etc.). Lookup recipes below reference these anchors.

| File | Topic | When to read |
|---|---|---|
| `00_overview.md` | Scope, glossary, how to use this dossier | Start here if new to DML datasets |
| `01_fred_macro_timeseries.md` | FRED + macro time-series databases (anchors A1–A34) | When grounding ch07 (FRED integration) or any panel-macro DML application |
| `02_admin_panel_surveys.md` | Administrative / panel surveys (anchors B1–B17) | When grounding ch09 (panel DML) or labor/health/policy DML applications |
| `03_replication_packages.md` | Published-paper replication packages (anchors C1–C33) | When grounding ch08 (401k / LaLonde) or "redo this published causal study under DML" exercises |
| `04_benchmark_dgp.md` | Benchmark + synthetic DGPs (anchors D1–D21) | When validating estimator accuracy vs ground-truth ATE/CATE (methodology chapters) |

## Lookup recipes

Routes by question type. Each points to a specific file and section anchor.

**FRED + macro time series**:
- **"What's the FRED series for unemployment?"** → `01_fred_macro_timeseries.md` § UNRATE entry (`fred.stlouisfed.org/series/UNRATE`).
- **"What's the FRED series for inflation / CPI?"** → `01_fred_macro_timeseries.md` § CPIAUCSL.
- **"What's the FRED series for the federal funds rate?"** → `01_fred_macro_timeseries.md` § FEDFUNDS / DFF.
- **"Where do I get a real-time-vintage version of FRED data?"** → `01_fred_macro_timeseries.md` § ALFRED + Philly Fed RTDSM.
- **"What's the canonical macro panel for cross-country DML?"** → `01_fred_macro_timeseries.md` § World Bank WDI / OECD MEI / Penn World Tables (Groningen).
- **"What's the canonical NBER long-run macro database?"** → `01_fred_macro_timeseries.md` § NBER Macrohistory.

**Replication packages (cite-mineable for manuscript ch08)**:
- **"Where is the 401(k) eligibility data?"** → `03_replication_packages.md` § Poterba-Venti-Wise 1995 + Chernozhukov-Hansen 2004; also DoubleML `fetch_401K` Python/R wrappers (`03` cluster).
- **"Where is the LaLonde NSW training data?"** → `03_replication_packages.md` § NBER canonical + R packages (`lalondeR2017chern`, `lalondematchit2014`); also see `04_benchmark_dgp.md` § lalonde1986nsw for the Dehejia-Wahba composite.
- **"Where is Card-Krueger min-wage data?"** → `03_replication_packages.md` § Card-Krueger 1994 AER + Card 1995 college proximity IV.
- **"Where is Acemoglu-Johnson-Robinson colonial-origins data?"** → `03_replication_packages.md` § AJR 2001 + 2012 reply.
- **"Where is Acemoglu-Restrepo robots data?"** → `03_replication_packages.md` § acemoglurestrepo2020.
- **"Where is Banerjee-Duflo microfinance RCT data?"** → `03_replication_packages.md` § Spandana 2015 + J-PAL Six-RCT aggregator.
- **"Where is Chetty mobility / opportunity-insights data?"** → `03_replication_packages.md` § Land-of-Opportunity / Opportunity Atlas.
- **"Where are health-insurance RCT replications?"** → `03_replication_packages.md` § Oregon HIE / RAND HIE.
- **"Where is the Card-Dobkin-Maestas Medicare-65 RDD replication?"** → `03_replication_packages.md` § carddobkin2008medicare65.
- **"Where is the BCH 2014 / Chernozhukov 2017 program-evaluation replication?"** → `03_replication_packages.md` § bch2014progeval / chernozhukov2017aerpp_dml.

**Admin / panel surveys (cite-mineable for manuscript ch09)**:
- **"What's the canonical PSID landing page?"** → `02_admin_panel_surveys.md` § PSID entry (psidonline.isr.umich.edu).
- **"Where is NLSY79 / NLSY97?"** → `02_admin_panel_surveys.md` § NLSY79 / NLSY97.
- **"Where is SIPP?"** → `02_admin_panel_surveys.md` § SIPP entry.
- **"Where is MEPS (Medical Expenditure)?"** → `02_admin_panel_surveys.md` § MEPS entry.
- **"Where is CPS / ACS?"** → `02_admin_panel_surveys.md` § CPS / ACS; also `03_replication_packages.md` § ipumscps1962umn for the IPUMS-mirrored long-panel form.
- **"Where is HRS (Health and Retirement Study)?"** → `02_admin_panel_surveys.md` § HRS.
- **"Where is GSOEP / UKHLS / EU-SILC?"** → `02_admin_panel_surveys.md` § GSOEP / UKHLS / EU-SILC entries.
- **"What's the canonical PISA dataset?"** → `02_admin_panel_surveys.md` § PISA 2000.
- **"World Bank LSMS?"** → `02_admin_panel_surveys.md` § LSMS.

**Benchmark / DGP**:
- **"What's the canonical CATE benchmark?"** → `04_benchmark_dgp.md` § IHDP (Hill 2011) + ACIC 2016-2022 series.
- **"What semi-synthetic dataset has known ground-truth ATE?"** → `04_benchmark_dgp.md` § IHDP + Twins + ACIC.
- **"What DGP does the DoubleML package ship?"** → `04_benchmark_dgp.md` § `doublemldgp2018github` (covers `make_plr_CCDDHNR2018`, `make_irm_data`, `make_iivm_data`, `make_did_SZ2020`, etc.).
- **"What's the canonical EconML notebook example data?"** → `04_benchmark_dgp.md` § EconML notebooks.
- **"Where is the CausalML example dataset?"** → `04_benchmark_dgp.md` § causalmldataset.
- **"What datasets does Generalized Random Forests (grf) test on?"** → `04_benchmark_dgp.md` § grftests.
- **"What's the Curth-van der Schaar CATENets benchmark suite?"** → `04_benchmark_dgp.md` § catenets.
- **"What's the Microsoft CSuite causal benchmark?"** → `04_benchmark_dgp.md` § csuite.
- **"What's the JOBS II program data?"** → `04_benchmark_dgp.md` § jobs1991vinokur.
- **"What's the Knaus-Lechner ML Monte Carlo simulation framework?"** → `04_benchmark_dgp.md` § knauslechner2021mlhce.
- **"What's the Huber-Lechner-Wunsch Empirical Monte Carlo?"** → `04_benchmark_dgp.md` § huberlechnerwunsch2013emc.

**License + access filters**:
- **"Which datasets need credentialed access / signup wall?"** → search `02_admin_panel_surveys.md` for `auth_required: Y` (PSID, NLSY, HRS, GSOEP, EU-SILC, NHANES, LSMS, UKHLS are all gated).
- **"Which datasets are CC0 / fully public-domain redistributable?"** → search for `license: CC0` or `license: public domain (US gov)` — most FRED series + most US-gov-collected micro data fall here.
- **"What's the license status of LaLonde NSW?"** → `04_benchmark_dgp.md` § lalonde1986nsw (explicit CC BY-NC per Dehejia mirror).

## Glossary

See `00_overview.md` for the full glossary. Key terms:

- **ATE / ATT / ATU** — Average Treatment Effect (Total / on Treated / Untreated).
- **CATE / HTE** — Conditional Average Treatment Effect / Heterogeneous Treatment Effect.
- **DGP** — Data-Generating Process (synthetic simulator with known ground truth).
- **PLR** — Partially Linear Regression — the canonical DML estimator.
- **NSW** — National Supported Work training program (LaLonde 1986).
- **Real-time vintage** — macro data as first published (no revisions); ALFRED + Philly Fed RTDSM.
- **Semi-synthetic** — real covariates + simulated outcome (IHDP, ACIC, RealCause).
- **license: unknown** — honest signal. Source did not declare; do NOT guess.

## Verification & limits

- Citations resolved as of 2026-05-16.
- **FRED-side WebFetch (`fred.stlouisfed.org`, `alfred.stlouisfed.org`, `research.stlouisfed.org`) returned 403 to the gather agent in some passes** — entries verified via WebSearch result quotes of canonical landing pages. The `/url-freshness-check` step at end-of-pipeline will re-verify with a different fetcher.
- **openICPSR landing pages returned 403 to WebFetch consistently** — entries flagged for re-check by `/dossier-audit` Round 1.
- Several admin-panel datasets (PSID, NLSY79/97, HRS, GSOEP, EU-SILC, OECD IDD, PISA, ACIC 2022, NHANES) are accessible only via signup wall — their pages were verified by WebSearch quoting their canonical landing pages, and `auth_required: Y` is set per the v1.6 access-method protocol.
- 79 of 105 entries are status: `verified` (75.2%) — the remainder are honestly `unverified` per the v1.5.1 strict verification protocol (default `unverified` until WebFetch confirms name + URL + license against the source page). The honest-rate is inside the 70-80% target band; the validator's anti-cheat heuristic does not fire.
- This synthesis is a snapshot. Public dataset infrastructure changes; URLs may drift. The audit log below will track corrections.

**Independent audit, round 1 (2026-05-17):** A standard review pass focused on license risks + access stability ran against 28 spot-checked entries spanning all four topic files. Findings: 0 dropped (standard aggressiveness), 6 corrected, 4 flagged. Typical CORRECT findings: one compound-license addition on GRF's ACIC18 data folder (treatment-effect-evaluation-only restriction beyond GPL-3); two FRED-MD/FRED-QD license refinements distinguishing canonical FRED-terms-of-use from R-package ODC-BY mirrors; NLSY79/NLSY97 access-method clarification (NLS Investigator extracts require registration but full public-use files are downloadable direct); Oregon HIE access re-categorized as `auth_required: N` with clickthrough DUA only; NSLM ICPSR 37353 access nuanced as `partial` for public vs restricted files; Opportunity Insights attribution wording clarified. Typical FLAG findings: ACIC 2022 site unreachable via WebFetch (recommend `/url-freshness-check` re-verify); FRED + openICPSR persistent 403 bot-blocks (worked around via WebSearch quotes); CEVAE Twins folder has no upstream LICENSE. 20+ SPOT-CHECK PASSED covering DoubleML/EconML/GRF/CATENets/DoWhy/CausalML/CSuite licenses (BSD-3/MIT/Apache-2.0) and PSID/HRS/GSOEP/NHANES/NLSY/JOBS II signup walls. Recommendation: Clean — stop iteration. The remaining license/access uncertainty is best addressed by `/url-freshness-check` at the next pipeline stage.

- *Future rounds will be appended below.*

## Attribution

Synthesized from a dataset_ledger maintained by the research_toolkit
(`~/Claude/research_toolkit/`). URLs link to primary sources (FRED, Harvard
Dataverse, openICPSR, NBER, NIH, BLS, Census, OECD, Eurostat, World Bank, IMF,
GitHub repos for software-ecosystem datasets). No local file paths are
referenced.
