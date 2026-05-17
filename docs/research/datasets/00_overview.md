# Datasets for Double Machine Learning (Time Series & Panel) — Overview

**Topic:** Datasets supporting Double Machine Learning research and practice
for time-series and panel-data applications.

**Coverage:** 105 datasets across 4 topic files. Curated to match
the parent manuscript's chapters (FRED for ch07, 401(k) and LaLonde for ch08,
panel admin data for ch09, benchmark/DGP for methodology validation).

**Generated:** 2026-05-16.

## Scope

In scope:
- Macro time-series databases used as DML controls / outcomes (FRED, OECD, WDI, IMF, Eurostat, NBER macro history)
- Long-running administrative / panel surveys (PSID, NLSY, SIPP, MEPS, CPS, ACS, HRS, EU panels)
- Published-paper replication packages where DML has been applied or is canonically applicable (401(k), LaLonde, Card-Krueger, Acemoglu colonial origins, Chetty mobility, Banerjee-Duflo microfinance, Oregon HIE)
- Methodology benchmarks and synthetic data-generating processes (ACIC, IHDP, Twins, JOBS II, DoubleML/EconML/CausalML/GRF/DoWhy/CATENets/CSuite/RealCause example data)

Out of scope (deliberately excluded — these would over-scope or pollute the dossier):
- Pure NLP / vision ML benchmarks (GLUE, ImageNet) — no causal-inference relevance
- Forecasting-only competition data (M3/M4/M5, Kaggle forecasting challenges) — different evaluation conventions
- Pure financial OHLC stock-price archives — not used as DML treatment/control in the manuscript
- Survival / duration outcome microdata (specialized literature; deferred)

## File map

| File | Topic | Anchors | Entries |
|---|---|---|---|
| `01_fred_macro_timeseries.md` | FRED + Macro Time-Series | A1–A34 | 34 |
| `02_admin_panel_surveys.md` | Administrative / Panel Surveys | B1–B17 | 17 |
| `03_replication_packages.md` | Replication Packages (Real-World Studies) | C1–C33 | 33 |
| `04_benchmark_dgp.md` | Benchmark Datasets + Synthetic DGPs | D1–D21 | 21 |

## Glossary (dataset-specific terms)

- **ATE / ATT / ATU** — Average Treatment Effect (Total / on Treated / Untreated). Most DML replications target one of these estimands.
- **CATE / HTE** — Conditional Average Treatment Effect / Heterogeneous Treatment Effect. The estimand for which IHDP, Twins, and ACIC are designed.
- **DGP** — Data-Generating Process. A simulator producing data with known ground-truth ATE (used in IHDP, ACIC, DoubleML's `make_plr_CCDDHNR2018` family).
- **PLR** — Partially Linear Regression model. The most common DML estimator (Robinson 1988; Chernozhukov et al. 2018).
- **IRM** — Interactive Regression Model. DML for binary treatment with full interaction between treatment and covariates.
- **IIVM** — Interactive IV Model. DML with instrumental variables.
- **NSW** — National Supported Work training program (origin of LaLonde 1986 / Dehejia-Wahba 1999 data).
- **SIPP** — Survey of Income and Program Participation. Origin of the Poterba-Venti-Wise 401(k) eligibility data.
- **Real-time vintage** — Macro data as first published, before later revisions; available from ALFRED or Philly Fed RTDSM. Necessary for honest backtest-style DML applications.
- **Replication package** — Bundle of code + data published with an AER/QJE/Restud paper, hosted on Harvard Dataverse / openICPSR / AEA archive.
- **Semi-synthetic** — Real covariates + simulated outcome (preserves realistic confounder structure while giving known ground-truth ATE). IHDP is the canonical example.
- **license: unknown** — honest signal. Means the source page did not declare a license; the dossier does NOT guess by inference from the source category.
- **auth_required: Y** — downloading requires login / API key / registration / signed agreement. Notes whether agents can pull the data unattended.

## How to use this dossier

- **Looking for the canonical version of a named dataset?** Each entry's `Source` URL is the dossier-of-record. Don't go from memory.
- **Need a license-clean dataset for redistribution?** Use the `Size+License` bullet. Treat `license: unknown` as effectively non-redistributable.
- **Want a DML-ready dataset (no preprocessing)?** Look in `04_benchmark_dgp.md` — the DoubleML/EconML/CausalML datasets are pre-cleaned for the library's example notebooks.
- **Want a real replication study to extend?** Look in `03_replication_packages.md` — the Harvard Dataverse / NBER / openICPSR entries are the canonical hosts.
- **Want macro covariates for time-series DML?** Look in `01_fred_macro_timeseries.md`.

For lookup recipes by question type, see `README.md`.

## Cross-reference

Paper-side synthesis for this topic: see `../papers/` (159 primary-source entries
organized by claim_family: theory_partialing, theory_orthogonality, estimator_xs,
estimator_temporal, inference_ts, replication_applied, tooling, survey).
