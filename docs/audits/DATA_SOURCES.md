# Data sources of the audit folders: licences, attribution, and what is held

The audit folders hold three kinds of data files:

1. **Numbers computed in this repository**: estimates, standard errors, counts, moments, digests
   and simulation draws. They reproduce no third-party data. Their source is cited in the note
   beside them.
2. **Third-party content under a licence that grants redistribution.** It is attributed below.
3. **Facts and published values quoted with their source**: values of a paper's table, a
   benchmark's true effect by design, bibliographic records. They are cited to the publication.

A file that reproduces a provider's own rows where the provider's terms do not grant
redistribution is **not in the repository**; those files are listed at the end. The rule
concerns files that reproduce a source's rows, not a fact that a report states in its own words.

The terms of the providers named here were read on their own sites on 2026-11-26. For each of
them a record (terms URL, licence, what the files hold, verdict) is in
[`docs/research/dataset_ledger.yml`](../research/dataset_ledger.yml), under "Licence check of
2026-11-26". Sources whose files here hold only computed numbers and which are not named below
are cited in the note beside those files. This note is not legal advice.

## Attribution

| Source | Licence | Files here | What was done to it |
|---|---|---|---|
| [Cricsheet](https://cricsheet.org/), Twenty20 international records (JSON) | [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1.0/) | `2026-11-19_time_series_dml/prospective/cricket-history.csv.gz` and the `cricket-*.json` forecasts | One row per match and focal team was derived from the match files; the forecasts are computed from those rows |
| UK Power Networks, [SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d) (Low Carbon London) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/field/london/` | `events.csv` lists the 48 tariff events with household counts and mean consumption computed here; the other files hold estimates and simulation draws. No meter reading is included |
| Gabriel Kreindler, [replication package for "Peak-Hour Road Congestion Pricing: Experimental Evidence and Equilibrium Implications"](https://zenodo.org/records/10637043) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/field/bangalore/` | Coefficients and monthly counts were computed from the package; archive member names and digests are listed |
| Casey, Glennerster, Miguel and Voors, [Sierra Leone community-development replication data](https://zenodo.org/records/7411565) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/rolling/external-controls.json` (`sierra_leone`) | Estimates computed here |
| ZOZO, Inc., [Open Bandit Dataset](https://huggingface.co/datasets/zozonext/open-bandit) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/rolling/external-controls.json` (`retail`) | Estimates and three aggregate cells computed here |
| DellaVigna and Linos, nudge-units replication package (Econometric Society supplement to "RCTs to Scale: Comprehensive Evidence From Two Nudge Units", *Econometrica* 2022) | The [Econometric Society's permission](https://www.econometricsociety.org/publications/econometrica/permissions): readers may save and share a package's data as part of their own replication package after attribution (citing the paper and the package and saying how the data was obtained); other uses need permission. No data of the package is stored here | `2026-11-19_time_series_dml/rolling/external-controls.json` (`nudge_units`) | Estimates computed here |
| Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | the bike entries of `2026-11-21_evaluation_methods/reproduction/standalone-stdout.json.gz` | Estimates computed here |
| Hofmann and Siebenbrunner, [iFlex dynamic-price experiment](https://zenodo.org/records/8248802) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | the iFlex entries of the same archive | Estimates and arm counts computed here; one schedule row is printed |
| Klasnja and co-authors, [HeartSteps V1 data](https://github.com/klasnja/HeartStepsV1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | the `heartsteps` entry of the same archive | Estimates computed here |

Contains information from Cricsheet, which is made available under the ODC Attribution License.

## Citations that a provider prescribes

- **General Social Survey.** The files `2026-11-19_time_series_dml/prospective/gss-*` hold
  wave-level estimates computed here from the cumulative file; they hold no respondent record.
  Davern, Michael; Bautista, Rene; Freese, Jeremy; Herd, Pamela; and Morgan, Stephen L. General
  Social Survey 1972-2024. [Machine-readable data file]. Principal Investigator, Michael Davern;
  Co-Principal Investigators, Rene Bautista, Jeremy Freese, Pamela Herd, and Stephen L. Morgan.
  NORC ed. Chicago, 2026. 1 datafile and 1 codebook (2024 Release 3a).
- **Real GDP vintages.** `2026-11-19_time_series_dml/external/forward/gdp-vintages.csv` holds
  vintage dates and digests of the downloads, and no series value. U.S. Bureau of Economic
  Analysis, Real Gross Domestic Product [GDPC1], retrieved from FRED, Federal Reserve Bank of
  St. Louis; the vintages were read through its ALFRED archive.

## Facts and values quoted from publications

- `2026-11-19_time_series_dml/rolling/external-controls.json` (`effort`) quotes the 30 arm means
  of Table 1 of DellaVigna and Pope, "Stability of Experimental Results: Forecasts and Evidence",
  and two reference values of its Table 2.
- `2026-11-19_time_series_dml/401k-summary.csv`, `401k-decomposition.csv` and
  `2026-11-21_evaluation_methods/reproduction/401k-final-table.csv` quote published estimates
  and standard errors of Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins
  (2018) beside the estimates computed here.
- The summaries in `2026-11-22_acic_coverage/` quote true effects from the organizers of the
  ACIC 2019 Data Challenge: `by-design.csv` that of each of the 32 designs, and
  `reference-propensity.json` that of design 30.
- `2026-11-19_time_series_dml/forward/journals-2018.csv` lists the 215 articles of the 2018
  volumes of the *American Economic Review*, *AEJ: Applied Economics* and the *Quarterly Journal
  of Economics* (journal, volume, issue, pages, title, DOI), taken from the publishers' issue
  lists and checked against Crossref, beside this repository's own coding of each. Crossref
  treats bibliographic metadata as facts that are not subject to copyright; no abstract is
  included.

## Held: files that are not in the repository

These seven files exist in the working copy that produced the audits. Each reproduces a
provider's rows, and the terms read on 2026-11-26 do not grant redistribution.

| File | What it reproduces | Terms |
|---|---|---|
| `2026-11-19_time_series_dml/external/forward/gdp-survey.csv` | 117 rows of Survey of Professional Forecasters medians, release dates and first-release real GDP growth (Federal Reserve Bank of Philadelphia) | "may be used for informational, educational, and research purposes only"; no clause on redistribution |
| `2026-11-21_evaluation_methods/reproduction/fresh-stdout.json.gz`, `historical-stdout.json.gz` | The same 104 quarterly rows, and growth rates computed from ALFRED vintages, inside the printed output of 40 studies | As above; the FRED terms prohibit providing stored FRED content to third parties |
| `2026-11-22_acic_coverage/draws.csv`, `reference-propensity.csv`, `2026-11-21_evaluation_methods/additional/acic-results.csv` | The organizers' file name and true effect for each of 3,200 contest files, for the 50 evaluation files of design 30, and for 16 development files, beside the estimates computed here | No licence is stated for the datasets or the truth file |
| `2026-11-19_time_series_dml/prospective/cricket-fixture-examples.csv` | Six scheduled fixtures typed from an ICC and an ECB announcement, each with its link | The ICC reserves all rights and permits extracts for personal use; the ECB's terms were not found |

Consequences in the folders:

- `external/forward/gdp.py` reads `gdp-survey.csv` and does not run without it. With
  `--verify-survey` it compares the file with the provider's files; it does not rebuild it.
- `2026-11-22_acic_coverage/verify.py` and `reference_propensity.py` read `draws.csv`. `run.py`
  writes it again from the challenge files, and `reference_propensity.py` writes
  `reference-propensity.csv`.
- `reproduction/registry.py` and `supplement.py` read `historical-stdout.json.gz`; `registry.py`
  writes `fresh-stdout.json.gz`.
- No script reads `cricket-fixture-examples.csv`. `2026-11-19_time_series_dml/prospective/cricket.md`
  states in its own words which fixtures were announced, with links to the announcements.

A second group of files is held for a different reason, the pending redesign of `PanelDML`; they
are listed in [`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md).
