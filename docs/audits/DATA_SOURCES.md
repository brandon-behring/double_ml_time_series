# Data sources of the audit folders: licences, attribution, and what is held

The audit folders hold three kinds of data files:

1. **Numbers computed in this repository**: estimates, standard errors, counts, moments, digests
   and simulation draws. They reproduce no third-party data. Their source is cited in the note
   beside them.
2. **Third-party content under a licence that grants redistribution.** It is attributed below.
3. **Values quoted from a publication** for comparison with this repository's numbers (a paper's
   table, a benchmark's true effects by design). They are cited to the publication.

A file that reproduces a provider's own rows where the provider's terms do not grant
redistribution is **not in the repository**; those files are listed at the end.

Each provider's terms were read on its own site on 2026-11-26. The record per source (terms URL,
licence, what the files hold, verdict) is in
[`docs/research/dataset_ledger.yml`](../research/dataset_ledger.yml), under "Licence check of
2026-11-26". This note is not legal advice.

## Attribution

| Source | Licence | Files here | What was done to it |
|---|---|---|---|
| [Cricsheet](https://cricsheet.org/), Twenty20 international records (JSON) | [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1.0/) | `2026-11-19_time_series_dml/prospective/cricket-history.csv.gz` and the `cricket-*.json` forecasts | One row per match and focal team was derived from the match files; the forecasts are computed from those rows |
| UK Power Networks, [SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d) (Low Carbon London) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/field/london/` | `events.csv` lists the 48 tariff events with household counts and mean consumption computed here; the other files hold estimates and simulation draws. No meter reading is included |
| Gabriel Kreindler, [replication package for "Peak-Hour Road Congestion Pricing: Experimental Evidence and Equilibrium Implications"](https://zenodo.org/records/10637043) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/field/bangalore/` | Coefficients and monthly counts were computed from the package; archive member names and digests are listed |
| Casey, Glennerster, Miguel and Voors, [Sierra Leone community-development replication data](https://zenodo.org/records/7411565) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/rolling/external-controls.json` (`sierra_leone`) | Estimates computed here |
| ZOZO, Inc., [Open Bandit Dataset](https://huggingface.co/datasets/zozonext/open-bandit) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-19_time_series_dml/rolling/external-controls.json` (`retail`) | Estimates and three aggregate cells computed here |
| DellaVigna and Linos, nudge-units replication package (Econometric Society supplement to "RCTs to Scale: Comprehensive Evidence From Two Nudge Units", *Econometrica* 2022) | The [Econometric Society's permission](https://www.econometricsociety.org/publications/econometrica/permissions) for replication packages, after attribution | `2026-11-19_time_series_dml/rolling/external-controls.json` (`nudge_units`) | Estimates computed here |
| Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-21_public_data_extensions/reference/bike-*.json`; the bike entries of `2026-11-21_evaluation_methods/reproduction/standalone-stdout.json.gz` | Estimates computed here on the hourly histories |
| Hofmann and Siebenbrunner, [iFlex dynamic-price experiment](https://zenodo.org/records/8248802) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `2026-11-21_public_data_extensions/reference/iflex-results.txt`; the iFlex entries of the same archive | Estimates and arm counts computed here; one schedule row is printed in the archive |
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
  Analysis, Real Gross Domestic Product [GDPC1], retrieved from ALFRED, Federal Reserve Bank of
  St. Louis.

## Sources cited without a stated licence

- **Quarter-hour ride counts** (Blanchet, Glynn, Johari, Wu and Xing,
  [repository](https://github.com/wenqian-xing/conservative-inference-switchback-experiments)).
  `2026-11-21_public_data_extensions/reference/ride-switchback.json` holds estimates and design
  summaries computed here on the count series; no count is stored. The repository states no
  licence and asks for a citation.
- **The cooperation panel** of `2026-11-21_public_data_extensions/` uses synthetic outcomes and
  assignments. Of the supplement of Bojinov, Rambachan and Shephard (2021) only the digest of
  the archive is stored; its participant workbook asks for the original authors' permission
  for scientific use, and no participant value is in the repository.

## Values quoted from publications

- `2026-11-19_time_series_dml/rolling/external-controls.json` (`effort`) quotes the 30 arm means
  of Table 1 of DellaVigna and Pope, "Stability of Experimental Results: Forecasts and Evidence".
- `2026-11-19_time_series_dml/401k-summary.csv` and
  `2026-11-21_evaluation_methods/reproduction/401k-final-table.csv` quote the published estimates
  and standard errors of Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins
  (2018) beside the estimates computed here.
- The summaries in `2026-11-22_acic_coverage/` quote the true effect of each of the 32 designs
  from the organizers of the ACIC 2019 Data Challenge.

## Held: files that are not in the repository

These six files exist in the working copy that produced the audits. Each reproduces a provider's
rows, and the terms read on 2026-11-26 do not grant redistribution.

| File | What it reproduces | Terms |
|---|---|---|
| `2026-11-19_time_series_dml/external/forward/gdp-survey.csv` | 117 rows of Survey of Professional Forecasters medians, release dates and first-release real GDP growth (Federal Reserve Bank of Philadelphia) | "may be used for informational, educational, and research purposes only"; no clause on redistribution |
| `2026-11-21_evaluation_methods/reproduction/fresh-stdout.json.gz`, `historical-stdout.json.gz` | The same 104 quarterly rows, and growth rates computed from ALFRED vintages, inside the printed output of 40 studies | As above; the FRED terms prohibit providing stored FRED content to third parties |
| `2026-11-22_acic_coverage/draws.csv`, `2026-11-21_evaluation_methods/additional/acic-results.csv` | The organizers' design and true effect for each of 3,200 contest files and 16 development files, beside the estimates computed here | No licence is stated for the datasets or the truth file |
| `2026-11-19_time_series_dml/prospective/cricket-fixture-examples.csv` | Six scheduled fixtures typed from an ICC and an ECB announcement, each with its link | The ICC reserves all rights and permits extracts for personal use; the ECB's terms were not found |

Consequences in the folders:

- `external/forward/gdp.py` reads `gdp-survey.csv` and does not run without it. With
  `--verify-survey` it compares the file with the provider's files; it does not rebuild it.
- `2026-11-22_acic_coverage/verify.py` and `reference_propensity.py` read `draws.csv`. `run.py`
  writes it again from the challenge files.
- `reproduction/registry.py` and `supplement.py` read `historical-stdout.json.gz`; `registry.py`
  writes `fresh-stdout.json.gz`.
- No script reads `cricket-fixture-examples.csv`. The announced fixtures are also stated in
  `2026-11-19_time_series_dml/prospective/cricket.md`, with their links.

A second group of files is held for a different reason, the pending redesign of `PanelDML`; they
are listed in [`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md).
