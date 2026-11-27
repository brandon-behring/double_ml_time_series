# External replication benchmarks for time-series and panel DML

**November 20, 2026.** Ten public archives yielded executable checks of published estimates. The Italian bank-capital study tests a time-series DML estimator directly. Two randomized household trials exercise the repository’s residualization against independently published treatment contrasts. The remaining studies test fixed effects, seasonal controls, uncertainty, treatment timing, and the construction of instruments and outcome windows. Each replay checks the source bytes, analysis sample and numerical reference. [Dataset ledger](../../research/dataset_ledger.yml) · [Parallel runner](external/run.py)

The follow-up uses the same runner to revisit five substantive research questions. Select those analyses using `--suite questions`. [Questions, estimates and limitations](12-historical-causal-questions.md)

## Published time-series DML and a forward-only sensitivity

Ciganovic, D’Amario and Tancioni report the response to a bank-capital shock normalized to raise the Tier 1 ratio by 50 basis points. We rebuilt their 71-quarter analysis file byte-for-byte from the four deposited source-frequency files, reran their six-block reverse cross-fitting and regenerated Figure 2. All 45 estimates and 180 interval endpoints agreed with the deposited plot inputs within **1.15 × 10⁻¹³**. At two quarters, corporate lending fell 0.15857% (95% interval −0.31440% to −0.00275%). Some raw inputs in the deposit originate from subscription data providers. Reproducing the figure uses the deposited copies. [Paper, §5](https://arxiv.org/html/2603.10999v2#S5) · [Versioned archive](https://zenodo.org/records/21030557)

| Two-quarter real GDP response | Origins scored / available | Estimate | 95% interval |
|---|---:|---:|---:|
| Authors’ reverse cross-fitting | 54 / 54 | −0.14381% | [−0.24426%, −0.04336%] |
| Repository forward-only ridge, authors’ rows | 39 / 54 | −0.01787% | [−0.20655%, +0.17082%] |
| Repository forward-only ridge, contiguous calendar windows | 36 / 51 | −0.10276% | [−0.24305%, +0.03753%] |

The authors remove 2020Q1 through 2021Q4 before creating lags and leads. Their shift operation makes 2019Q4 and 2022Q1 adjacent. Requiring a complete calendar window for three lags of first differences and the two-quarter outcome removes 2019Q3, 2019Q4 and 2022Q1. The forward fits use the same shock scale and 105-column design, three expanding splits, a two-observation gap, fixed ridge adjustment and Bartlett HAC with three lags. Their out-of-fold outcome R² values were −1.81 and −2.54. The comparison changes training information, nuisance fitting and scored dates together. Both forward intervals include zero, and the outcome prediction diagnostics warrant caution about either point estimate. [Specification, environment and results](external/sequential/README.md)

## Randomized microeconomics benchmarks

| Trial and original specification | Replayed estimate | Check performed by the repository |
|---|---|---|
| **Ferraro and Price, Cobb County water conservation.** Table 2 model A, 106,669 households, three letter arms and a control, adjusted for two pre-treatment billing totals. | Peer comparison reduced summer use by **1.73868 thousand gallons**, HC1 SE **0.16863**. The independent-row homoskedastic SE was 0.17854. All three slopes and robust SEs match the printed table. | Native FWL reproduces the coefficient. A separate five-fold ridge DML fit on 83,319 peer and control households gives **−1.73685**, influence-function SE **0.16896**. The offline fixture tests conversion to gallons and changes in treatment units. [Paper](https://doi.org/10.1162/REST_a_00344) · [CC0 data and code](https://doi.org/10.7910/DVN1/22633) · [Specification](external/taxes_water/README.md) |
| **Gerber, Green and Larimer, Michigan voter turnout.** Table 3 model A, five arms, 344,084 voters in 180,002 randomized households. | The neighbors mailing raised turnout by **8.1310 percentage points**, household-clustered SE **0.3370 points**. Native FWL’s homoskedastic SE was **0.2601 points**. | The offline fixture retains household size, turnout count, arm and multiplicity. Assigning each voter a distinct cluster preserves the slope and understates uncertainty. A separate native ridge fit using six earlier-election turnout indicators gives **8.0745 points** (SE **0.3230**). All 229,444 voters in 119,999 neighbors-mailing or control households were scored once in five household-preserving folds. [Paper](https://isps.yale.edu/resource/social-pressure-and-voter-turnout-evidence-from-a-large-scale-field-experiment) · [CC0 archive](https://doi.org/10.60600/YU/CGMWNW) · [Author log](https://dataverse.yale.edu/api/access/datafile/42233) |

For turnout, the source is Yale Dataverse v2, file 42238, SHA-256 `05807bcbedaacf02ec6e6838e1b9c9b2fc99ad9d8c63f80c162991970044c6ce`. Treatment code 3 is the neighbors mailing and code 0 is no mail. The regression includes indicators for the other three arms. The offline calculation sums each household’s turnout residuals before squaring and applies the CR1 correction with 180,002 groups and five fitted parameters. In the native DML extension a single election-date effect supplies the intercept. Household fixed effects would absorb a household-level mailing assignment. Both trial fixtures contain aggregate statistics. Full replays reconstruct them from the public files. [Turnout implementation](external/turnout/replay.py) · [Water fixture provenance](external/taxes_water/README.md)

## Seven additional published results

| Study and benchmark | Measured result | What can invalidate the comparison |
|---|---|---|
| **Wolfers, state divorce reform**, Table 2. Population-weighted state and year fixed effects on 1,631 state-years. | In the first two reform years, **+0.2667** (SE 0.0849) divorces per 1,000. At 15 years or more, **−0.5074** (0.0805). State-specific quadratic trends move the late estimate to **+0.2510**. All 24 coefficients and SEs matched. | Weights must apply to the intercept and every regressor. Native FWL independently reproduced the early and late effects in all three specifications. [Table 2](https://users.nber.org/~jwolfers/Papers/Divorce%28AER%29.pdf) · [Archive and replay details](external/panels/README.md) |
| **Chetty, Looney and Kroft, beer taxes**, Table 6 column 1. First differences and year effects, 1,607 state-years. | Excise and sales slopes **−0.8768** and **−0.1992**. Conventional SEs **0.1691** and **0.3006** match the printed 0.17 and 0.30. | The text specifies state clustering. The replication code uses conventional SEs. The explicit state-clustered calculation gives **0.1321** and **0.2935**. Both calculations are checked. The retailer withheld the separate grocery microdata. [Paper](https://www.korykroft.com/files/Chetty_Kroft_Looney_AER_2009.pdf) · [Archive and definition of the SEs](external/taxes_water/README.md) |
| **Dahl and DellaVigna, movie attendance and assaults**, Tables II–III. Weekend-clustered OLS and instrumental variables. | The strongly violent movie audience slope changes from **+0.0324** (SE 0.0053) to **−0.0091** (0.0026) when full controls enter. Thirty coefficients and SEs matched, including all twelve time-of-day IV estimates. | Omitting weather, holiday and calendar controls can reverse the sign. The historical IV design uses attendance the following weekend. A prospective predictor would lack that information. [Published tables](https://eml.berkeley.edu/~webfac/dellavigna/e101a_f09/moviescrimeQJEProofs2009.pdf) · [Archive and specifications](external/panels/README.md) |
| **Drink Less, randomized notifications**, Tables 2 and 6. 349 participants × 30 evenings. | One-hour app-opening risk ratio **3.5233** [2.9175, 4.2548]. The 24-hour ratio was **1.2597** [1.1867, 1.3371]. | The file stores the probability of the observed arm. Applying the design treatment probability of 0.6 throughout changes the one-hour ratio to **3.5243**. A separate augmented inverse-probability fit with five participant-disjoint folds gives an absolute one-hour increase of **9.10 percentage points** [7.47, 10.73]. [Paper](https://mhealth.jmir.org/2023/1/e38342/) · [Dataset A](https://osf.io/download/mtcfa/) · [Analysis](external/sequential/README.md) |
| **myBPmyLife, randomized activity prompts**. 287 people, 187,517 randomized decisions, 153,983 available complete occasions. | Next-hour step-count ratio **1.0125** [0.9829, 1.0421], (p=0.403). This matches the paper at its reported precision. | Including complete occasions without prior heart-rate availability changes the analyzed population and gives **1.0224** [0.9851, 1.0597]. The paper uses a transformed step outcome and participant-level inference. [Paper](https://doi.org/10.2196/78218) · [Pinned sources and analysis](external/sequential/README.md) |
| **Lopez Bernal, Cummins and Gasparrini, Sicily smoking ban**, Figure 4. 48,914 acute coronary events over 59 consecutive months. | The quasi-Poisson rate ratio was **0.88507** [0.83940, 0.93323]. The fit uses the standardized population offset, linear time trend and two annual Fourier pairs. | The policy begins January 2005, after 36 pre-policy months. With four expanding folds, the second test block contains the first treated month while its training block contains zero treated months. The script reports treatment support per fold. [Figure 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC5407170/) · [Corrected data and code](https://github.com/gasparrini/2017_lopezbernal_IJE_codedata/tree/85456263850d03ae39439e032929347c40ff3a26) |
| **Ramey and Zubairy, US fiscal multipliers**, Table 1. Quarterly spending instrumented by military news, with four lags of news, output and spending. | The raw-data two- and four-year multipliers were **0.66371** and **0.71336**, matching the published **0.66** and **0.71**. Extending the two-year window from eight to nine quarters gives **0.66896**, which rounds to 0.67. | The supplied results workbook gives **0.66191** and **0.71194** with SEs 0.06822 and 0.04548. A follow-up reproduced the published SEs with automatic Bartlett bandwidth 29. The raw-input estimates of **0.067114** and **0.043575** round to the published 0.067 and 0.044. The source of the output workbook’s different coefficients and SEs remains unidentified. Both archived raw input formats give the same refit to numerical precision. [Follow-up with bandwidth and sample checks](classical/README.md#historical-fiscal-multipliers) · [Published Table 1](https://econweb.ucsd.edu/~vramey/research/Ramey_Zubairy_JPE_published.pdf) · [Author archive](https://econweb.ucsd.edu/~vramey/research/Ramey_Zubairy_replication_codes.zip) |

The Sicily replay reads the corrected author CSV, SHA-256 `63a867b3fc8748425ff27a28202c19b4c2741e69a1ab753ad8e77c4e56457dda`. The fiscal replay reads the author ZIP, SHA-256 `e268f8d4abc997227401d5b5f218d0cfa3c527cb9091f6e715baa5a7a6391d07`. It reads the authors’ full-sample raw CSV from 1889Q1 through 2015Q4, normalizes by potential output and the GDP deflator, and includes the current quarter in each cumulative horizon. Both samples start in 1891Q1. Lagging news and forming complete forward sums leaves 493 origins through 2014Q1 and 485 through 2012Q1, respectively. The workbook comparison reads the contemporaneous archive’s `Slack-newsy` sheet. [Replay and numerical checks](external/time_series/replay.py)

## Reproduce the checks

From the checkout root, install the locked project and development dependencies. The default tests read small local fixtures. The full replays download the checksum-pinned studies into a user cache. The Italian raw-data builder uses the supplied X-13 binary and a separate Python 3.12 environment. The Drink Less replay uses base R to read the original RDS. Set up the Italian environment and export `DML_TS_ITALY_PYTHON` before running all ten. [Environment setup](external/sequential/README.md#run-from-the-repository-root)

```bash
uv sync --locked --extra dev
audit=docs/audits/2026-11-19_time_series_dml
export DML_TS_BENCHMARK_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/dml-ts/external"

# Offline fixture and falsification checks
uv run --locked --extra dev python -m pytest -q -m 'not network' test/integration/test_external_*.py

# Full published-data checks, including the native DML sensitivities
uv run --locked python "$audit/external/run.py" --jobs 4

# Run selected studies using the same cache
uv run --locked python "$audit/external/run.py" water turnout beer

# Regenerate the CC0 fixtures to independent output paths
uv run --locked python "$audit/external/turnout/replay.py" --cache-dir "$DML_TS_BENCHMARK_CACHE" --write-fixture /tmp/turnout-counts.csv
uv run --locked python "$audit/external/taxes_water/replay.py" water --cache-dir "$DML_TS_BENCHMARK_CACHE" --write-fixture /tmp/water-moments.json
```

The runner uses up to four independent study processes by default and gives each process one numerical-library thread. A failed study returns a nonzero overall exit code after the other studies finish. Each successful raw replay checks its own study outcomes. Full-precision author artifacts allow tight tolerances, while rounded journal tables are checked at the printed precision. Native DML sensitivities report their own samples, intervals and diagnostics. Reproducing an observational estimate establishes numerical agreement under the specified analysis. Interval coverage and causal identification require additional evidence. [Falsification and coverage tests](10-property-tests-and-falsification.md)

### Verification on November 20

The four-worker runner passed all ten studies. With source replays enabled, all 17 external pytest cases passed. The default run passed seven offline cases and skipped the ten source replays. A broader offline check of the integration, estimator-property and scientific-reference tests passed 42 cases. Both CC0 fixtures together occupy 3,332 bytes. Every new ledger ID maps to an executable study. In the ledger, license verification and a successful numerical replay are recorded separately through the existing status field and this report.

```bash
DML_TS_EXTERNAL_REPLAY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked --extra dev python -m pytest -q test/integration/test_external_*.py
uv run --locked --extra dev python -m pytest -q -m 'not network' \
  test/integration test/dml/test_estimator_properties.py test/dml/test_scientific_oracles.py
```
