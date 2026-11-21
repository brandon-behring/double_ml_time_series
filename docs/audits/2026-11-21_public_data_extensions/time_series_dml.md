# A public-data test of time-series double machine learning

Washington bike rentals and January ride counts supply observed demand histories for interventions with known simulated effects. Norway’s iFlex trial supplies an observed household offer and observed outcomes. A synthetic rematching experiment isolates how shared treatment affects inference. An extension on regional electricity demand would require an observational identification strategy. [Paired-assignment audit](cooperation_panel.md) · [Inputs, software and results](README.md)

## First test on actual hourly travel demand

The Washington, DC Capital Bikeshare file `hour.csv` contains 17,379 distinct local date/hour labels between 1 January 2011 and 31 December 2012. There are 17,544 possible naive clock labels in that span and 165 absent labels. The four weather categories have 11,413, 4,544, 1,419 and 3 observations. The final category alone cannot sustain an hour-by-hour treatment comparison. The file includes total rentals and their two components, casual and registered rentals. Including either same-hour component as an explanatory variable leaks outcome information. The archive SHA-256 starts with `b70182d0d050`. [Dataset and variable definitions](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) · [Operator trip data](https://capitalbikeshare.com/system-data)

Start with a **known-effect benchmark**. Retain the actual hourly counts and calendar. Generate a binary synthetic offer at each available hour with known probability determined by hour and working day. The implemented probability ranges from 0.25 to 0.65. Add 10, 5 and 2 rentals to the outcome at elapsed hours zero, one and two after each generated offer, summing overlapping offers. Choose the assignment independently across hours conditional on the calendar. This makes the mean effects 10, 5, 2 and then zero by construction. Only add a contribution when both endpoints exist. Use calendar timestamps for lookups, explicitly handle repeated or missing local hours across daylight-saving changes, and report the number of usable pairs by horizon. This is a test of recovery in realistic demand noise. It yields no estimate of a historical offer.

1. Evaluate an oracle using known assignment probabilities and a simple calendar regression. Then estimate treatment propensity and outcome regressions on whole training blocks. Fit transformations and choose hyperparameters inside each training block. Keep an untouched last season as a forward test.
2. Keep outcome windows out of neighboring training folds. Vary the extra gap across a day, a week and four weeks to probe remaining serial dependence. Record overlap by season/hour, calibration, score autocorrelation and the sample count at each horizon. A random-row split is a diagnostic for leakage.
3. Across independent assignment seeds report bias, interval coverage, runtime and failure rate at the four known effects. Use a dependence-aware variance estimate or a block bootstrap on the ordered score and examine sensitivity to bandwidth or block length. Separate Monte Carlo error in coverage from the width of each interval.
4. An observational extension could examine rentals following a change to light precipitation. Weather forecasts, commuting decisions, supply, rebalancing and serially correlated weather complicate the interpretation. Define whether later weather is allowed to evolve naturally and which covariates were known before exposure. The original field records broad conditions, so a weather-category association alone cannot identify a rain shock. [UCI variables and provenance](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

## What actually happened in the bike benchmark

The runnable pilot scores 17,377 unambiguous UTC hours before splitting. It uses five expanding-window test blocks beginning in July 2011, a seven-day gap, and separate regressions for each horizon. The observation windows contain 13,127, 13,101, 13,096 and 13,084 scored treatment hours at horizons 0, 1, 2 and 6. Each of the 200 assignments uses the same observed counts and a fresh randomized offer series. The estimator divides the cross-fitted residual-treatment/residual-outcome covariance by the residual-treatment variance. The nominal 95% intervals use a Bartlett heteroskedasticity and autocorrelation consistent (HAC) estimate with 48 elapsed-hour lags. Coverage is empirical over these 200 draws. Near 95% its Monte Carlo standard error is about 1.5 percentage points. [Executable benchmark](checks/bike_dml.py)

| Treatment-probability model | Mean effect at 0, 1, 2, 6 hours (truth 10, 5, 2, 0) | 95% interval coverage in the same order |
|---|---|---|
| Known assignment probability | 9.96, 4.89, 1.90, −0.24 | 96.0%, 95.5%, 95.0%, 96.0% |
| Earlier assignments within each hour/workday group | 10.07, 5.00, 2.00, −0.12 | 95.0%, 94.5%, 93.0%, 94.0% |
| Ridge fit with additional month and time-trend terms | 25.25, 20.42, 18.18, 16.29 | 12.5%, 11.5%, 11.0%, 8.0% |

All three use the same outcome model. The large errors in the expanded treatment model occur when it extrapolates to calendar regimes poorly represented in the short early training windows. Fitting that design to the **true** assignment probabilities in the first two training blocks still produces mean absolute prediction errors of 0.120 and 0.105 on their corresponding test blocks. An evaluation restricted to September–December 2012 gives a very different picture. The expanded model returns 9.98, 5.01, 2.34 and 0.19 in that final block. Reporting only that block would hide the larger failures earlier in the record. Conversely, a one-day or four-week gap changes the simple hour/workday model’s mean at any horizon by at most 0.05 rental relative to the weekly gap. This is evidence on a controlled partially linear benchmark. It does not establish an effect of actual bikeshare promotions.

Run these commands from the repository root after preparing the input cache and Python environment. Write fresh output to a separate directory so comparisons preserve the reference. [Setup and source downloads](README.md#reproduce-the-calculations) · [Weekly gap](reference/bike-168.json) · [One-day gap](reference/bike-24.json) · [Four-week gap](reference/bike-672.json) · [Final block](reference/bike-final-block.json)

```bash
audit=docs/audits/2026-11-21_public_data_extensions
cache=output/time-series-dml
replay=$(mktemp -d)
for gap in 24 168 672; do
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -B "$audit/checks/bike_dml.py" \
        "$cache/bike.zip" --repetitions 200 --gap "$gap" > "$replay/bike-$gap.json"
done
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B "$audit/checks/bike_dml.py" \
    "$cache/bike.zip" --repetitions 200 --last-block > "$replay/bike-final-block.json"
```

To isolate extrapolation from sampling noise, substitute the true probabilities for the fitted treatment labels in the first two horizon-zero training sets. Their held-out mean absolute errors are 0.120273 and 0.105380.

```bash
PYTHONPATH=docs/audits/2026-11-21_public_data_extensions/checks OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python -B - <<'PYTHON'
from pathlib import Path
import numpy as np
from bike_dml import data, fit_blocks, shifted_indices

hours, dates, _, features, propensity = data(Path("output/time-series-dml/bike.zip"))
folds = fit_blocks(features, dates, hours, {0: shifted_indices(hours, 0)}, 168)[0]
for train, test, solve in folds[:2]:
    forecast = np.clip(features[test] @ (solve @ propensity[train]), 0.05, 0.95)
    print(f"{np.mean(np.abs(forecast - propensity[test])):.6f}")
PYTHON
```

## Observed microeconomic intervention with hourly outcomes

The Norwegian iFlex experiment supplies actual household assignment, prices and electricity use. Its Phase 2 Oslo group `Os_1` experienced 20 dates with the same price profile and four peak-price levels. We compared its 255 households with the 257 in `Os_Control`. All 512 had complete published peak-hour observations on the 20 chosen dates. The source RData file contains 13,515,600 household-hour rows overall. The price message was sent the day before, test dates targeted forecast cold days, and the project sometimes altered the signal sequence to balance exposures. The household contrast measures assignment to the whole offer, including information and rewards, in the released Oslo sample. Interpreting it causally requires that selection into that sample preserve the arm comparison. A causal one-hour price elasticity would require additional design assumptions. [Original data release, version 8248802](https://zenodo.org/records/8248802) · [Authors’ experimental design and data dictionary](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539631/)

Across all Phase 2 regions the authors report 702 exclusions among 5,077 recruits, including 577 with missing readings, 86 withdrawals and 30 consumption outliers. In the 7,410-row participant file, the only rows labeled `Os_1` or `Os_Control` are the 255 and 257 retained households, all coded `OK`. The group table reports these same counts. The recruitment table gives losses by supplier, with no original assignment-by-arm denominators for this Oslo contrast. Complete readings for the 512 retained households therefore cannot establish the offer effect among everyone recruited. The intervals below condition on the released households and the chosen dates. [Design paper, Tables 14–15](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539631/)

To examine selection, recover the number assigned and the number observed in each Oslo arm. If assignment could only change retention in one direction for all households, trimming the arm with higher retention gives Lee bounds for the always-observed households. Supplier-wide losses alone cannot determine that trim. [Lee, sample selection in randomized experiments](https://doi.org/10.1111/j.1467-937X.2009.00536.x)

For an independent test, five folds held out whole households. For each arm the outcome model learned from other households on the first ten event days. It used peak and nighttime consumption from December 1–14, ending before the first notification. The last ten event dates, February 1 through March 22, provided the outcomes. The augmented estimator standardized the two predictions to the same population, then used each group's held-out mean residual. Each household contributes once, so the reported standard errors treat the 512 households as the units of variation conditional on the observed dates. These are trial-specific uncertainty estimates. They do not address generalization to other winters. [Runnable base-R analysis](checks/iflex.R)

| Published hourly window on final ten event dates | Control mean kWh/hour | Raw price-group minus control | Cross-fitted adjusted difference | Approximate 95% interval |
|---|---:|---:|---:|---:|
| Peak, hour labels 8–20 | 2.892 | −0.144 | −0.042 | [−0.139, 0.054] |
| Late, hour labels 22–24 | 2.886 | −0.181 | −0.097 | [−0.208, 0.014] |

The raw peak-hour difference before the intervention was already −0.081 kWh/hour. Baseline adjustment moves both event-window estimates closer to zero, and each interval contains zero. This small two-arm check supplies an experimental reference for testing dependence-aware methods. The outcomes are averages over the prespecified ten dates. Dynamic effects by exposure level, event order and rebound would require more experimental groups, explicit price-schedule contrasts, and uncertainty at the appropriate assignment levels. Ito, Ida and Tanaka's independently randomized Japanese moral-suasion and dynamic-pricing experiment offers a second design for studying habituation. [Japanese field experiment and replication package](https://www.aeaweb.org/articles?id=10.1257/pol.20160093)

A design-based comparison uses the same 512 households and outcomes on the last ten event dates. It fits OLS with assignment interacted with each centered baseline predictor and uses household-level HC2 standard errors. The peak estimate is **−0.055 kWh/hour** with interval [−0.139, 0.029]. The late estimate is **−0.105** [−0.211, 0.001]. The intervals include zero for both methods. On this sample the simple adjustment actually has slightly smaller standard errors (0.043 and 0.054) than the earlier-period cross-fit (0.049 and 0.057). This makes precision gain an empirical question for future DML extensions. Lin gives the randomized-experiment rationale for full interactions and robust inference. Wager and coauthors develop cross-estimation for randomized assignment. Our additional date holdout tests transfer across event days in this experiment. [Lin’s regression-adjustment paper](https://www.stat.berkeley.edu/~winston/agnostic.pdf) · [Wager, Du, Taylor and Tibshirani](https://pmc.ncbi.nlm.nih.gov/articles/PMC5111650/)

The script checks the release MD5 (`7547c0931b0cda997609d9e7df70fdd2`), the 512 unique households, the 20 unique dates and the household–date–hour key. The output gives both retained arm sizes and household-level estimates. [Household comparison output](reference/iflex-results.txt) · [Data download](README.md#reproduce-the-calculations)

```bash
audit=docs/audits/2026-11-21_public_data_extensions
Rscript --vanilla "$audit/checks/iflex.R" output/time-series-dml/iflex.RData
```

## Observational extension with aggregate electricity demand

EIA publishes hourly balancing-authority demand, forecast demand and interchange. NOAA supplies station-level hourly temperature and other weather observations. An initial estimand is the response of regional demand at horizons 0, 1, 6 and 24 hours to an unusually hot hour, defined against a pre-specified seasonal baseline. Start with one balancing authority, multiple years and a pre-specified station or population-weighted station set. Align hour-ending demand and weather observations in UTC, log revisions and missingness, and reserve whole summers as holdouts. Changes in service territory, station representativeness and the choice of weather baseline can all change the estimate. In particular, observed temperature minus a seasonal mean and an unexpected forecast error are different exposures. A forecast-error claim requires forecasts as issued at the time. This is a proposed application. The EIA and NOAA sources have been located, but the series have yet to be joined. [EIA definitions and downloads](https://www.eia.gov/electricity/930-content/about.html) · [NOAA hourly observations](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)

## When randomization happens in time blocks

Blanchet and colleagues release 2,976 consecutive quarter-hour ride counts from January 2022. The CSV has only timestamp and demand. They use the counts to calibrate a ridesharing simulation. In our benchmark the background demand is observed, and both the schedule and its effect are generated. This lets us vary how long a treatment remains in force while retaining realistic demand swings. [Pinned author data](https://raw.githubusercontent.com/wenqian-xing/conservative-inference-switchback-experiments/e29f6f09716e0df2a4edb8012c09a31c627338f5/lyft_ride_counts_15min.csv) · [Blanchet et al., 2026](https://proceedings.mlr.press/v300/blanchet26a.html)

Train a fixed calendar prediction on the first seven days by averaging demand in each quarter of the day separately for weekdays and weekends. On the remaining 24 days, independently assign each block treatment with probability one half. Start the treatment response at zero. In each quarter it moves a fraction $1-2^{-1/4}$ of the way toward 100 extra rides if treated or zero if untreated. The long-run contrast is 100 rides per quarter. Averaged over positions in a newly assigned block of length $b$, the contemporaneous assignment contrast is $100 b^{-1}\sum_{j=1}^{b}(1-2^{-j/4})$. It averages over previous randomized blocks. The first block has the same contrast because its starting state cancels between assignments.

| Block length | Assignments in 24 days | Known current-block contrast | Adjusted estimate, mean over 2,000 draws | Across-draw SD, raw → adjusted |
|---|---:|---:|---:|---:|
| 1 hour | 576 | 33.93 | 34.03 | 126.61 → 36.72 |
| 6 hours | 96 | 78.32 | 78.13 | 303.04 → 86.48 |

The raw score is the inverse-probability weighted treatment contrast. The adjusted score subtracts the same pre-trained prediction from both arms before weighting. Its Monte Carlo standard errors for the two reported means are 0.82 and 1.93. Enumerating all eight assignments in a separate three-block example recovers both analytical contrasts. Calendar adjustment reduces variation. Lengthening the block changes the scientific target and reduces the number of random assignments. For a time-series DML comparison, give every learner the same past-only training data, keep the whole assignment block together in inference and score against the appropriate analytical target. These are simulated effects on a fixed demand history. [Runnable calculation](checks/ride_switchback.py) · [Simulation results](reference/ride-switchback.json)

```bash
audit=docs/audits/2026-11-21_public_data_extensions
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B "$audit/checks/ride_switchback.py" \
    output/time-series-dml/lyft_ride_counts_15min.csv
```

## Literature that changes the study design

- **Bojinov and Shephard** formulate lagged treatment effects for a single randomized time series and use the assignment mechanism to test sharp nulls. **Bojinov, Rambachan and Shephard** extend the idea to repeated assignments in panels. Their economics example uses 110 participants, rematched in pairs over 20 rounds in five sessions. Stakes are assigned to each pair. The paper describes the outcome as first-stage cooperation. This offers a repeated, real treatment for studying learning across rounds. A comparison should preserve paired assignments and rematching, with earlier rounds providing the available history. The Econometric Society mirror supplied the data-and-code ZIP on November 21. Its treatment cutoff, action label and stage selection need reconciliation with the two papers before an empirical DML fit. The released estimator now has an exact synthetic rematching test. [Code and design comparison](cooperation_panel.md) [Time-series experiment](https://doi.org/10.1080/01621459.2018.1527225) · [Panel study and supplement](https://onlinelibrary.wiley.com/doi/full/10.3982/QE1744) · [Original experiment](https://econweb.ucsd.edu/~jandreon/Publications/JET2006.pdf)

- **Ballinari and Wehrli** study impulse responses to repeated discrete treatments in a dependent series and develop inference for horizon-specific effects. Their setting is the closest comparison for the synthetic offer. Use a score for the chosen treatment contrast and state the sequential assignment and dependence assumptions. [Authors’ paper](https://arxiv.org/abs/2411.10009)
- **Lewis and Syrgkanis** develop an orthogonal version of dynamic g-estimation. When earlier interventions change later states or policy choices, effects of a treatment sequence require a different target and nuisance regressions. Their single-unit results impose additional stationarity conditions. [Authors’ paper](https://arxiv.org/abs/2002.07285)
- **Ciganovic, D’Amario and Tancioni** develop reverse cross-fitting for stationary, time-reversible series and show how nuisance tuning can affect causal bias. Test their splitting strategy only with explicit justification for reversibility. Calendar seasonality, growth in the bike network and policy shifts make a forward holdout informative. [Authors’ paper](https://arxiv.org/abs/2603.10999)

The cooperation example now has an exact size check under a fixed sharp null. At a nominal 5%, the two pair-based tests reject 4.57% and 4.27% of the time. Using an independent coin for each person raises those rates to 17.72% and 17.78%. [Assignment weights, lags and inference](cooperation_panel.md#does-inference-respect-the-shared-pair-assignment)
