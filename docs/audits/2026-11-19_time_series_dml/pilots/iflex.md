# Norwegian iFlex · household demand during electricity price events

**In Trondheim, households assigned to the incentive program used 0.29 kWh less across the 13 peak-price hours of the first event than concurrent controls.** The 95% interval runs from a reduction of 3.91 kWh to an increase of 3.32 kWh. Across 21 event days, the baseline-adjusted household contrast was −1.08 kWh [−2.22, 0.06]. The repository’s temporal estimator detected a negative association in the daily difference between the two arms. These quantities answer different questions and have different sources of uncertainty.

## Population, treatment and timing

The pilot uses new phase-two participants from a single region and electricity supplier. The authors describe randomized household assignment to an incentive group and a concurrent control group. Treatment included a day-ahead message, a reward for changing consumption relative to a calculated baseline and information on saving energy. The ordinary retail tariff continued. Organizers chose dates with cold weather forecasts and could override random price-profile draws to balance the schedule. The first event, December 16, 2020, used the daytime B_10 profile. The analysis sums readings over the flat high-price portion, hour labels 8 through 20 inclusive. The baseline covers the same hours December 1–14, ending before the first message at 18:00 on December 15. [Design, eligibility and data dictionary](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539631/) · [Event calendar](https://zenodo.org/records/8248802/files/experiment_days.csv?download=1) · [Hourly profiles](https://zenodo.org/records/8248802/files/price_signals.csv?download=1)

The versioned R workspace contains 13,515,600 hourly readings. Counting distinct IDs within phase gave 742 in phase one and 4,375 in phase two. The second count matches the release’s roster and the paper’s Table 14. Table 1 prints 3,741 in the phase-two total cell although its regional subtotals sum to 4,375. The subset comprises 201 `Trond_1` and 204 `Trond_Control` households with status `Ok`, each observed for all 24 hours on all 116 days, December 1, 2020 through March 26, 2021. We checked 1,127,520 unique household/date/hour records. Only the treatment arm received signals. Other regions include phase-one returnees and additional assignment groups. [Version 2 data](https://zenodo.org/records/8248802) · [Participant roster](https://zenodo.org/records/8248802/files/participants.csv?download=1)

## Household assignment benchmark

Each row below contributes one outcome per household. For all-event rows, we average each household’s daily peak-period kWh across the 21 realized event dates. For baseline-adjusted rows, we subtract its own pre-announcement mean. The contrast is incentive minus control. Welch intervals measure uncertainty across the households observed in the two arms for this fixed set of days.

| Household outcome (kWh per 13-hour period) | Incentive mean | Control mean | Difference | 95% interval |
|---|---:|---:|---:|---:|
| Pre-announcement baseline | 28.85 | 28.51 | +0.34 | [−3.45, 4.12] |
| First event | 26.07 | 26.36 | −0.29 | [−3.91, 3.32] |
| First event minus baseline | −2.77 | −2.15 | −0.63 | [−1.70, 0.44] |
| Average event day | 33.33 | 34.07 | −0.74 | [−5.24, 3.76] |
| Average event day minus baseline | 4.48 | 5.56 | −1.08 | [−2.22, 0.06] |

These estimates describe retained experimental households on observed event dates. The source excludes some consenting households after recruitment for incomplete meter histories, contract changes and consumption outliers. Missing outcomes for excluded households limit a full randomized-cohort intention-to-treat interpretation. The first event avoids carryover from prior signals. The 21-event average can reflect accumulated learning and anticipation. Transferring this result to customers receiving routine retail price changes would require assumptions about the experiment’s information and reward components. [Study methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539631/)

## Temporal PLR on one ordered series

For each date, we computed the incentive arm’s mean 13-hour consumption minus the control arm’s mean. The outcome is therefore kWh per household. Treatment is an indicator of a scheduled event that day. The regression conditions on the previous event indicator, scaled day index, weekly sine and cosine, and regional mean temperature from two days earlier. That full-day temperature was observable before the previous evening’s message. Both nuisance learners are ridge regressions. Four expanding folds drop 25 warm-up rows after the two weather-lag rows and one treatment-lag row. The model scores 88 days from December 29 through March 26, including 19 events. [Repository temporal estimator](../../../../dml_ts/dml/temporal_plr_dml.py) · [Pilot runner](iflex_temporal.py)

| Training gap (days) | HAC bandwidth (days) | Event coefficient | HAC SE | Reported 95% interval |
|---:|---:|---:|---:|---:|
| 2 | 7 | −0.571 | 0.200 | [−0.963, −0.179] |
| 2 | 14 | −0.571 | 0.192 | [−0.947, −0.195] |
| 7 | 7 | −0.599 | 0.199 | [−0.989, −0.209] |

The default run warned about poor nuisance prediction. Out-of-fold R² was −0.543 for the arm difference and 0.028 for the event indicator. Price-event timing depends on demand forecasts, and the cohorts may respond differently to temperature or earlier signals. The event coefficient describes the conditional change in the between-arm demand difference on selected dates relative to other scored dates. Its HAC interval treats calendar days as observations. The randomized-arm benchmark compares households within the same event dates and its intervals use between-household variation. Agreement in direction is an empirical comparison. This run provides no direct coverage check or hourly price elasticity.

## Reproduce

Run from the repository root. The download is the 40.3 MB R workspace, with Zenodo MD5 `7547c0931b0cda997609d9e7df70fdd2` and locally verified SHA-256 `0ed8a76365f6379cf613c12069180d10db0ef6f56de0e73268e2de28f62908a2`. The R script checks the archive and writes a 116-row aggregate into the temporary directory. The audit used base R 4.5.2, Python 3.13.17, temporalcv 2.2.0, scikit-learn 1.6.1, NumPy 2.4.6 and pandas 3.0.3. [Data preparation and household estimates](iflex.R) · [Zenodo file manifest and checksums](https://zenodo.org/records/8248802)

```bash
curl -fL 'https://zenodo.org/records/8248802/files/iFlex%20dataset.RData' -o /tmp/iflex_audit.rdata
sha256sum /tmp/iflex_audit.rdata
Rscript docs/audits/2026-11-19_time_series_dml/pilots/iflex.R /tmp/iflex_audit.rdata /tmp/iflex_audit_daily.csv
PYTHONPATH=. uv run --frozen python docs/audits/2026-11-19_time_series_dml/pilots/iflex_temporal.py /tmp/iflex_audit_daily.csv
```
