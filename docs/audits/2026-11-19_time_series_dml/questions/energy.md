# Did Trondheim households shift electricity use to other hours?

**The baseline-adjusted estimates favor a reduction in total use that lasted through the following day.** On seven pricing events with a clear follow-up day, households assigned to the incentive program used an estimated **1.93 kWh less on the event day and 2.08 kWh less the next day**, after subtracting their own pre-announcement consumption. Both peak and off-peak consumption contributed to the event-day reduction. The unadjusted comparison between the assigned arms was much less precise and allowed a rebound.

## The same households on the same dates

Every estimate below compares 201 households assigned to the program with 204 concurrent control households. Negative values indicate lower consumption in the program arm. Each household contributes one mean over the same seven dates to each comparison. The interval next to each estimate is a pointwise 95% interval. Peak denotes hour labels 8–20 inclusive and off-peak the other 11 hours. The follow-up day had no price event.

| kWh per household per event window | Unadjusted arm difference | Difference in change from baseline | Cross-fitted ridge adjustment |
| --- | ---: | ---: | ---: |
| Event day, 13 peak-price hours | −0.66 [−4.79, 3.47] | −1.00 [−1.94, −0.06] | −1.01 [−1.91, −0.11] |
| Event day, 11 off-peak hours | +0.03 [−3.36, 3.42] | −0.93 [−1.78, −0.08] | −0.93 [−1.63, −0.23] |
| Event day, all 24 hours | −0.63 [−8.08, 6.83] | −1.93 [−3.58, −0.28] | −1.94 [−3.41, −0.47] |
| Next day alone, all 24 hours | −0.78 [−8.05, 6.48] | −2.08 [−3.65, −0.52] | −2.15 [−3.63, −0.67] |
| Event day plus next day, all 48 hours | −1.41 [−16.12, 13.30] | −4.01 [−7.16, −0.87] | −4.09 [−6.97, −1.22] |

The next-day estimate uses next-day household outcomes directly. The 48-hour interval uses each household’s two-day sum, preserving dependence between the days. The command checks that peak and off-peak estimates sum to total consumption and that the isolated event-day and next-day point estimates sum to the matched 48-hour result.

## Which treatment the data identify

Recruited households were randomly assigned within the study. Program households continued to pay their ordinary retail tariff and received a next-day signal at 6 pm, an incentive to adjust load and conservation information. Organizers selected cold forecast dates and could override a randomly drawn price profile to balance its frequency. The arm difference therefore estimates assignment to this program among the retained Trondheim households on the chosen dates. Earlier messages and rewards could affect both the event and follow-up days. The design cannot isolate the carryover of a single day's message or extrapolate to an ordinary tariff increase. The analysis conditions on households retained after the source's meter-data and participation exclusions. [Study design and reward rules, sections 4.4–4.7](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539631/) · [Dataset and codebook](https://zenodo.org/records/8248802)

The release contains 1,127,520 unique meter readings for these 405 households over 116 complete days, from December 1, 2020, to March 26, 2021. We verified that household signal records agree with the 21 dated events and that controls received no price signals. For an isolated event on day *d*, we require observations from *d*−1 through *d*+2, with no other event on *d*−1, *d*+1 or *d*+2. An event on *d*+2 would generate a new message on the evening of the measured follow-up day. Seven starts qualify in 2021 — January 18 and 29, February 11, and March 5, 8, 19 and 24. The two-day window starts at midnight on the event day. Shifting consumption to the preceding evening after the message or beyond the following day remains possible. [Event schedule](https://zenodo.org/records/8248802/files/experiment_days.csv?download=1) · [Hourly signal profiles](https://zenodo.org/records/8248802/files/price_signals.csv?download=1)

For the baseline comparison we subtract each household’s average matching-hour use during December 1–14, ending before the first announcement on December 15. For a 48-hour outcome we subtract two days of baseline usage. Interpreting both adjustments as precision gains for the arm contrast assumes that learning one's assignment did not already affect baseline consumption. Welch intervals use the household-level arm variances. The repository’s partially linear DML uses the same households and outcomes with the two pre-announcement covariates (peak and off-peak baseline use). It fits ridge nuisance regressions in five shuffled folds of households, seed 42. Its interval uses household influence scores. These intervals quantify uncertainty across households for the realized dates. They do not estimate how much results would vary across other winters or event schedules.

## Sensitivity to the schedule and baseline

| Total kWh per household | Unadjusted arm difference | Difference in change from baseline | Cross-fitted ridge adjustment |
| --- | ---: | ---: | ---: |
| First event alone, December 16 | +0.54 [−5.77, 6.86] | −0.76 [−2.35, 0.83] | −0.64 [−2.13, 0.84] |
| Average of all 21 event days | −0.70 [−8.81, 7.42] | −2.00 [−4.10, 0.11] | −2.16 [−3.76, −0.55] |
| All 21 event-plus-next-day windows, overlaps allowed | −1.46 [−17.60, 14.68] | −4.06 [−8.17, 0.05] | −4.45 [−7.61, −1.30] |
| Pre-announcement check, December 8–14 versus December 1–7 | +0.95 [−5.71, 7.62] | −0.69 [−1.74, 0.35] | −0.61 [−1.65, 0.43] |

Across all 21 events, baseline subtraction gave −1.08 kWh [−2.22, 0.06] during the 13 peak hours and −0.92 [−1.98, 0.14] during the other 11 hours. For the overlap-allowed calculation a following day can include another intervention or its announcement, and an overlapping calendar day contributes to both windows. The placebo’s unadjusted value is the arm gap in the second pre-announcement week. The adjusted columns use the first week as baseline. A published analysis across Norwegian regions reported 2.92% lower consumption in high-price hours. That result benchmarks the direction of peak-hour response in a broader population and on a different scale. The Trondheim totals and next-day estimates are new analyses. [Published electricity study](https://doi.org/10.1016/j.enpol.2023.113821) · [Earlier Trondheim peak-hour replication](../pilots/iflex.md)

## Reproduce

The script checks the SHA-256 of the versioned iFlex R workspace, decodes the household-meter data with base R, and outputs aggregate JSON. A warm cache still runs the analysis from hourly readings. Run from the repository root.

```bash
export DML_TS_BENCHMARK_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/dml-ts/external"
OPENBLAS_NUM_THREADS=1 uv run --frozen python \
  docs/audits/2026-11-19_time_series_dml/external/questions/energy.py \
  --cache-dir "$DML_TS_BENCHMARK_CACHE"
TEST=test/integration/test_questions_energy_retail.py
DML_TS_EXTERNAL_REPLAY=1 uv run --frozen --extra dev pytest -o addopts='' -q \
  "$TEST::test_followup_excludes_adjacent_events_and_new_announcements" \
  "$TEST::test_energy_events_are_aggregated_within_each_household_by_calendar_date" \
  "$TEST::test_source_reanalysis[energy]"
```
