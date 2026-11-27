# What do better causal methods change about five historical questions?

**November 20, 2026.** Baseline adjustment made a two-day reduction in electricity use easier to detect. In a randomized reminder trial, the absolute benefit fell over the month under both the design-based and adjusted analyses. For minimum wages, changing the comparison geography mattered more than changing the learner. The five reports below start from substantive questions and compare methods on a common outcome and sample wherever the design permits.

| Historical question | What the data say | What the framework adds |
|---|---|---|
| **Do grocery promotions borrow sales from the following week?** | A price cut of at least 10% was associated with higher current sales of the discounted brand and lower current sales of its two rivals. In the forward ridge fit, the following-week change for the promoted brand was **+1.35 thousand fluid ounces** (95% interval −6.86 to +9.55). A negative next-week effect was not established. | Defining the sale using only current and past prices, examining rival sales, and showing pre-sale changes exposes alternative explanations for a post-sale dip. The scanner panel measures aggregate purchases at these stores. [Retail report](questions/retail.md) |
| **Does an electricity incentive move demand into the following day?** | For 405 retained Trondheim households on seven isolated event starts, baseline-adjusted consumption was **2.08 kWh lower on the next day** (−3.65 to −0.52) and **4.01 kWh lower over 48 hours** (−7.16 to −0.87). The corresponding raw 48-hour contrast was −1.41 kWh (−16.12 to +13.30). | Cross-fitted ridge estimated −4.09 kWh over the same windows (−6.97 to −1.22). The gain came largely from using pre-announcement meter readings. The comparison concerns assignment to the ongoing incentive program on these dates. [Electricity report](questions/energy.md) |
| **Do higher minimum wages reduce restaurant employment?** | In historical level specifications, the employment elasticity was **+0.016** using neighboring counties and **−0.141** using state portions of commuting zones. Using the geographic state-pair crosswalk in the latter regression increased its SE from **0.0705 to 0.1391** with the same coefficient and rows. | In a separate one-quarter change analysis, switching from linear to forest adjustment moved the employment slope from −0.029 to −0.037 for county pairs and from −0.128 to −0.108 for commuting-zone pairs. The study also reproduces the original two-state restaurant survey and examines pre-policy employment, earnings, geographic duplication and different sources of uncertainty. [Minimum-wage report](questions/minimum-wage.md) |
| **Does a fine deter late day-care pickups?** | Fined centers recorded **9.22 more late pickups per center-week** during the fine, after subtracting the baseline gap. Formerly fined centers recorded **12.46 more** after repeal. In each period only one of the 210 possible center assignments produced as extreme a change. | Randomization gives the main inference with ten independent centers. A simple baseline-adjusted fit gave +9.66 during the fine. All fines ended together, so the post-period comparison cannot isolate the effect of repeal itself. [Day-care report](questions/daycare.md) |
| **Do app reminders lose effectiveness with repeated use?** | For the same 349 people, the adjusted absolute effect on opening the app within one hour fell from **+14.75 percentage points** in the first week to **+5.98** in the last. The paired difference was **−8.77 points** (−11.88 to −5.66). | Known randomization probabilities alone gave a decline of 8.73 points. Distinguishing a change over study time from the effect of past reminders on responsiveness would require an explicit treatment-history comparison. [Reminder report](questions/reminders.md) |

## What generalizes

With genuine repeated assignment, compare the randomized contrast and the adjusted score on the same eligible occasions. For a fixed household assignment, changes across days still share an assignment unit. Ten centers permit 210 assignments of six labels even though the table has 200 rows. For observational exposure histories, evaluate support and leads before interpreting lagged sales or employment as effects. In all five cases, the uncertainty calculation depends on who or what could independently have received a different treatment.

The applicability guide evaluates three additional questions: fish demand with an instrument, highway congestion around one transit strike, and emergency-department utilization after a Medicaid lottery. It maps the intervention, horizon, source of variation and pre-treatment information to a defensible comparison, and gives a diagnostic that could undermine the interpretation. The transit and Medicaid entries are design assessments of published work. [Guide to other time-series causal problems](questions/applicability.md)

## Reproduce the research

From the repository root, the following commands run all five studies or a named comparison. Each study verifies the original input checksums and prints numerical results. The runner reuses the same cache as the original replication suite, limits numerical libraries to one thread per study and defaults to four concurrent studies. The question suite uses the locked project Python environment, base R for the original Norwegian workspace and Drink Less RDS, and Poppler's `pdftotext` for the day-care count table.

```bash
audit=docs/audits/2026-11-19_time_series_dml
export DML_TS_BENCHMARK_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/dml-ts/external"
uv run --locked python "$audit/external/run.py" --suite questions --jobs 4
uv run --locked python "$audit/external/run.py" wage-employment daycare-fines

uv run --locked --extra dev python -m pytest -q -m 'not network' test/integration/test_question*.py
DML_TS_EXTERNAL_REPLAY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked --extra dev python -m pytest -q test/integration/test_question*.py
```

Calling the runner without a study name selects the ten earlier replications. The combined `--suite all` now also includes the forward benchmarks and requires the separate Italian author environment. [Forward validation results](13-forward-causal-validation.md) [Original benchmarks and setup](11-external-replication-benchmarks.md#reproduce-the-checks)

The four-worker run completed all fifteen studies on November 20. All sixteen tests for these five reanalyses passed with source replays enabled. They include checks against direct group means, exhaustive center reassignment, within-participant uncertainty, household energy totals and a synthetic panel with a known effect. A broader offline run of replication, estimator-property and scientific-reference tests passed 53 cases.
