# Forecasting reminder effects for new participants

At each quarter start we forecast the effect of assigning an activity prompt versus no prompt on log(next-hour steps + 0.5) during study days 14–97 among new enrollees. Every eligible observed decision has equal weight. All seven enrollment quarters appear below. Four support both historical methods and between-person error intervals. The final quarter has one new participant. These are retrospective operator forecasts using a public release published after the study. [Original trial](https://doi.org/10.2196/78218) · [Pinned decisions and codebook](https://github.com/WIRED-L-um/MRT_myBPmyLife/tree/fa86c0d44ced3876f4a7351bc2d475fc4249342e)

| Forecast issued | Earlier / new people | Earlier arm difference | Earlier AIPW | Later randomized effect | AIPW error and 95% interval |
|---|---:|---:|---:|---:|---:|
| 2022-01-01 | 0 / 33 | — | — | -0.0436 | — |
| 2022-04-01 | 0 / 52 | — | — | +0.0767 | — |
| 2022-07-01 | 28 / 48 | -0.0450 | -0.0352 | +0.0728 | -0.1080 [-0.2575, +0.0396] |
| 2022-10-01 | 83 / 39 | +0.0371 | +0.0309 | -0.0276 | +0.0585 [-0.0685, +0.1812] |
| 2023-01-01 | 131 / 58 | +0.0424 | +0.0370 | -0.0065 | +0.0435 [-0.0597, +0.1446] |
| 2023-04-01 | 172 / 56 | +0.0271 | +0.0263 | -0.0313 | +0.0576 [-0.0417, +0.1599] |
| 2023-07-01 | 226 / 1 | +0.0170 | +0.0101 | +0.6272 | -0.6171 |

The zero forecast can be issued at every origin. Both other forecasts require mature earlier outcomes. At each of the four fully evaluable dates, the AIPW forecast and later point estimate fell on opposite sides of zero. Each corresponding signed-error interval includes zero. The July 2023 point describes the one available participant. The signed error is the frozen forecast minus the later randomized contrast.

## Incremental forecast accuracy

We gave each of the four fully evaluable quarters equal weight. A positive difference in squared errors favors AIPW. The intervals propagate uncertainty in the people sampled in these particular quarters. Four dates provide limited evidence about new calendar periods.

| Comparator | Mean squared-error gain of AIPW (log units squared) | 95% interval |
|---|---:|---:|
| Zero | -0.003301 | [-0.014648, +0.000572] |
| Earlier arm difference | +0.000895 | [-0.002196, +0.005522] |

For tolerance sensitivity we ask whether a pointwise 90% signed-error interval fits entirely within each symmetric margin. Each cell counts qualifying quarters among the four shared evaluation dates. These are illustrative accuracy margins in log units. The trial supplies no monetary break-even threshold for acting on the forecast.

| Method | ±0.025 | ±0.05 | ±0.10 |
|---|---:|---:|---:|
| zero | 0/4 | 0/4 | 1/4 |
| historical | 0/4 | 0/4 | 0/4 |
| aipw | 0/4 | 0/4 | 0/4 |

## How the estimates were produced

The temporalcv splitter operates on a complete calendar-day grid with a 98-day gap. Enrollment date plus 98 days must precede the forecast date for every training participant. Evaluation uses new people enrolled in the next three months. The archive gives dates of enrollment and relative study days. The one-day allowance after day 97 covers a final next-hour outcome. Historical ingestion dates are unavailable. Eligibility requires the trial’s active phase, pre-decision heart rate, valid assignment and baseline covariates. The CSV reports outcomes missing after these filters by period and arm. Complete-case causal interpretation assumes that assignment did not selectively affect observation.

The earlier and later randomized contrasts each subtract the observed no-prompt arm mean from the activity-prompt arm mean. Their assignment probabilities are 0.75 and 0.125, respectively. The later contrast is a noisy, self-normalized estimate with random denominators. Paired loss gains describe the comparison against that estimate. A third arm assigns diet prompts.

DML uses the known randomization probabilities and separate ridge regressions for the two arms. Age, study day, baseline steps and steps in the prior 30 minutes enter with fixed scaling. Each earlier participant stays in one of five folds. Each residual correction uses predictions trained on the other folds among the mature history. The AIPW score averages over all earlier eligible complete decisions, including diet-arm decisions. This forecasts the earlier population's marginal effect for the new cohort without using its covariates. The original single-cutoff analysis pools multiple later quarters. [Original split and eligibility](../forward/reminders.md)

The 999 shared bootstrap draws use one exponential weight per person, normalized within enrollment quarter and reused across dates. Each draw refits both arm regressions, recomputes the earlier and later effects, and then recomputes the paired losses. Calendar periods and fold membership stay fixed. Follow-up overlaps calendar weeks across quarterly enrollment cohorts. These intervals condition on the observed calendar and do not propagate shared calendar shocks to a new year. The reported percentile intervals are pointwise.

## Reproduce

`uv run --locked python docs/audits/2026-11-19_time_series_dml/external/forward/reminder_rolling.py --cache-dir /tmp/dml-external-bench-cache --output-dir docs/audits/2026-11-19_time_series_dml/rolling`

[All 21 method-date rows, arm counts, missing outcomes and intervals](mybp-origins.csv) · [Paired summary and freeze digests](mybp-summary.json) · [Frozen forecasts](mybp-freezes/) · [Source and scoring code](../external/forward/reminder_rolling.py)
