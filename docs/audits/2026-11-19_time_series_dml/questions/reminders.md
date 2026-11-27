# Does a reminder still prompt an app opening in the last week?

**Yes. Sending a Drink Less reminder increased the chance of opening the app in the next hour by about 15 percentage points in days 1–7 and 6 points in days 24–30.** The estimated absolute benefit fell by **8.77 points** (participant-paired 95% CI −11.88 to −5.66). An estimate using only the randomization probabilities gave a decline of 8.73 points.

| One-hour effect of assigning either reminder, percentage points | Days 1–7 | Days 24–30 | Late minus early |
| --- | ---: | ---: | ---: |
| Inverse probability weighting | 14.26 [11.34, 17.18] | 5.53 [3.39, 7.66] | −8.73 [−12.02, −5.45] |
| Cross-fitted augmented inverse probability weighting | 14.75 [11.96, 17.54] | 5.98 [3.84, 8.12] | −8.77 [−11.88, −5.66] |

Bell and colleagues randomized each of 349 people at 8 pm every evening for 30 days. The probabilities were 0.3 for a new notification, 0.3 for the standard notification and 0.4 for none. Each contrast estimates the change in the chance of opening the app from 8 to 9 pm when assigned one of the two equally likely reminders, averaged over people, days in that window, and the preceding assignments under the study policy. Each person contributed seven decisions to each window. [Trial design and primary outcome](https://mhealth.jmir.org/2023/1/e38342/) · [Public trial data](https://osf.io/w3szp/)

| Observed one-hour openings | Days 1–7 | Days 24–30 |
| --- | ---: | ---: |
| Any notification | 278 / 1,452 (19.15%) | 120 / 1,440 (8.33%) |
| None | 46 / 991 (4.64%) | 26 / 1,003 (2.59%) |

The first estimator weights each observed opening by the reciprocal of its randomized assignment probability (0.60 or 0.40) and divides by the total number of decisions in the window. The second also predicts opening under each assignment from age, AUDIT score, study day, sex, employment type and two pre-notification measures of app use. It fits separate regularized logistic regressions for each arm and window. Five folds divide participants by sorted study ID, keeping every decision from a person in the same fold in both windows. Both methods score all 4,886 decisions. Their risk differences can differ slightly from subtracting the observed arm percentages, which use the realized arm sizes as denominators.

For uncertainty the analysis forms an early and a late mean score for each of the 349 participants. It takes the covariance of those pairs, divides by 349, and uses a Student t critical value with 348 degrees of freedom. This gives a standard error of 1.58 percentage points for the late-minus-early augmented contrast. Ignoring the within-person covariance would give 1.79 points.

The paper’s Table 2 reports an all-month relative risk of 3.523 [2.918, 4.255]. The command reproduces this result before fitting the two absolute contrasts. Table 3 reports a multiplicative change in relative risk per day of 0.993 [0.975, 1.012]. A lower baseline opening probability can yield a smaller absolute gain even with a stable relative effect. Also, elapsed study time was shared by all participants and previous reminders accumulated under the trial policy. The observed decline alone cannot identify whether repeated exposure caused habituation. The study’s outcome measures app use and gives no direct estimate of a change in alcohol consumption. [Published Tables 2 and 3](https://mhealth.jmir.org/2023/1/e38342/) · [Author analysis](https://github.com/lauren-bell/DrinkLessMRT/blob/9beb720e39d987b268b0423a0fdf91aea0cfbb20/pooled%20estimates.R)

## Reproduce

The command requires base R for reading the original RDS and the repository’s Python environment for estimation. It checks the download’s SHA-256, shares the same cache as the [full-month replication](../external/sequential/README.md), and prints aggregate results as JSON.

```bash
export DML_TS_BENCHMARK_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/dml-ts/external"
uv run --frozen python docs/audits/2026-11-19_time_series_dml/external/questions/reminders.py \
  --cache-dir "$DML_TS_BENCHMARK_CACHE"
DML_TS_EXTERNAL_REPLAY=1 uv run --frozen --extra dev pytest -o addopts='' -q \
  test/integration/test_question_reminders.py
```
