# Bangalore congestion pricing

The published high-rate assignment reduced the normalized daily hypothetical charge rate by **12.94 points** relative to control, with a commuter-clustered SE of **6.04**. We reconstructed Table I from the author’s trip, GPS quality, assignment and daily report files. The original estimate and all four planned checks reproduce. In the full sample, allowing the pre-to-post change to vary with baseline travel histories reduced the SE to **5.78**. That precision gain did not carry through consistently when later months were evaluated individually. [Published paper, Table I, p. 1247](https://drive.google.com/file/d/16NMJE_VYGDC9qaAz9DhAXvyLrTrKTvmy/view) · [Versioned replication package](https://zenodo.org/records/10637043)

## Published result

Each eligible trip receives a hypothetical per-kilometre rate on the commuter’s personalized peak schedule, normalized to 100 at the maximum. The outcome adds these rates across qualifying trips that day. The rate schedule was defined for every participant before randomization. A lower value can come from retiming or taking fewer eligible trips. The effect is measured in normalized rate points per commuter-day. Actual payments also depend on trip distance and the assigned charge. [Published design and outcomes, sections 6.1–6.2](https://doi.org/10.3982/ECTA18422)

| Table I, panel A, column 1 | Published | Reconstructed |
|---|---:|---:|
| High rate × post | −12.94 | −12.94491 |
| Commuter-clustered SE | 6.04 | 6.03777 |
| Commuter-days | 15,574 | 15,574 |
| Post-period control mean | 96.88 | 96.87823 |

The replay uses the 497 completed commuter assignments. It reproduces the author’s day-quality categories, trip exclusions, no-trip zeros, and assigned early/late three-week schedules. A day tagged “bad” is still usable under the author’s coding. A half-day or missing day is excluded. The model absorbs commuter intercepts and includes study-cycle effects and the high, low and information assignments interacted with post. It uses the cluster correction for Stata’s `areg`, including absorbed intercepts in the residual degrees of freedom. Low rate and information also match the table to rounding at −8.85 (6.13) and −0.19 (5.40). The earlier working paper prints −12.29 on 15,585 days. [Published table](https://drive.google.com/file/d/16NMJE_VYGDC9qaAz9DhAXvyLrTrKTvmy/view) · [Working paper](https://www.nber.org/papers/w30903) · [Measured results](results.json)

## What did adjustment add?

The second full-sample fit used the same rows and fixed effects. We interacted post with each commuter’s mean and standard deviation of the daily hypothetical rate, mean daily trip count and number of observed baseline days. The history stops strictly before roster addition, a conservative operational cutoff for assignment. All commuters had 3–62 qualifying days. The adjusted high-rate coefficient was **−14.46 (SE 5.78)**, a **4.3%** reduction in the estimated SE. This is a regression sensitivity analysis on the same trial. It cannot establish that the shift toward a larger effect removed bias. [Reconstruction and feature definitions](replay.py) · [Measured alternatives](results.json)

For each calendar month we also estimated the high-versus-control contrast in the change from the commuter’s own frozen pre-assignment mean. Only commuters already on the roster at the start of the month contributed. Each person supplied one average over their usable scheduled days in that month. We weighted each arm by the inverse of its assignment probability conditional on high or control, obtained from the deposited randomization schedule by stratum. These probabilities range from 0.20 to 0.60 for high rate. This targets a participant-average observed change among eligible participants in the two arms. The published daily within-person regression weights observations differently.

The adjustment predicts untreated change from the same four baseline features with standardized ridge regression and fixed penalty 10. For each held-out group of commuters it trains on earlier control outcomes from other commuters, using five deterministic groups. The two arm means then use the residual change and the same design weights. The month-level SE is a sandwich estimate for the weighted arm means with one observation per commuter, conditional on the fitted adjustment. It does not measure between-origin selection, model-training variability or common calendar shocks. Commuters can contribute to adjacent months, so these estimates are dependent. [Rolling implementation](replay.py)

| Month, 2017 | Scored high / control commuters | Earlier high–control change | Observed change (SE) | Ridge-adjusted change (SE) | Control prediction RMSE, ridge / zero change |
|---|---:|---:|---:|---:|---:|
| April | 11 / 15 | −34.45 | −21.16 (21.29) | −23.94 (23.07) | 77.74 / 68.98 |
| May | 14 / 21 | −22.89 | −16.13 (12.87) | −11.99 (13.44) | 41.53 / 38.47 |
| June | 18 / 16 | −19.12 | −15.12 (15.09) | −14.54 (17.47) | 53.60 / 49.04 |
| July | 39 / 50 | −14.48 | −14.86 (11.40) | −13.26 (11.30) | 58.44 / 56.97 |
| August | 20 / 18 | −14.34 | −7.91 (21.81) | −0.69 (20.39) | 55.11 / 60.29 |

The earlier contrast pools information available by the end of the previous month. April’s earlier contrast uses only 14 participants and has SE 49.60. March has no prior information. September has one observed control commuter and no observed high-rate commuters. August is the only scored month where the ridge predictor beat the zero-change forecast for controls. Its adjusted effect is also the least precisely distinguished from zero in magnitude. With 20 high-rate and 18 control commuters and an SE over 20 points, the data allow both persistence and substantial change. [All monthly estimates, support and SEs](results.json)

## GPS availability and assignment support

The table gives **usable / scheduled** days for both arms, including people enrolled during the month. The narrower rolling cohort requires enrollment before the month starts. The assignment unit is the commuter. The daily records supply repeated measurements of the original assignment.

| Month | High-rate days | Control days |
|---|---:|---:|
| March | 30 / 42 | 47 / 57 |
| April | 121 / 168 | 220 / 286 |
| May | 180 / 249 | 236 / 327 |
| June | 295 / 360 | 450 / 579 |
| July | 464 / 605 | 587 / 769 |
| August | 92 / 131 | 66 / 99 |
| September | 0 / 0 | 5 / 10 |
| Total | 1,182 / 1,555 | 1,611 / 2,127 |

Overall availability was 76.0% under high charges and 75.7% in control. These similar rates leave open the possibility that missing days differ in travel behavior. The receipt includes scheduled and observed people and days for all four arms. [Monthly support and attrition](results.json)

## Replay

From the repository root, run the following command. It verifies the SHA-256 of six original ZIP members and downloads about 30 MB of compressed inputs on first use. Raw travel data stay in the external cache. The default output is aggregate JSON on standard output. Pass `--output PATH` to save it. The command exits with an error if any of the four published targets fail at the paper’s printed precision. [Member hashes](source-members.json) · [Source archive](https://zenodo.org/records/10637043)

```bash
.venv/bin/python docs/audits/2026-11-19_time_series_dml/external/run.py bangalore-pricing
```
