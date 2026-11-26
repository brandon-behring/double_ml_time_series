# Washington, DC bike rentals and temperature

On 584 scored days, the temporal partially linear regression estimated **365 additional rentals per day for a 0.1 increase in normalized temperature**, with a 95% interval of **195 to 535** using a seven-day HAC bandwidth. A separate experiment assigned simulated treatments with known probabilities to the same dates. Its nominal 95% intervals covered the true effect in **141 of 200 assignments (70.5%)**. Agreement with an ordinary regression on the observed data therefore gives limited evidence about this estimator's interval calibration. [Pilot script](bike.py)

## Data and comparison

The downloaded UCI archive contains 731 consecutive daily observations from January 1, 2011 through December 31, 2012. The 17,379 hourly rows represent a separate aggregation and were excluded. We verified unique dates on the full daily grid, complete daily fields and the identity `cnt == casual + registered`. The two component counts would reveal the outcome and were excluded from the design. UCI licenses the files under CC BY 4.0 and credits Hadi Fanaee-T. The source study used the series to detect events. It supplies no causal temperature coefficient against which to score this estimate. [UCI archive and variable definitions](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) · [Source study](https://doi.org/10.1007/s13748-013-0040-3)

The outcome is total rentals, `cnt`. The exposure is `10 * temp`, so each one-unit contrast corresponds to one tenth of UCI's normalized daily temperature scale. The reference adjustment uses a linear time trend, annual and semiannual sine/cosine pairs, weekday indicators and a holiday flag. This targets an average conditional linear association, under an assumed common slope. Demand changed as the service grew. Unobserved events, system capacity and nonlinear temperature response limit a weather-intervention interpretation. The co-weather sensitivity additionally conditions on humidity, wind speed and weather category. Perceived temperature, `atemp`, shares much of the exposure signal and is excluded throughout.

## Observed series

Both nuisances used the repository's ridge learner with four expanding forward folds. The reference used zero treatment lags, a seven-day train/test gap and Bartlett HAC with seven lags. All specifications scored 584 days from May 28, 2011 to December 31, 2012. The reference discarded the 147-day initial prefix and achieved out-of-fold outcome/treatment R² of 0.345/0.434. Mean squared residualized treatment was 1.737 in the rescaled exposure units. The first training fold saw a maximum exposure of 6.32 and the following test fold reached 8.49. The third test fold also exceeded its training maximum. Seasonal extrapolation therefore affects two folds. [Estimator implementation](../../../../dml_ts/dml/temporal_plr_dml.py) · [Pilot specifications](bike.py)

| Specification | Slope in rentals/day | SE | 95% interval |
| --- | ---: | ---: | ---: |
| Calendar, seven-day gap, HAC(7) | 364.95 | 86.68 | [195.06, 534.85] |
| HAC(0) | 364.95 | 42.59 | [281.48, 448.42] |
| HAC(28) | 364.95 | 138.38 | [93.74, 636.17] |
| Zero-day gap, HAC(7) | 437.81 | 111.60 | [219.07, 656.55] |
| Adjust for yesterday's temperature, HAC(7) | 596.91 | 137.65 | [327.13, 866.69] |
| Add co-weather controls, HAC(7) | 386.62 | 89.70 | [210.80, 562.43] |

The full-sample OLS slopes were 664.07 with temperature alone and 365.15 with the calendar controls. They use all 731 days. Conditioning on yesterday's exposure changes the contrast to a same-day increase at fixed prior temperature. The substantial slope change and wider uncertainty deserve attention whenever a chapter proposes a lagged specification for a similar series.

## Known-effect calibration

We held the daily rental trajectory fixed as the untreated potential outcome. In the reference experiment we generated independent binary assignments conditional on calendar features. The daily probability was `expit(-0.4 + 0.8 * workingday + 0.45 * sin(2πt/365.25))`, where `t` counts days since January 1, 2011. Probabilities ranged from 0.299 to 0.701. For each of 200 seeded draws we constructed outcomes `cnt` and `cnt + 200 * D`. The second outcome adds 200 rentals on treated days by construction. The temporal estimator used the same calendar controls, four folds and seven-day gap and HAC settings as above. Simulated assignment is independent of potential outcomes given the calendar. Serial dependence, growth and seasonality in the untreated outcomes are inherited from the data.

| True effect | Mean estimate | Bias (MCSE) | RMSE | Mean reported SE | 95% interval coverage (Wilson interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | −153.47 | −153.47 (12.83) | 237.34 | 131.46 | 70.5% [63.8%, 76.4%] |
| +200 | 46.53 | −153.47 (12.83) | 237.34 | 131.46 | 70.5% [63.8%, 76.4%] |

The paired estimates shifted by exactly 200 to numerical precision. With identical ridge learners, a shift proportional to treatment passes through both nuisance residualization and the final slope. This verifies the estimator's response to an added effect while exposing how far the unshifted estimate can lie from zero. Across the 400 reference fits, 34 emitted the repository's poor-nuisance warning. The randomized design also permits a benchmark using the known assignment probabilities. The centered Horvitz–Thompson contrast on the same scored days had bias −3.66 (MCSE 11.88) under the null and −3.88 (MCSE 12.16) under +200, with RMSEs of 167.69 and 171.52. The Monte Carlo intervals quantify uncertainty across assignments conditional on this one observed trajectory. These results concern the chosen nuisance learners, folds and assignment law.

### Where the null bias enters

For independent daily assignments, the expected numerator of the final regression can be calculated exactly. In each fold, apply the fitted ridge prediction weights to the **true assignment probabilities** in the training days. Subtract these predictions from the probabilities on the test days and multiply by the corresponding outcome residuals. The sums in the four folds are −23,444, +1,316, +50 and +264. The numerator therefore averages −21,813.55 under a true zero effect. The first fold accounts for the negative shift. Dividing by the expected denominator of 160.301 gives −136.08. The empirical mean of the estimated slopes is −153.47 because the denominator also changes with each draw. A change to the HAC bandwidth can change the interval width while leaving this score bias intact. [Exact calculation and assignment law](bike.py)

Using the actual probabilities to center each test-day assignment gives a score with zero expectation. We tested each injected constant effect by subtracting it from the treated outcomes and comparing the absolute score with 10,000 independent redraws from the daily Bernoulli assignment law. At a 5% level, the randomization test retained the true effect in **191 of 200 assignments (95.5%)**, for either train/test gap. Both targets restore the same untreated trajectory when tested at their true value. The calculation uses a shared simulated reference distribution for the 200 tests and applies to the independent-assignment arms. [Randomization test](bike.py)

### Persistence and train/test gap

We repeated both injections with identical seeds across four scenarios. At persistence 0.7, the intervention repeats the preceding day's assignment with probability 0.7 and draws afresh with probability 0.3. Thus the true conditional assignment probability is `0.7 * D_yesterday + 0.3 * p_today`, spanning 0.090 to 0.910. Outcomes have no added carryover. A fresh calendar-dependent intervention daily corresponds to persistence zero. Each design uses the same 584 scored days, four folds, and HAC(7). The gap changes which earlier observations can train the nuisances. MCSE for coverage uses 200 independent assignments per scenario. The null and +200 runs are paired within an assignment. [Full scenario grid](bike.py)

| Retention probability | Gap in days | Mean estimate, null / +200 | Bias (MCSE) | RMSE | Mean SE | Coverage of 95% intervals (MCSE) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7 | −153.47 / 46.53 | −153.47 (12.83) | 237.34 | 131.46 | 70.5% (3.2 pp) |
| 0 | 0 | −136.61 / 63.39 | −136.61 (14.59) | 247.06 | 131.03 | 68.5% (3.3 pp) |
| 0.7 | 7 | −58.39 / 141.61 | −58.39 (20.38) | 293.39 | 165.98 | 72.0% (3.2 pp) |
| 0.7 | 0 | −46.09 / 153.91 | −46.09 (22.47) | 320.35 | 162.67 | 64.5% (3.4 pp) |

The seven-day gap gave higher measured coverage under both assignment laws. Every scenario still fell well below the nominal 95%, and the persistent design increased RMSE even as absolute bias fell. At persistence 0.7, the known-propensity contrast using the realized assignment history had biases of −18.26 (MCSE 17.28) and −19.60 (17.65) at the two targets. Changing the training gap leaves this benchmark unchanged because it uses a known probability on the same evaluation days.

## Reproduce

Run from the repository root with Python dependencies from the repository lock. The pilot used Python 3.13.17, temporalcv 2.2.0 and scikit-learn 1.6.1. The script verifies the archive SHA-256 and prints all diagnostics. It uses NumPy SeedSequence 20261119 with one spawned stream per assignment. [Executable analysis](bike.py)

```sh
curl -fLsS 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip' -o /tmp/dml-audit-bike.zip
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python docs/audits/2026-11-19_time_series_dml/pilots/bike.py /tmp/dml-audit-bike.zip
```

Archive SHA-256 is `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`. The extracted `day.csv` SHA-256 is `a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be`.
