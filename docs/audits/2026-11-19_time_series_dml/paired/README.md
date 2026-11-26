# When does a time-forward split change the causal estimate?

On a stable serial process with a nonlinear interaction and 1,200 dates, boosted-tree DML estimated the unit effect with RMSE **0.075** under shuffled cross-fitting and **0.079** with a five-date gap and past-only training. Every comparison used the same scoring dates, training count and learner. Changing the learner mattered more. On the same buffered forward folds, linear adjustment had RMSE **0.600** and additive splines **0.237**. The tree intervals still missed the true effect in one in five replications. [Summary and Monte Carlo errors](summary.csv) · [Paired differences](paired.csv)

## How the comparison works

Each of 500 seeds generated 300- and 1,200-date processes. Within each size and seed, all methods saw the same observations. The four designs use independent observations, stationary serial observations, a break in the effect, or a persistent hidden common cause. Observed covariates enter the outcome and treatment through the same nonlinear function, including a threshold interaction. In the stationary serial design, independent Gaussian streams each have AR coefficient 0.6. In the hidden design one additional latent stream enters both treatment and outcome with unit loading. The structural effect remains one. In the break design it falls from one to −0.5 at the midpoint. [Generator and estimators](experiment.py)

Four folds score the same contiguous interior span, dates 75–222 or 300–899 (zero-based). Each fit uses 70 or 295 training dates. Shuffled folds mix evaluation dates across this span and sample training dates throughout the record. Two-sided buffered folds train before and after each evaluation block. Past-only folds train on the most recent eligible dates. Buffered folds skip five dates next to the scoring block. The fixed memberships make the Monte Carlo results conditional on these particular partitions. Heterogeneity over time is evaluated against the date-average structural effect on exactly the scored dates. Residual treatment variance in the data-generating process is constant over dates. [Actual training and scoring indices](folds.json)

All three nuisance learners use the same residual-on-residual score. The spline uses an unpenalized, additive cubic basis with fixed knots. The tree learner uses 32 boosting iterations. Both covariance calculations use each fitted score sequence in chronological order. The IID variance uses the sample variance, and HAC uses the temporalcv automatic Bartlett bandwidth. Each row below represents 500 estimates. The coverage Monte Carlo SEs for the tree entries are 1.66–1.79 percentage points. [Per-draw estimates, SEs and prediction losses](draws.csv.gz)

| Stable serial process, 1,200 dates | Bias | Effect RMSE | IID coverage | HAC coverage | Mean HAC interval width |
|---|---:|---:|---:|---:|---:|
| Trees, shuffled | +0.047 | 0.075 | 71.8% | 83.6% | 0.203 |
| Trees, two-sided, gap 5 | +0.045 | 0.081 | 70.4% | 80.2% | 0.208 |
| Trees, past only | +0.044 | 0.078 | 69.4% | 81.8% | 0.208 |
| Trees, past only, gap 5 | +0.046 | 0.079 | 70.4% | 80.0% | 0.208 |
| Additive spline, past only, gap 5 | +0.230 | 0.237 | 0.4% | 1.0% | 0.193 |
| Linear, past only, gap 5 | +0.599 | 0.600 | 0.0% | 0.0% | 0.152 |

Switching trees from shuffled to buffered forward fitting changed bias by −0.0012 (paired 95% Monte Carlo interval −0.0049 to +0.0026). Mean squared effect error rose by 0.00060 (+0.00005 to +0.00115) and HAC coverage fell by 3.6 percentage points (−7.1 to −0.1). With only 300 dates and 70 training rows per fit, buffered forward tree RMSE rose to **0.252** and coverage fell to **39.6%**. The small-sample spline was unstable in a few realizations. Median outcome prediction MSE was 6.34, the mean was 156, and the maximum was 55,251. All draws are included. These results describe the stated learner settings and sample sizes. [All designs and sample sizes](summary.csv) · [Paired Monte Carlo uncertainty](paired.csv)

## What rolling validation measures

At each origin the effect estimate pools only earlier, already scored blocks. The nuisance models are trained on earlier dates and held fixed over the next block. They can use each date's covariates as those covariates arrive. The outcome prediction loss measures prediction of individual outcomes. The frozen effect is compared with the known next-block causal effect. A separate next-block estimate uses the held-out outcomes and treatment variation to show how well that effect could actually be measured. The next-block interval uses its own chronological scores. Overlapping estimation histories make the three origins dependent. [All origins, counts, prediction losses and uncertainties](rolling.csv) · [Rolling draws](rolling-draws.csv.gz)

| Changing effect, trees, 1,200 dates | Frozen effect | Next-block truth | Next-block estimate | Mean next-block 95% interval width | Outcome prediction MSE |
|---|---:|---:|---:|---:|---:|
| Dates 450–599 | 1.043 | 1.0 | 1.055 | 0.394 | 2.814 |
| Dates 600–749 | 1.049 | −0.5 | −0.420 | 1.060 | 9.192 |
| Dates 750–899 | 0.556 | −0.5 | −0.397 | 0.664 | 3.897 |

The first block after the break gives two pieces of evidence. The historical effect misses the new effect by **1.549**, and the later estimate has a much wider interval. By the next origin prediction loss has fallen substantially, while the frozen causal estimate still has the wrong sign. In the hidden-cause control, the opposite problem occurs. Frozen and later estimates both stay around **1.49–1.50** across all three origins while the structural effect is **1**. Historical agreement can coexist with systematic causal bias. [Rolling results by design](rolling.csv)

## An exact standard-error check

With the true nuisance functions, the score at the true effect is the product of the independent treatment and outcome disturbances. Its lag-kk correlation is the product of their two lag-kk correlations. If both disturbances have AR coefficient 0.6, the infinite-horizon variance multiplier is `(1 + 0.36)/(1 - 0.36) = 2.125`. If only the outcome disturbance persists, the multiplier is **1**. At 1,200 dates the measured variances of the normalized sum about its known zero mean were **2.120** (MC SE 0.136) and **1.004** (0.063). The mean automatic HAC estimates were **1.969** and **1.004**. This explains why serial dependence in outcomes alone does not determine how much the causal SE should change. [Analytic targets and measured variance](oracle.csv) · [Oracle draws](oracle-draws.csv.gz)

## Reproduce

From the repository root run the command below. Across 4,000 data paths, all 48,000 matched estimates and 36,000 rolling comparisons were finite. Workers use one numerical-library thread each. Replaying the first two seeds on one worker gave exactly the same archived values for all 192 matched estimates, 144 rolling comparisons and eight oracle draws across both sample sizes and all regimes. The driver also checks an individual path against a serial fit on every run. Use `--output` on the standalone driver for a separate destination. [Run record](run.json) · [Driver](experiment.py)

```sh
uv run --locked python docs/audits/2026-11-19_time_series_dml/external/run.py paired-time-splits --jobs 32
```
