# When does flexible adjustment improve a known causal estimate?

**Tree DML reduced estimation error when treatment and outcome shared nonlinear confounding. A conventional additive spline captured nearly all the gain in the additive designs. Trees had the lowest error among the five primary methods when the confounding also included an interaction. Even there, their nominal 95% intervals covered the true effect in only 80.6% of independent datasets and 85.6% of serial datasets.** These are accuracy and coverage results for a known contemporaneous causal effect. The design supplies no test of whether its effect would change in a future population.

## A comparison on the same observations

Each of six cells has 500 paired datasets of 1,200 observations. Every method sees the same treatments, outcomes and covariates in each comparison. There are four observed covariates. The treatment is continuous and the true common slope is one. In each design we generate treatment and outcome as follows.

```text
D = q(X) + v
Y = D + q(X) + u
```

The linear setting uses `q = 0.65 X₀ + 0.45 X₁`. The additive setting uses `sin(2 X₀) + 0.6 X₁²`. The interaction setting adds `2 × I(X₀ > 0 and X₁ > 0)` to that additive function. X₂ and X₃ are irrelevant. The four covariate innovations, treatment noise and outcome noise are mutually independent standard normal streams. Each covariate is mapped through the normal CDF to a uniform distribution on [−2, 2]. Serial datasets first pass each of the six Gaussian streams through a stationary AR(1) process with coefficient 0.6 and innovation multiplier 0.8. The initial state has its stationary distribution. Treatment and outcome disturbances therefore have serial dependence but remain independent of one another and of the covariates. The same innovations are reused across the six scenarios for each replicate. There are 500 distinct independent seed replicates per cell. No unmeasured confounder, effect heterogeneity, treatment feedback or regime change is present.

The three main alternatives estimate one slope. Joint OLS adjusts linearly for all four covariates. Joint spline OLS controls for an additive cubic B-spline basis with five equally spaced knots per covariate on its declared support. The tree DML estimator uses distinct histogram gradient-boosted regressions for the outcome and treatment, followed by the repository's residual-on-residual slope. Its settings are fixed at 48 iterations, 12 leaves, minimum leaf size 12, 32 bins, learning rate 0.12 and L2 regularization 1. Two further comparisons apply exactly the same cross-fitting and orthogonal score to ordinary linear and spline nuisances. They distinguish changes in the nuisance function class from changes in the fold design. No hyperparameters are selected using measured effects. [Driver](simulation.py)

Independent data use three shuffled folds and all 1,200 rows contribute to the final estimating equation. Serial data use three expanding-window folds, with 295, 595 and 895 prior training observations and a five-observation gap before each test block. The same 900 rows enter every estimator's final estimating equation. The two classical regressions fit jointly on those 900 rows. Cross-fitted methods additionally use the initial prefix for nuisance training. The cross-fitted linear and spline comparisons hold both the training opportunity and scoring dates fixed. There is no forward forecast of an effect in this simulation. Evaluation in each cell asks how often a fitted slope and interval recover the one generating effect.

Every method uses the residual-treatment influence score. The independent designs use its sample standard deviation divided by the square root of the scored sample size. Serial designs use the same Bartlett HAC calculation on 900 consecutive scores with automatic bandwidth 9. Intervals use a normal 95% critical value. Within each dataset all methods use the same interval convention.

## Measured accuracy and coverage

Each table cell reports **RMSE / coverage percentage**. All 15,000 scheduled fits returned finite estimates and positive standard errors. For example, additive serial spline OLS and tree DML had RMSEs of 0.0488 and 0.0497. Their paired difference of 0.0009 had a Monte Carlo interval from −0.0012 to 0.0030. A single ranking on that pair would overstate the evidence.

| Confounding and sampling | Joint linear OLS | Joint spline OLS | Cross-fitted linear | Cross-fitted spline | Tree DML |
|---|---:|---:|---:|---:|---:|
| Linear, independent | 0.0307 / 94.4 | 0.0308 / 93.8 | 0.0309 / 93.6 | 0.0318 / 92.6 | 0.0357 / 88.8 |
| Linear, serial | 0.0487 / 93.2 | 0.0490 / 94.0 | 0.0500 / 93.0 | 0.0507 / 92.2 | 0.0515 / 91.4 |
| Additive, independent | 0.4789 / 0.0 | 0.0308 / 93.8 | 0.4791 / 0.0 | 0.0318 / 92.8 | 0.0350 / 89.0 |
| Additive, serial | 0.4799 / 0.0 | 0.0488 / 93.8 | 0.4782 / 0.0 | 0.0506 / 92.2 | 0.0497 / 91.0 |
| Interaction, independent | 0.6053 / 0.0 | 0.2387 / 0.0 | 0.6053 / 0.0 | 0.2384 / 0.0 | 0.0453 / 80.6 |
| Interaction, serial | 0.6063 / 0.0 | 0.2421 / 0.0 | 0.6036 / 0.0 | 0.2397 / 0.0 | 0.0602 / 85.6 |

The following differences are tree RMSE minus conventional spline RMSE. Negative values indicate lower error for the tree estimator. Bias is the mean signed error of tree DML and is expressed in slope units.

| Confounding and sampling | Tree bias | Paired RMSE difference | Pointwise 95% Monte Carlo interval |
|---|---:|---:|---:|
| Linear, independent | −0.0114 | +0.0049 | [+0.0033, +0.0065] |
| Linear, serial | −0.0153 | +0.0025 | [+0.0003, +0.0047] |
| Additive, independent | +0.0011 | +0.0041 | [+0.0028, +0.0055] |
| Additive, serial | +0.0055 | +0.0009 | [−0.0012, +0.0030] |
| Interaction, independent | +0.0293 | −0.1934 | [−0.1956, −0.1911] |
| Interaction, serial | +0.0330 | −0.1819 | [−0.1854, −0.1783] |

For example, in the additive designs the linear model remains biased by about +0.48 even when cross-fitted. In the interaction designs the additive spline remains biased by about +0.24 even when cross-fitted. Here the conditional means omitted by those model families are correlated. Cross-fitting changes which records train them and cannot restore the omitted function. Trees reduce that bias but still have a positive error in the interaction settings. A smaller point-estimation error does not guarantee nominal interval coverage.

The raw results contain all estimates, standard errors, elapsed times and failure reasons. The summary also gives mean bias, empirical standard deviation, mean estimated SE, RMSE, Monte Carlo SEs, Wilson intervals for coverage, and failure counts. Paired results compare tree DML with each of the four alternatives by bias, squared error, RMSE and coverage. Each reported Monte Carlo SE uses independent replications within a cell. For RMSE, the paired delta method uses the covariance of squared errors on the same draws. These pointwise intervals quantify uncertainty from using 500 simulated datasets. The inferential intervals inside each dataset evaluate the unknown treatment effect. [Draws](simulation-draws.csv.gz) · [Method summaries](simulation-summary.csv) · [All paired differences and Monte Carlo intervals](simulation-paired.csv)

## How much of the tree advantage was recognizing the interaction?

After inspecting the primary results, we fit an additional spline regression on the same 500 datasets in each interaction cell. We supplied the exact joint threshold indicator from the generator. This uses information a practitioner would need to justify or discover independently. It isolates what happens when a conventional regression can represent that interaction. All 1,000 supplementary fits succeeded.

| Sampling | Informed spline RMSE | Informed spline coverage | Tree minus informed spline RMSE, paired MC interval |
|---|---:|---:|---:|
| Independent | 0.0309 | 93.8% | +0.0145 [+0.0121, +0.0168] |
| Serial | 0.0488 | 93.8% | +0.0114 [+0.0083, +0.0145] |

The primary comparison gives a reason to try flexible adjustment when interactions are plausible and their form is unknown. The diagnostic also shows why a favorable comparison against an additive model alone cannot establish a general advantage for DML. Recovering the effect after supplying the right term is evidence about model specification. The simulation does not identify a way to discover that term from a new observational study or verify that all confounders were observed. [Supplementary draws](simulation-oracle-draws.csv.gz) · [Method summaries](simulation-oracle-summary.csv) · [Paired differences](simulation-oracle-paired.csv)

## Cost and reproduction

Replaying with the separate temporalcv 2.3.0 checkout reproduced all 16,000 archived estimates and standard errors and every numerical comparison from the locked 2.2.0 environment at the stored precision. The three regression tests passed in both environments. They check fold chronology, the generating effect, exact agreement with the public cross-sectional DML entry point, equivalence to joint OLS, and paired failure accounting. [Simulation tests](../../../../test/validation/test_audit_dml_simulation.py)

The primary run took 20.4 seconds with 24 worker processes and one numerical-library thread per worker. Across cells, median elapsed time per complete tree fit ranged from 0.141 to 0.148 seconds. Joint spline fits took about 0.0016–0.0017 seconds and simple linear OLS 0.0002–0.0003 seconds. Timing includes fit and inference under concurrent execution and will vary by machine. Draw order and seeds are independent of worker allocation. Set an output directory distinct from the archived report to replay the results. [Primary run settings](simulation-run.json) · [Supplementary run settings](simulation-oracle-run.json)

```bash
audit=docs/audits/2026-11-19_time_series_dml/adversarial
out=/tmp/dml-method-simulation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python "$audit/simulation.py" \
  --workers 24 --repetitions 500 --output "$out"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python "$audit/simulation.py" \
  --workers 12 --repetitions 500 --output "$out" --interaction-oracle
uv run --locked python -m pytest -q test/validation/test_audit_dml_simulation.py
```

## Does excluding the scored row from tree training improve accuracy here?

This post hoc comparison holds the tree architecture, all three fitted pairs of nuisance models, and the evaluation rows fixed. It uses the 500 original independent-data replicates for each of the three confounding functions. Within a dataset, the original procedure fits on 800 rows and predicts the 400 excluded rows. For the comparison, each of those **same fitted models** also predicts the next 400-row fold in cyclic order. Every row in that fold participated in training. Cycling the three folds scores every observation exactly once under each method. A third fit uses all 1,200 observations to train and score the trees, as in a conventional in-sample Robinson residualization. The larger training sample changes a second factor in that comparison.

All three variants use the same residual-on-residual slope, influence-score standard error and nominal 95% interval. Every regenerated held-out estimate and standard error matched the archived primary fit at its stored precision. All 4,500 attempts succeeded. The in-sample intervals are included to measure their empirical coverage in these simulations. Their use with a flexible learner requires separate theoretical justification.

| Confounding | Held-out tree RMSE / coverage | Cycled in-sample RMSE / coverage | Full in-sample RMSE / coverage | Held-out minus cycled RMSE, paired 95% MC interval |
|---|---:|---:|---:|---:|
| Linear | 0.0357 / 88.8% | 0.0365 / 88.6% | 0.0337 / 89.8% | −0.0008 [−0.0025, +0.0008] |
| Additive | 0.0350 / 89.0% | 0.0332 / 91.8% | 0.0333 / 91.8% | +0.0018 [+0.0002, +0.0034] |
| Interaction | 0.0453 / 80.6% | 0.0425 / 82.4% | 0.0431 / 82.4% | +0.0029 [+0.0011, +0.0047] |

The paired differences in coverage, held-out minus cycled in percentage points, were **+0.2 [−2.5, +2.9]** for linear confounding, **−2.8 [−5.4, −0.2]** for additive confounding and **−1.8 [−5.2, +1.6]** for the interaction. These pointwise Monte Carlo intervals allow for the paired data and model fits. Pooling the same trees' predictions on training rows preserved the large gain over additive splines in the interaction design. Excluding each scored row from training had no detectable RMSE benefit in the linear case and modestly higher RMSE in the other two. This comparison concerns one regularized learner, three specified data-generating processes, and one sample size. All three procedures use a residualized score, so it cannot isolate the effect of orthogonalization itself. [Driver](simulation_crossfit.py) · [All ablation draws](simulation-crossfit-draws.csv.gz) · [Method summaries](simulation-crossfit-summary.csv) · [Paired RMSE, bias and coverage uncertainty](simulation-crossfit-paired.csv) · [Run settings](simulation-crossfit-run.json)

The following command, run from the repository root, compares each recreated held-out fit against the primary archive and writes the results separately. The fourth test in the combined simulation contract suite checks both fold memberships, complete coverage of the scored rows, the full-sample estimate and agreement with the original archive. [Combined tests](../../../../test/validation/test_audit_dml_simulation.py)

```bash
audit=docs/audits/2026-11-19_time_series_dml/adversarial
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python "$audit/simulation_crossfit.py" \
  --workers 24 --repetitions 500 --output /tmp/dml-method-crossfit
```
