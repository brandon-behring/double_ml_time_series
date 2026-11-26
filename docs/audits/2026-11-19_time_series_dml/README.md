# Time-series double machine learning — audit and empirical replications

## Classical benchmarks and time-forward validation

- **Implementation accuracy.** The repo reproduced randomized job-training earnings, the published Grunfeld panel slopes and the fiscal IV standard errors. On identical folds its adjusted earnings estimate agreed with DoubleML to numerical precision. The impulse-response experiment shows how recovery changes with horizon and sample size. [Classical econometrics](classical/README.md)
- **Causal accuracy under dependence.** Across 4,000 simulated paths, flexible adjustment substantially improved estimates when confounding involved an interaction. Nominal 95% intervals from buffered forward tree fits covered only 80.0% in the 1,200-date stable serial design. [Matched splits, uncertainty and rolling effects](paired/README.md)
- **Field evidence.** The Bangalore congestion-pricing replay recovered all four published targets. In London the final quarter provided no matched high- and low-price event start times. Together the studies show how later effects lose precision or become unavailable as independent assignments run out. [Economic field experiments](field/README.md)

Run all five entry points from the repository root with `uv run --locked python docs/audits/2026-11-19_time_series_dml/external/run.py --suite method-validation --jobs 32`. Downloads are cached outside the repository. At this budget the runner assigns 24 workers to the matched simulation, four to the meter archive, two to the impulse responses and one each to the two other studies. Numerical libraries use one thread per worker.

All five study entry points passed. All 18 focused regression tests passed with both the locked temporalcv 2.2.0 and the adjacent 2.3.0 checkout. The broader offline run passed 1,049 tests and skipped 41, with 70 network-marked tests excluded. [Simulation and suite checks](../../../test/validation/test_matched_temporal_experiments.py) · [Fiscal checks](../../../test/validation/test_classical_benchmarks.py) · [Bangalore checks](../../../test/validation/test_bangalore_field_replay.py) · [London checks](../../../test/validation/test_london_field_benchmark.py)

## Other November findings

**November 21 follow-up.** The adversarial review compares DML with ordinary and spline regression, randomized contrasts and local projections. It includes 3,000 paired simulated datasets, common-sample checks in three published studies, and a method recommendation for each application. [When does DML earn its place?](adversarial/README.md)

**November 19–20, 2026.** The book has a reproducible path from residualization to scalar temporal PLR, and its Chapter 6 fixed-effects simulation reproduces. The audit found that Chapter 3's bias test rejected a true zero bias 192 times in 200 experiments at the nominal 5% level. After the November 20 correction, the validator rejected in 6 of the same 200 experiments. The follow-up traces the effect on two archived validation studies. [Correction and historical verdicts](08-bias-test-correction.md)

The [rolling causal-effect benchmark](rolling/README.md) tests forecasts on fresh reminder cohorts, measures power and false drift alarms, reproduces four public comparisons, and preserves both the original JTPA holdout forecasts and a correction to the earnings decoder.

The prospective follow-up froze four forecasts before future assignments in international cricket and the planned 2028 General Social Survey. The report gives the numerical forecasts, the precision each experiment would need to tell them apart, and the results of applying the same comparison to earlier reminder and job-training data. [Prospective report and forecast records](prospective/README.md)

## Manuscript revisions, in order

| Priority | Evidence from this audit | Revision |
|---|---|---|
| Completed November 20. Chapter 3 validation verdicts | The original calculation divided the spread of bootstrap means by the square root of the bootstrap count a second time. | All seven validators now test independent simulation errors. Rescoring the 42-row comparison yields 41 PASS and one WARNING. [Original diagnosis](02-code-and-methodology.md) · [Results by row](08-bias-test-correction.md) |
| Completed November 20. 401(k) replication claims | The fivefold partially linear author-code fits gave **$9,243.65** for forest and **$8,182.27** for rigorous lasso. The paper reports **$9,247** and **$8,187**. | Chapter 3 now distinguishes these estimates from the single-partition Python fits and from the participation effect estimated with an instrument. The Python API uses the verified references and mean-effect intervals. [Published and refitted results](07-401k-replication.md) |
| Completed November 20. Chapter 5 interval coverage | On the chapter's process, coverage was **89.5% at lag zero** and **88.5% with automatic HAC** for the same forward fit. With persistent scores, HAC raised coverage from **60.5% to 89.0%**. | Both editions report the five measured rows, explain how score autocovariance affects the standard error, and give Monte Carlo uncertainty for the paired changes. [Simulation results](04-reproducibility.md#chapter-5--matched-score-reconstruction) · [Results by replication](chapter5-draws.csv) |
| Completed November 20. Panel, retail and insurance inference | The corrected panel fit gives identical estimates and SEs under input reordering. The retail fit gives a conditional slope of **−1.1673** and a store/week-clustered SE of **0.1171**. | Both editions report the corrected panel designs, retail slopes and simulated insurance results. All 800 Chapter 8 panel fits completed. [Results and reproduction](09-panel-retail-insurance.md) |

All **600 joint 401(k) refits** finished. Across 42 published cells the median absolute point-estimate difference was **$42**. The largest difference was **$672** for the two-fold interactive “Best” estimate. In all 200 interactive partitions, the public R code returned exactly the lasso estimate and standard error for “Best.” The published columns differ. [Completed comparison and explanation](07-401k-replication.md)

The follow-up checks the simulation controls, causal scores, numerical intervals, parallel result storage and the published table values. It includes 500 paired replications in each of four stress scenarios. [Property tests and falsification results](10-property-tests-and-falsification.md)

The external benchmark suite reproduces ten more published studies, including all 225 numerical cells behind an Italian time-series DML figure. Two randomized trials supply small offline checks of coefficients, units and uncertainty. [External replication results and commands](11-external-replication-benchmarks.md)

Five follow-up reanalyses revisit promotion timing, electricity demand, restaurant employment, day-care fines and behavioral reminders. The guide also screens three additional causal designs. [Historical causal questions and measured results](12-historical-causal-questions.md)

The calendar-validation study tests a frozen causal effect against later randomized participants, replays GDP forecasts with historical data vintages, and screens the 2018 journal roster. [Forward-validation results, data access and commands](13-forward-causal-validation.md)

## Four further tests with public research data

| Empirical question | Measured result | What can be validated |
|---|---|---|
| Do earlier retail recommendations predict later click lift? | **+0.1638** percentage points predicted versus **+0.1401** measured. In a separate same-period test the uniform policy produced **0.5417%** clicks, compared with **0.5284%** from weighting and **0.3609%** from doubly robust estimation. | Randomized future traffic tests the frozen effect. The same-week counterfactual estimates expose the influence of rare recommendations. [Retail](forward/retail-recommendations.md) |
| Did US farm wages rise after bracero exclusion? | A ten-point difference in exposure predicted **−0.36 cents/hour** in relative wage change [−1.21, +0.50]. Two of three pre-policy placebo dates had nominal p-values below .05. The smallest multiplicity-adjusted p-value was **.0615**. | Pre-policy wage movements challenge the comparison across states. The panel contains one national termination. [Bracero wages](forward/bracero.md) |
| Did the Indonesian teacher-pay effect on learning persist? | Test-score effects were **−0.0055 SD** in 2011 and **+0.0104 SD** in 2012. The early-minus-late difference was **−0.0158 [−0.0448, +0.0132] SD**. | Repeated follow-ups of the same randomized schools measure persistence. Joint school-level uncertainty accounts for their overlap. [Teacher pay](forward/teacher-pay.md) |
| Did assistance bring households into Michigan’s weatherization program? | Among 28,888 households, offers raised applications **12.58 points**, audits **5.27** and completed retrofits **4.93**. Mean projected discounted bill savings were **$10,611** in the public engineering records. | The public data reproduce take-up and projected savings. The restricted monthly bills are needed to reestimate realized energy savings. [Weatherization](forward/weatherization.md) |

Run all four from the repository root with `uv run --locked python docs/audits/2026-11-19_time_series_dml/external/run.py retail-recommendations bracero-wages teacher-pay weatherization --jobs 4`. Allcott's Opower study provides a longer-horizon published comparison across 111 energy trials. [Opower data access and findings](05-literature.md#can-successful-field-experiments-overpredict-savings-when-a-program-expands)

## Research priorities

1. **Teach from a known answer.** In the bike-share experiment, 1,600 fits on a real daily history covered the planted effect only **64.5%–72.0%** of the time, depending on the gap and assignment persistence. The chosen adjustment can transmit calendar extrapolation error into the treatment estimate. Pair this example with Chapter 5 to separate point-estimation bias from interval calibration. [Bike-share pilot](pilots/bike.md)
2. **Use a randomized benchmark for applied microeconomics.** In Trondheim, the first-event arm difference was **−0.29 kWh** per household over peak-price hours, with 95% interval **[−3.91, 3.32]**. Pooling 21 events with baseline adjustment gave **−1.08 [−2.22, 0.06] kWh**. The intervention was an announced reward program, and the dates were chosen by organizers. [Electricity pilot](pilots/iflex.md)
3. **Teach how fold selection can change the target population.** The public HeartSteps release reproduced **6,061 eligible decisions**, the published coefficient **0.1312 log steps**, and participant-robust SE **0.0674**. Forward scoring excluded 960 early eligible decisions while the intervention effect was changing. [HeartSteps pilot](pilots/heartsteps.md)
4. **Show a case where identification limits the estimator.** The California synthetic DiD benchmark reproduced at **−15.60 packs per person per year**. Temporal PLR gave **−34.50**, including an adoption-spanning fold trained exclusively on untreated years. A small reported SE does not establish support for the policy counterfactual. [California pilot](pilots/california.md)

The literature review points to overlooked work on repeated randomized treatment paths and switchbacks, and to already-catalogued research on dependent-data DML that deserves direct comparison in the text. Reviewing all 163 paper-ledger records in two passes found four links to different works, including the fixed-bandwidth HAC reference, and two broken links with verified replacements. The public-data shortlist adds market-price and spillover exercises, including a short fish-demand series whose final test block contains no stormy days. [Source checks](literature-source-checks.md) · [Literature review](05-literature.md) · [Public-data research agenda](06-public-datasets.md)

## Software readiness and evidence

The initial audit checkout passed **all 880 tests collected at that point**. The 695 tier 1/2 tests covered **75.48%** of the package. All eight examples ran, and the book, Sphinx documentation and ten-chapter web edition built. The November 20 revision added regression checks for unbalanced fixed effects, date and unit folds, unscored rows, and treatment-unit changes. The follow-up synchronized and verified the lockfile. Incremental PDF dependencies remain open. [Code audit](02-code-and-methodology.md) · [Verification commands and results](04-reproducibility.md)

The snapshot covers all **307 tracked paths**, compares the dirty checkout to committed main and identifies the six pre-existing local modifications. The May audit and June status page remain evidence for their original dates. This collection supplies the later measurements and research findings. [Inventory and history](01-inventory-and-history.md)
