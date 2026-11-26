# Chapter 3 bias tests and historical verdicts

**November 20, 2026.** The corrected test rejected a true zero bias in **6 of 200** experiments at the nominal 5% level. The previous test rejected in **192 of the same 200**. Rescoring the archived seven-method comparison changes **all 37 of its FAIL verdicts**. The measured biases, errors and coverage proportions come from the original fits. [Row-level results](bias-archive-rescored.csv)

## What counts as an independent observation

For each fresh simulated dataset, record one signed error, the estimate minus the known effect. With $R$ independent errors and their sample standard deviation $s$, the Monte Carlo standard error of estimated mean bias is $s/\sqrt R$. The two-sided test uses $R-1$ degrees of freedom. The DML validator and all six comparison methods now use this calculation. Bootstrap means still supply percentile limits for a bias interval. Changing the number of bootstrap resamples changes the numerical accuracy of those limits. It supplies no additional independent datasets. [Inference helper](../../../dml_ts/validation/_simulation_inference.py) · [DML validator](../../../dml_ts/validation/bias_validation.py)

The DML verdict also uses an exact binomial test against the configured per-draw coverage level and applies a Bonferroni correction by default. It reports separate bias and coverage verdicts. A significant coverage deficit gives FAIL and a significant excess gives WARNING. The practical thresholds apply to bias. The overall verdict uses the worse component. The six baselines classify the bias p-value alone. For the two ML baselines, the result records how many draws contributed to the bias test when a simulated dataset lacked a treatment arm. [Regression tests](../../../test/validation/test_baseline_comparison.py) · [DML status tests](../../../test/validation/test_bias_validation.py)

Each null experiment used 100 independent standard-normal draws. The live package agreed with SciPy's one-sample test on all 200 p-values. Its observed rejection rate was 3.0%, with a 95% Wilson interval of 1.4% to 6.4%. The interval describes Monte Carlo uncertainty in this experiment. [Seeded null check](method_checks.py)

## November 2025 method comparison

The archive contains 20 runs for each of seven methods at six settings. Given the saved signed mean error $b$ and mean squared error $M$, one can recover the Monte Carlo standard error exactly to stored precision as $\sqrt{(M-b^2)/(R-1)}$. This permits a same-run comparison after replacing the bias p-value. As a cross-check, replaying the historical generator and refitting the 12 OLS rows reproduced their bias, MSE and recalculated p-values to within $2\times10^{-13}$. All 840 generated datasets contained both treatment arms. The smallest arm had 218 observations. [Original comparison](../../../results/method_comparison/method_comparison_detailed.csv) · [Run metadata](../../../results/method_comparison/findings.json) · [Historical generator and method order](https://github.com/brandon-behring/double_ml_time_series/tree/22a4b0b/src/validation)

| Archived status | Rescored PASS | Rescored WARNING | Rescored FAIL |
|---|---:|---:|---:|
| PASS | 5 | 0 | 0 |
| WARNING | 0 | 0 | 0 |
| FAIL | 36 | 1 | 0 |

The one WARNING is augmented IPW with 500 observations and confounding strength 0.5. Its mean error was −0.05040, its Monte Carlo SE was 0.02150, and the two-sided p-value was 0.03008. For all six DML rows, 20 of 20 archived intervals covered the truth (exact coverage p-value 0.623), each absolute bias was below 0.1, and each bias p-value exceeded 0.05. The original combined rule and the current practical gates therefore agree on these six rescored labels. A baseline can pass this bias test even with severe undercoverage. At 1,000 observations and confounding strength 2, naive OLS passed and its intervals covered in only 2 of 20 runs. [Rescoring script](rescore_bias_archive.py) · [Row-level p-values](bias-archive-rescored.csv)

## Earlier 81-setting DML sweep

The November 14 morning sweep used 100 simulations per setting. Its labels came directly from absolute bias cutoffs of 0.1 and 0.2. Those cutoffs reproduce all **43 PASS, 25 WARNING and 13 FAIL** entries. The faulty p-value calculation came later. A retrospective two-sided bias test gives $p<0.05$ for 53 of the 81 settings, considered individually. [Archived sweep](../../../results/simulations/BIAS_VALIDATION_results_20251114_103701.csv) · [Original scoring code](https://github.com/brandon-behring/double_ml_time_series/blob/88f99f2/src/validation/bias_validation.py)

This sweep constructed placeholder intervals around each fit using the standard error of the **Monte Carlo mean**. It saved aggregate rows. Its reported coverage of 9% to 30% cannot establish estimator interval coverage. To find what the saved bias statistics alone establish under today's rule, we evaluated every possible coverage count from 0 through 100 for each setting. **31 settings give FAIL for every possible count.** Of the other 50, 28 admit WARNING or FAIL and 22 admit PASS, WARNING or FAIL. Before the coverage verdict was separated from bias, 27 of those 50 always received WARNING under the earlier rule. The separate bias verdicts are 31 FAIL, 28 WARNING and 22 PASS. In the row-level CSV, the last column lists the possible current outcomes for each setting. These are retrospective implications of applying a later rule to the historical fits. [Sweep metadata](../../../results/simulations/BIAS_VALIDATION_metadata_20251114_103701.json) · [Row-level results](bias-archive-rescored.csv)

## Chapter 3 checks

In Experiment 3.1, the fixed-seed naive OLS run gives bias +0.13266 (Monte Carlo SE 0.12530, $p=0.2923$, PASS). The controlled OLS run gives +0.01798 (SE 0.00904, $p=0.04948$, WARNING). Their intervals covered in 15 and 69 of 100 runs, respectively. The controlled fit's covariance still omits the fitted intercept. A reference calculation including the intercept and its degrees of freedom covered in 91 of the same 100 runs. [Same-draw check](chapter3_baselines.py) · [Original diagnosis](02-code-and-methodology.md)

Experiment 3.6 now runs the two-sided one-sample test in the text. For an injected bias of 0.05 and draw-level SD of 0.10, it detected bias in **998 of 1,000 trials** at $R=100$. The exact normal-model power is 99.86%. For a bias of 0.01 at the same sample size and SD, exact power falls to 16.77%. A replication count is meaningful only together with the size of bias the experiment is designed to detect. [LaTeX chapter](../../../chapters/chapter_03.tex) · [Web chapter](../../../web/src/content/chapters/chapter03-validation-framework.mdx)

The listings for Experiments 3.1, 3.2 and 3.6 reproduced every displayed row in both editions. The optional seven-method grid was syntax-checked. The updated book built to 208 pages with nonfatal typesetting warnings, the web build indexed 15 pages, and the chapter source hashes matched. The package passed 707 tier 1/2 tests, all six real-fit multiple-testing checks and 50 doctests. Sphinx, Ruff and mypy passed. An independent check passed the archived moments through the corrected package, verifying all 123 reconstructed p-values and all 8,223 status scenarios.

Run the checks from the repository root in the project Python environment.

```sh
PYTHONPATH=. DML_N_JOBS=1 python docs/audits/2026-11-19_time_series_dml/method_checks.py
PYTHONPATH=. python docs/audits/2026-11-19_time_series_dml/chapter3_baselines.py
PYTHONPATH=. python docs/audits/2026-11-19_time_series_dml/rescore_bias_archive.py
```
