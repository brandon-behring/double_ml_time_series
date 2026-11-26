# The Chapter 2 example reverses its original learner ranking

The original editions said that the forest recovered a true slope of 2.5 and that OLS gave 2.134. The displayed OLS code actually controlled for all twenty covariates. On its seeded dataset it gave **2.501**. The forest gave **2.407**, with a marginal ATE interval **[2.343, 2.471]** that excluded 2.5. Both teaching editions now show the measurements. [Captured output](additional/chapter2.json) · [Executable cross-edition check](additional/chapter2.py)

| Nuisance learner for both roles | Effect | 95% marginal interval | Absolute error on this dataset |
|---|---:|---:|---:|
| Lasso | 2.499 | [2.462, 2.535] | 0.001 |
| Ridge | 2.498 | [2.464, 2.532] | 0.002 |
| Random forest | 2.407 | [2.343, 2.471] | 0.093 |
| Gradient boosting | 2.272 | [2.216, 2.328] | 0.228 |

The original recommendation favored boosting on the basis of a reported error of 0.001. Replaying the actual seeded generator, 2,000 rows, five folds and the stated learner settings reversed the ordering. A single draw establishes neither the bias of an estimator nor its interval coverage. The chapter now distinguishes single-draw absolute error from the average signed error and nominal coverage measured across independent datasets. EconML 0.16.0 reported that it handled the CV configuration of LassoCV and RidgeCV internally. The replay records the library versions. [Chapter 2 print source](../../../chapters/chapter_02.tex) · [Web source](../../../web/src/content/chapters/chapter02-neyman-orthogonality-dml.mdx)

The [separate audit of the Chapter 2 derivations and exercises](additional/chapter2-theory.md) corrects two score interpretations and compares conditional versus average effects.

The point estimate and interval now use EconML's marginal ATE interface on the same evaluation covariates. Previously the code averaged the lower and upper limits of individual conditional-effect intervals. Shared model coefficients induce covariance among predictions, which such an average omits. The error can persist even when point estimates agree. In the separate binary-treatment driver, nominal per-dataset ATE coverage in the five configurations was **57% to 92% for the classifier**, compared with five historical cells at 100% using the old endpoint construction. Switching between a probabilistic classifier and a regressor of a binary treatment also changes the predictions unless the two estimate the same conditional mean. [Paired binary-treatment study](standalone-reproduction.md) · [Conditional versus population uncertainty](heterogeneity-and-policy.md)

The manual fivefold forest diagnostic gave mean fold R² **0.611** for treatment and **0.577** for outcomes, with cross-fold SDs of **0.025** and **0.018**. Both clear the old 0.1 threshold by a large margin, while the forest's marginal interval misses in this draw. A useful probability predictor can also have population R² zero under fair random assignment. Chapters 2 and 3 now call for a design-specific predictive comparator, residual treatment variation, assignment-unit counts and a separate effect or interval evaluation. [Known-randomization control](software-and-monitoring.md) · [Crossed learner experiments](nuisance-prediction.md)

To replay, install the optional EconML and plotting dependencies and run from the repository root. The script extracts the three linked listings, checks that they match in both editions, executes them in a temporary directory, and writes their measured output.

```bash
audit=docs/audits/2026-11-21_evaluation_methods
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python "$audit/additional/chapter2.py" /tmp/chapter2-evaluation.json
diff "$audit/additional/chapter2.json" /tmp/chapter2-evaluation.json
```
