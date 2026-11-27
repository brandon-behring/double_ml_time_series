# Chapter 2 theory and executable examples

For the structural partially linear model, both the IV-type and partialling-out scores are orthogonal at their respective true nuisance functions. The previous chapter incorrectly called the IV-type score nonorthogonal by perturbing one nuisance away from truth while ignoring the evaluation point for the other. Both editions now derive both directions and contrast the IV-type moment with the genuinely first-order-sensitive moment that uses unresidualized treatment. A local bias of order $n^{-1/2}$ can shift limiting confidence-interval coverage even when the slope remains root-$n$ consistent. The stated product-rate condition now uses the actual population drift. The sampling rate of the centered empirical process is distinct. [Print chapter](../../../../chapters/chapter_02.tex) · [Web chapter](../../../../web/src/content/chapters/chapter02-neyman-orthogonality-dml.mdx) · [DoubleML score definitions](https://docs.doubleml.org/stable/guide/scores.html)

The revised algorithm distinguishes the pooled residual slope implemented by the package from the equal-weight average of fold slopes. The randomized exercise distinguishes a realized Horvitz–Thompson signal from its expectation over treatment assignment. The sensitivity section derives the denominator of omitted-variable bias from residual treatment variation. In the example the population shift is **0.320**, since residual treatment is $0.5U+v$ with independent unit-variance disturbances. It also explains the equal-strength interpretation of an ordinary linear-regression robustness value. [Original PLR framework](https://doi.org/10.1111/ectj.12097) · [Authors’ robustness-value documentation](https://carloscinelli.com/sensemakr/reference/robustness_value.html)

| Frozen example | What the replay measured |
|---|---|
| Age-specific one-unit effects at income 50,000 | Points 2.047, 1.198 and 0.349. All three pointwise intervals missed the respective truths of 2.6, 1.4 and 0.2 on this one training draw. |
| Omitted confounder, scalar fits | Omitting $U$ gave 2.242, including it gave 1.951. The unrounded difference was +0.290, or 14.5% of the causal effect of 2. This is the change between two fits on one dataset. |
| Same 50-tree specification and through-origin residual score | Five-fold estimate 3.174 and in-sample estimate 3.177 against a truth of three. The small difference on this path supplies no repeated-sampling bias comparison. |
| Nuisance families on common folds | Absolute errors were 0.619 for ridge, 0.060 for a depth-two tree, 1.123 for a depth-ten tree, and 0.180 for a 100-tree forest. The shallow tree gave the smallest error in this realization. |

The five runnable listings in each edition, including the shared exercise setup, were executed in isolated namespaces as appropriate. Every displayed output matched the corresponding measured text, and the two editions agreed. [Full numerical receipt](chapter2-theory.json) · [Replay implementation](chapter2_theory.py)

```bash
audit=docs/audits/2026-11-21_evaluation_methods/additional
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python "$audit/chapter2_theory.py" --output /tmp/chapter2-theory.json
```
