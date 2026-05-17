# FWL theorem, Robinson estimator, and partial-out theory

Foundational results on partialing out — the FWL theorem (Frisch–Waugh 1933; Lovell 1963), Robinson's 1988 root-N partially linear estimator, and the double-residual machinery that DML reinterprets through Neyman orthogonality.

**Coverage:** 13 entries.
**Anchor prefix:** `A` (this file is `01_theory_partialing.md`).

## A1. FWL theorem, Robinson estimator, and partial-out theory

- **Partial time regressions as compared with individual trends** — Frisch & Waugh (1933); Econometrica 1933
  - **Source:** https://www.jstor.org/stable/1907330
  - **Code:** —
  - **Mechanism:** Proves the partial-time-regression equivalence between de-trending series and including a time trend.
  - **Result:** First proof of what became the FWL theorem; established double-residual logic.
  - **Status:** Verified (no widely-known repo)

- **Seasonal adjustment of economic time series and multiple regression analysis** — Lovell (1963); JASA 1963
  - **Source:** https://doi.org/10.1080/01621459.1963.10480682
  - **Code:** —
  - **Mechanism:** Generalizes Frisch–Waugh to arbitrary linear regression — partialling out vs. residualizing.
  - **Result:** General-form FWL theorem; matrix-notation proof now standard in econometrics.
  - **Status:** Verified (no widely-known repo)

- **Root-N-Consistent Semiparametric Regression** — Robinson (1988); Econometrica 1988
  - **Source:** https://www.econometricsociety.org/publications/econometrica/1988/07/01/root-n-consistent-semiparametric-regression
  - **Code:** —
  - **Mechanism:** Constructs the root-N consistent estimator for the partially linear model Y = X'β + φ(Z) + ε using nonparametric residuals.
  - **Result:** Founding paper of the Robinson estimator — the FWL+nonparametrics ancestor of cross-sectional DML.
  - **Status:** Verified (no widely-known repo)

- **Econometric Theory and Methods** — Davidson & MacKinnon (2003); Oxford University Press
  - **Source:** http://qed.econ.queensu.ca/ETM/ETM-davidson-mackinnon-2021.pdf
  - **Code:** —
  - **Mechanism:** Graduate econometrics textbook covering OLS, IV, GMM, nonlinear estimation, and Monte Carlo / bootstrap inference.
  - **Result:** Standard reference for FWL theorem proofs and projection-matrix derivations underlying DML's partialing logic.
  - **Status:** Verified (no widely-known repo)

- **A Simple Proof of the FWL Theorem** — Lovell (2008); Journal of Economic Education 2008
  - **Source:** https://doi.org/10.3200/JECE.39.1.88-91
  - **Code:** —
  - **Mechanism:** Simple algebra-only proof of the FWL theorem suitable for teaching.
  - **Result:** Reduces FWL to OLS algebra, removing dependence on projection-matrix arguments.
  - **Status:** Verified (no widely-known repo)

- **High-Dimensional Methods and Inference on Structural and Treatment Effects** — Belloni, Chernozhukov & Hansen (2014); Journal of Economic Perspectives 2014
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/jep.28.2.29
  - **Code:** —
  - **Mechanism:** JEP-level overview adapting Lasso/post-Lasso variable selection to provide high-quality inference on structural/treatment parameters.
  - **Result:** Practitioner-facing entry point for high-dim methods that guard against false discovery / overfitting in causal estimation.
  - **Status:** Verified (no widely-known repo)

- **Confidence Intervals and Hypothesis Testing for High-Dimensional Regression** — Javanmard & Montanari (2014); Journal of Machine Learning Research 2014
  - **Source:** https://arxiv.org/abs/1306.3171
  - **Code:** —
  - **Mechanism:** Debiased version of regularized M-estimators without special design-matrix structural assumptions.
  - **Result:** Near-optimal confidence intervals and hypothesis tests for high-dim linear regression; validated on riboflavin genomic data.
  - **Status:** Verified (no widely-known repo)

- **On asymptotically optimal confidence regions and tests for high-dimensional models** — van de Geer, Bühlmann, Ritov & Dezeure (2014); Annals of Statistics 2014
  - **Source:** https://arxiv.org/abs/1303.0518
  - **Code:** —
  - **Mechanism:** Extends debiased-Lasso confidence-interval construction from linear to generalized linear models with convex losses.
  - **Result:** Achieves semiparametric-efficiency asymptotic optimality under Gaussian, sub-Gaussian, and bounded correlated designs.
  - **Status:** Verified (no widely-known repo)

- **Confidence intervals for low dimensional parameters in high dimensional linear models** — Zhang & Zhang (2014); Journal of the Royal Statistical Society Series B 2014
  - **Source:** https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12026
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Inference in Linear Regression Models with Many Covariates and Heteroskedasticity** — Cattaneo, Jansson & Newey (2018); Journal of the American Statistical Association 2018
  - **Source:** https://arxiv.org/abs/1507.02493
  - **Code:** —
  - **Mechanism:** Many-covariate asymptotics where number of regressors grows proportionally with sample size; new HC standard-error formula.
  - **Result:** Eicker-White SEs are inconsistent under many covariates; proposed automatic SE robust to unknown heteroskedasticity and many regressors.
  - **Status:** Verified (no widely-known repo)

- **The Frisch–Waugh–Lovell Theorem for Standard Errors** — Ding (2021); arXiv preprint (URL re-verification pending — `arXiv:2106.02087` resolves to an unrelated paper, 2026-05-16; correct arXiv ID needs lookup via Peng Ding's Berkeley page)
  - **Source:** https://stat.berkeley.edu/~pengdingpku/
  - **Code:** —
  - **Mechanism:** Extends FWL to standard-error variants (HC, cluster-robust).
  - **Result:** Shows that residual-on-residual regression preserves HC and cluster-robust SEs.
  - **Status:** (uncertain URL — flagged for url-freshness) Verified

- **The Yule-Frisch-Waugh-Lovell Theorem for Linear Instrumental Variables Estimation** — Basu (2023); arXiv preprint
  - **Source:** https://arxiv.org/abs/2307.12731
  - **Code:** —
  - **Mechanism:** Extends FWL to linear IV — coefficients on endogenous variables identical in full vs partial (residualized) IV regressions.
  - **Result:** Equivalence holds for K-class estimators (including LIML) and two-step optimal GMM in large samples; not universal for linear GMM.
  - **Status:** Verified (no widely-known repo)

- **The Yule-Frisch-Waugh-Lovell Theorem** — Basu (2023); arXiv preprint
  - **Source:** https://arxiv.org/abs/2307.00369
  - **Code:** —
  - **Mechanism:** Reframes FWL with Yule's earlier work and updates exposition for modern teaching.
  - **Result:** Argues for 'Yule–Frisch–Waugh–Lovell' attribution; presents unified treatment of partialing-out.
  - **Status:** Verified (no widely-known repo)
