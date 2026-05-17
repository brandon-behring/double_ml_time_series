# Neyman orthogonality and semiparametric DML theory

Neyman-orthogonal scores, sample splitting / cross-fitting, locally robust semiparametric estimation, and influence-function adjustment for nonparametric first steps.

**Coverage:** 21 entries.
**Anchor prefix:** `B` (this file is `02_theory_orthogonality.md`).

## B1. Neyman orthogonality and semiparametric DML theory

- **Optimal asymptotic tests of composite statistical hypotheses** — Neyman (1959); Probability and Statistics (1959)
  - **Source:** https://www.jstor.org/stable/i285003
  - **Code:** —
  - **Mechanism:** Optimal tests of composite hypotheses under nuisance parameters.
  - **Result:** Source of 'Neyman orthogonality' nomenclature — locally most powerful test under nuisance variation.
  - **Status:** Verified (no widely-known repo)

- **Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies** — Rubin (1974); Journal of Educational Psychology, 66(5), 688-701
  - **Source:** https://psycnet.apa.org/doi/10.1037/h0037350
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Statistics and Causal Inference** — Holland (1986); Journal of the American Statistical Association, 81(396), 945-960
  - **Source:** https://www.tandfonline.com/doi/abs/10.1080/01621459.1986.10478354
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Semiparametric Efficiency Bounds** — Newey (1990); Journal of Applied Econometrics 1990
  - **Source:** https://doi.org/10.1002/jae.3950050202
  - **Code:** —
  - **Mechanism:** Computes the semiparametric efficiency bound for parametric components of semiparametric models.
  - **Result:** Benchmark against which DML's efficiency claims are measured.
  - **Status:** Verified (no widely-known repo)

- **The Asymptotic Variance of Semiparametric Estimators** — Newey (1994); Econometrica 1994
  - **Source:** https://www.jstor.org/stable/2951752
  - **Code:** —
  - **Mechanism:** Asymptotic variance theory for two-step semiparametric estimators.
  - **Result:** Establishes when the influence-function adjustment is needed for valid inference after a nonparametric first stage.
  - **Status:** Verified (no widely-known repo)

- **Estimation of Regression Coefficients When Some Regressors are not Always Observed** — Robins, Rotnitzky & Zhao (1994); Journal of the American Statistical Association 1994
  - **Source:** https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476818
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Efficient and Adaptive Estimation for Semiparametric Models** — Bickel, Klaassen, Ritov & Wellner (1998); Springer
  - **Source:** https://link.springer.com/book/9780387984735
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Asymptotic Statistics** — van der Vaart (2000); Cambridge University Press 2000
  - **Source:** https://doi.org/10.1017/CBO9780511802256
  - **Code:** —
  - **Mechanism:** Graduate-level asymptotic statistics textbook including semiparametric theory.
  - **Result:** Standard reference for influence-function / score-function machinery DML builds on.
  - **Status:** Verified (no widely-known repo)

- **Targeted Maximum Likelihood Learning** — van der Laan & Rubin (2006); The International Journal of Biostatistics 2006
  - **Source:** https://biostats.bepress.com/ucbbiostat/paper213/
  - **Code:** —
  - **Mechanism:** Iteratively updates density estimator along hardest parametric submodel with score equal to efficient influence curve.
  - **Result:** Targeted estimator solves efficient-influence-curve equation; algebraically equivalent to locally efficient estimating-function approach.
  - **Status:** Verified (no widely-known repo)

- **Higher order influence functions and minimax estimation of nonlinear functionals** — Robins, Li, Tchetgen & van der Vaart (2008); IMS Collections 2008
  - **Source:** https://arxiv.org/abs/0805.3040
  - **Code:** —
  - **Mechanism:** Higher-order influence functions (higher-order U-statistics) extend semiparametric theory beyond first-order scores.
  - **Result:** Rate-optimal estimation and non-root-n inference for biostatistical functionals; multi-robustness extends double robustness.
  - **Status:** Verified (no widely-known repo)

- **Double/Debiased/Neyman Machine Learning of Treatment Effects** — Chernozhukov et al. (2017); AER Papers & Proceedings 2017
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/aer.p20171038
  - **Code:** —
  - **Mechanism:** Short AER P&P version summarizing the DML method for treatment effects.
  - **Result:** Compresses the DML pitch for econometrics audience; demonstrates 401(k) application.
  - **Status:** Verified (no widely-known repo)

- **Double/Debiased Machine Learning for Treatment and Causal Parameters** — Chernozhukov et al. (2017); arXiv preprint
  - **Source:** https://arxiv.org/abs/1608.00060
  - **Code:** —
  - **Mechanism:** arXiv preprint of the 2018 Econometrics Journal paper.
  - **Result:** Original public version of the DML framework, predating journal publication.
  - **Status:** Verified (no widely-known repo)

- **Double/debiased machine learning for treatment and structural parameters** — Chernozhukov et al. (2018); The Econometrics Journal 2018
  - **Source:** https://academic.oup.com/ectj/article/21/1/C1/5056401
  - **Code:** —
  - **Mechanism:** Neyman-orthogonal scores + cross-fitting yield root-n inference on causal parameters with ML nuisance.
  - **Result:** Establishes the DML framework — proves orthogonal moments + sample splitting eliminate regularization bias.
  - **Status:** Verified (no widely-known repo)

- **Automatic Debiased Machine Learning of Causal and Structural Effects** — Chernozhukov, Newey & Singh (2022); Econometrica 2022
  - **Source:** https://arxiv.org/abs/1809.05224
  - **Code:** —
  - **Mechanism:** Lasso-based automatic debiasing using only the functional of interest, without explicit closed-form bias correction.
  - **Result:** Works for neural nets, random forests, Lasso, boosting; applied to ATEs from training data and demand elasticities from scanner data.
  - **Status:** Verified (no widely-known repo)

- **Locally Robust Semiparametric Estimation** — Chernozhukov et al. (2022); Econometrica 2022
  - **Source:** https://arxiv.org/abs/1608.00033
  - **Code:** —
  - **Mechanism:** Locally-robust / Neyman-orthogonal moment functions for GMM with ML first-step estimators.
  - **Result:** Generalizes DML beyond PLR/IRM to a broad class of GMM parameters; recipe for adding orthogonality-restoring adjustment terms.
  - **Status:** Verified (no widely-known repo)

- **Debiased machine learning of global and local parameters using regularized Riesz representers** — Chernozhukov, Newey & Singh (2022); The Econometrics Journal 2022
  - **Source:** https://arxiv.org/abs/1802.08667
  - **Code:** —
  - **Mechanism:** Neyman-orthogonal equation incorporating Riesz representer as additional nuisance; covers regular and non-regular linear functionals.
  - **Result:** Double sparsity robustness — either regression or representer approximation may be dense if the other is sparse; honest uniform confidence bands.
  - **Status:** Verified (no widely-known repo)

- **Demystifying Statistical Learning Based on Efficient Influence Functions** — Hines, Dukes, Diaz-Ordaz & Vansteelandt (2022); The American Statistician 2022
  - **Source:** https://arxiv.org/abs/2107.00681
  - **Code:** —
  - **Mechanism:** Pedagogical derivation of efficient influence functions for treatment-effect estimands under nonparametric models.
  - **Result:** Step-by-step recipe demystifying TMLE/DML construction; broad applicability across treatment and policy estimands.
  - **Status:** Verified (no widely-known repo)

- **The Influence Function of Semiparametric Estimators** — Ichimura & Newey (2022); Quantitative Economics 2022
  - **Source:** https://arxiv.org/abs/1508.01378
  - **Code:** —
  - **Mechanism:** Derives the influence function for semiparametric estimators with first-step nonparametric estimation.
  - **Result:** Provides the building block for orthogonal moment construction — the influence-function adjustment term.
  - **Status:** Verified (no widely-known repo)

- **Semiparametric doubly robust targeted double machine learning: a review** — Kennedy (2022); arXiv preprint
  - **Source:** https://arxiv.org/abs/2203.06469
  - **Code:** —
  - **Mechanism:** Review of efficient nonparametric functional estimation focusing on causal-inference parameters and minimax efficiency bounds.
  - **Result:** Pedagogical treatment with worked examples and derivation shortcuts; canonical entry point into TMLE/DML theory.
  - **Status:** Verified (no widely-known repo)

- **Orthogonal Statistical Learning** — Foster & Syrgkanis (2023); Annals of Statistics 2023
  - **Source:** https://arxiv.org/abs/1901.09036
  - **Code:** —
  - **Mechanism:** Generalizes Neyman orthogonality to statistical learning: meta-algorithm achieves oracle rates under Neyman-orthogonal population risk.
  - **Result:** Bridges DML to ML excess-risk theory; sufficient conditions for CATE/PLR learners to attain oracle rates.
  - **Status:** Verified (no widely-known repo)

- **An Introduction to Double/Debiased Machine Learning** — Ahrens et al. (2025); arXiv preprint
  - **Source:** https://arxiv.org/abs/2504.08324
  - **Code:** —
  - **Mechanism:** Pedagogical introduction to DML aimed at applied researchers.
  - **Result:** Walks through PLR, IRM, IIVM, sample splitting; targets practitioners.
  - **Status:** Verified (no widely-known repo)
