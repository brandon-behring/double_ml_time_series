# Cross-sectional DML estimators and heterogeneous treatment effects

Cross-sectional / i.i.d. DML estimators (PLR, IRM, IIVM) and the heterogeneous-treatment-effect family: causal forests, generalized random forests, meta-learners (S/T/X/R/DR-learner), Bayesian causal forests, and neural-network adaptations (Dragonnet, CFR, GANITE).

**Coverage:** 20 entries.
**Anchor prefix:** `C` (this file is `03_estimator_xs.md`).

## C1. Cross-sectional DML estimators and heterogeneous treatment effects

- **Bayesian Nonparametric Modeling for Causal Inference** — Hill (2011); JCGS 2011
  - **Source:** https://doi.org/10.1198/jcgs.2010.08162
  - **Code:** —
  - **Mechanism:** BART applied to causal inference under strong ignorability.
  - **Result:** Established BART as a flexible response-surface method for treatment-effect estimation.
  - **Status:** Verified (no widely-known repo)

- **Inference on Treatment Effects After Selection Amongst High-Dimensional Controls** — Belloni, Chernozhukov & Hansen (2014); Review of Economic Studies 2014
  - **Source:** https://arxiv.org/abs/1201.0224
  - **Code:** —
  - **Mechanism:** Inference on a treatment effect after Lasso-selecting from many controls.
  - **Result:** Introduces post-double-selection / 'double Lasso' procedure for valid inference with high-dim controls.
  - **Status:** Verified (no widely-known repo)

- **Recursive Partitioning for Heterogeneous Causal Effects** — Athey & Imbens (2016); PNAS 2016
  - **Source:** https://doi.org/10.1073/pnas.1510489113
  - **Code:** —
  - **Mechanism:** Recursive partitioning that splits on treatment-effect heterogeneity rather than outcome variance.
  - **Result:** First causal-tree algorithm — changes split criterion to detect heterogeneous treatment effects.
  - **Status:** Verified (no widely-known repo)

- **Estimating individual treatment effect: generalization bounds and algorithms** — Shalit, Johansson & Sontag (2017); ICML 2017
  - **Source:** https://arxiv.org/abs/1606.03976
  - **Code:** —
  - **Mechanism:** Learns balanced representation minimizing Integral Probability Metric (Wasserstein/MMD) distance between treated and control distributions.
  - **Result:** Generalization bound decomposes ITE error into representation generalization error + induced distribution distance; yields TARNet algorithm.
  - **Status:** Verified (no widely-known repo)

- **Estimation and Inference of Heterogeneous Treatment Effects using Random Forests** — Wager & Athey (2018); JASA 2018
  - **Source:** https://arxiv.org/abs/1510.04342
  - **Code:** —
  - **Mechanism:** Causal forest as random-forest extension; asymptotically Gaussian CATE estimates with pointwise confidence intervals.
  - **Result:** First random-forest method with rigorous CATE inference — basis for the GRF / EconML stack.
  - **Status:** Verified (no widely-known repo)

- **GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets** — Yoon, Jordon & van der Schaar (2018); ICLR 2018
  - **Source:** https://openreview.net/forum?id=ByKWUeWA-
  - **Code:** https://github.com/jsyoon0823/GANITE
  - **Mechanism:** Two-stage GAN: counterfactual generator + ITE generator; counterfactual distribution learned via adversarial training.
  - **Result:** Outperforms state-of-the-art on three real-world datasets including binary and multi-treatment settings.
  - **Status:** Verified

- **Generalized Random Forests** — Athey, Tibshirani & Wager (2019); Annals of Statistics 2019
  - **Source:** https://arxiv.org/abs/1610.01271
  - **Code:** https://github.com/grf-labs/grf
  - **Mechanism:** Unified forest for solving local moment equations, including CATE.
  - **Result:** Generalizes causal-forest framework to any quantity identifiable by a local moment condition.
  - **Status:** Verified

- **Machine Learning Methods That Economists Should Know About** — Athey & Imbens (2019); Annual Review of Economics 2019
  - **Source:** https://doi.org/10.1146/annurev-economics-080217-053433
  - **Code:** —
  - **Mechanism:** Annual Review survey of ML methods relevant to econometrics.
  - **Result:** Canonical entry point for economists into modern causal-ML; pairs methods with empirical applications.
  - **Status:** Verified (no widely-known repo)

- **Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning** — Künzel et al. (2019); PNAS 2019
  - **Source:** https://arxiv.org/abs/1706.03461
  - **Code:** —
  - **Mechanism:** Survey of CATE meta-learners (S/T/X-learner); proposes X-learner with adaptive structure exploitation.
  - **Result:** X-learner targets small-treatment-group setting; achieves parametric rate when CATE is linear.
  - **Status:** Verified (no widely-known repo)

- **Orthogonal Random Forest for Causal Inference** — Oprescu, Syrgkanis & Wu (2019); ICML 2019
  - **Source:** https://arxiv.org/abs/1806.03467
  - **Code:** https://github.com/Microsoft/EconML
  - **Mechanism:** Combines Neyman-orthogonality with generalized random forests; local L1-penalized regression for high-dim nuisance.
  - **Result:** Achieves oracle error rates with high-dim controls under sparsity; supports discrete and continuous treatments.
  - **Status:** Verified

- **Adapting Neural Networks for the Estimation of Treatment Effects** — Shi, Blei & Veitch (2019); NeurIPS 2019
  - **Source:** https://arxiv.org/abs/1906.02120
  - **Code:** —
  - **Mechanism:** Dragonnet: shared-trunk neural net for outcome and propensity; targeted regularization for orthogonality.
  - **Result:** First neural-net method combining targeted regularization with propensity-score sufficiency.
  - **Status:** Verified (no widely-known repo)

- **Bayesian Regression Tree Models for Causal Inference: Regularization, Confounding, and Heterogeneous Effects** — Hahn, Murray & Carvalho (2020); Bayesian Analysis 2020
  - **Source:** https://arxiv.org/abs/1706.09523
  - **Code:** —
  - **Mechanism:** Bayesian Causal Forests (BCF): parameterizes mean and treatment-effect functions separately for RIC shrinkage.
  - **Result:** First BART variant explicitly regularizing prognostic effect separately from treatment effect.
  - **Status:** Verified (no widely-known repo)

- **Policy Learning with Observational Data** — Athey & Wager (2021); Econometrica 2021
  - **Source:** https://arxiv.org/abs/1702.02896
  - **Code:** —
  - **Mechanism:** Doubly-robust ATE estimator + policy-selection algorithm under decision-tree or budget/fairness constraints.
  - **Result:** Strong asymptotic utilitarian regret guarantees; supports binary/continuous treatments and observables-or-IV identification.
  - **Status:** Verified (no widely-known repo)

- **On Inductive Biases for Heterogeneous Treatment Effect Estimation** — Curth & van der Schaar (2021); NeurIPS 2021
  - **Source:** https://arxiv.org/abs/2106.03765
  - **Code:** https://github.com/AliciaCurth/CATENets
  - **Mechanism:** Three strategies (regularization, reparameterization, multi-task NN) encoding shared structure across potential outcomes.
  - **Result:** All three approaches improve over baselines on semi-synthetic CATE benchmarks; provides selection guidance by setting.
  - **Status:** Verified

- **Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms** — Curth & van der Schaar (2021); AISTATS 2021
  - **Source:** https://arxiv.org/abs/2101.10943
  - **Code:** https://github.com/AliciaCurth/CATENets
  - **Mechanism:** Meta-learner decomposition: plug-in estimation and pseudo-outcome regression strategies built on neural network base-learners.
  - **Result:** Comparative theoretical analysis + simulation revealing learner strengths under different DGPs; guides principled CATE-learner choice.
  - **Status:** Verified

- **Quasi-Oracle Estimation of Heterogeneous Treatment Effects** — Nie & Wager (2021); Biometrika 2021
  - **Source:** https://arxiv.org/abs/1712.04912
  - **Code:** https://github.com/xnie/rlearner
  - **Mechanism:** R-learner: two-step CATE estimation using marginal-effect and propensity-score residuals.
  - **Result:** Quasi-oracle rates for CATE — learner achieves rates as if nuisance were known.
  - **Status:** Verified

- **Shrinkage Bayesian Causal Forests for Heterogeneous Treatment Effects Estimation** — Caron, Baio & Manolopoulou (2022); Journal of Computational and Graphical Statistics 2022
  - **Source:** https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2067549
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests** — Chernozhukov, Newey, Quintas-Martinez & Syrgkanis (2022); ICML 2022
  - **Source:** https://arxiv.org/abs/2110.03031
  - **Code:** —
  - **Mechanism:** Automatically learns Riesz representer via multitask neural net (RieszNet) or locally-linear random forest (ForestRiesz).
  - **Result:** Competitive with Dragonnet for ATE; effective for average marginal effects with continuous treatment on semi-synthetic gasoline data.
  - **Status:** Verified (no widely-known repo)

- **Towards optimal doubly robust estimation of heterogeneous causal effects** — Kennedy (2023); Electronic Journal of Statistics 2023
  - **Source:** https://arxiv.org/abs/2004.14497
  - **Code:** —
  - **Mechanism:** Two-stage DR CATE estimator with model-free oracle inequality; local polynomial double-residual regression with sample splitting.
  - **Result:** Achieves oracle efficiency under weakest conditions in the literature; conjectures minimax optimality for nontrivially structured CATEs.
  - **Status:** Verified (no widely-known repo)

- **Causal Inference: A Statistical Learning Approach** — Wager (2024); Book draft (Stanford)
  - **Source:** https://web.stanford.edu/~swager/causal_inf_book.pdf
  - **Code:** —
  - **Mechanism:** Graduate-textbook treatment of causal inference from statistical-learning perspective, covering DML, GRF, policy learning.
  - **Result:** Stanford open textbook used in machine-learning-meets-econometrics courses; companion reading for GRF/EconML stack.
  - **Status:** Verified (no widely-known repo)
