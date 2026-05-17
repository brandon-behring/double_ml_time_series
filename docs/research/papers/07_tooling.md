# Implementations and software tooling

DML and causal-inference software: DoubleML (R/Python), EconML (Microsoft), DoWhy (PyWhy), CausalML (Uber), hdm (R), grf (R, GRF), `tmle3` / `causalweight` / `causaltune` / `xtdml` / `ddml` (Stata), CausalImpact. Tools, not papers — these are the implementing toolchain.

**Coverage:** 15 entries.
**Anchor prefix:** `G` (this file is `07_tooling.md`).

## G1. Implementations and software tooling

- **EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation** — Microsoft Research / ALICE Team; GitHub / Microsoft Research
  - **Source:** https://github.com/py-why/EconML
  - **Code:** https://github.com/py-why/EconML
  - **Mechanism:** Microsoft's Python library implementing DML, causal forests, dynamic-DML estimators, and meta-learners.
  - **Result:** Most production-ready open-source DML implementation; chapters 5–8 of the book reference its API.
  - **Status:** Verified

- **Dynamic Double Machine Learning** — Microsoft Research / PyWhy; EconML official documentation
  - **Source:** https://econml.azurewebsites.net/spec/estimation/dynamic_dml.html
  - **Code:** https://github.com/py-why/EconML
  - **Mechanism:** Extends DML to sequential treatments under adaptive dynamic policy with high-dim confounders, modeled as Markov decision process.
  - **Result:** Available via DynamicDML class in econml.panel.dml for panel data with sequential treatments and observed confounders.
  - **Status:** Verified

- **Inferring causal impact using Bayesian structural time-series models** — Kay H. Brodersen, Fabian Gallusser, Jim Koehler, Nicolas Remy, Steven L. Scott; Annals of Applied Statistics 9(1):247-274
  - **Source:** https://arxiv.org/abs/1506.00356
  - **Code:** https://github.com/google/CausalImpact
  - **Mechanism:** Bayesian state-space diffusion-regression model predicts counterfactual market response via MCMC posterior inference.
  - **Result:** Synthetic-control approach handles local trends, seasonality, time-varying covariates; canonical online-ad campaign uplift evaluation.
  - **Status:** Verified

- **hdm: High-Dimensional Metrics in R** — Chernozhukov, Hansen & Spindler (2016); The R Journal 2016
  - **Source:** https://arxiv.org/abs/1603.01700
  - **Code:** https://cran.r-project.org/package=hdm
  - **Mechanism:** hdm R package: Lasso + post-Lasso IV with CI support for treatment effects.
  - **Result:** Pre-DML high-dimensional econometrics toolkit; supports post-double-selection methodology.
  - **Status:** Verified

- **grf: Generalized Random Forests (R package)** — Tibshirani, Athey & Wager; CRAN / GitHub
  - **Source:** https://github.com/grf-labs/grf
  - **Code:** https://github.com/grf-labs/grf
  - **Mechanism:** R package implementing generalized random forests (causal, instrumental, quantile, survival).
  - **Result:** Canonical implementation of GRF; widely used outside Python ecosystems.
  - **Status:** Verified

- **CausalML: Python Package for Causal Machine Learning** — Chen et al. (2020); arXiv preprint
  - **Source:** https://arxiv.org/abs/2002.11631
  - **Code:** https://github.com/uber/causalml
  - **Mechanism:** Uber's CausalML Python package: meta-learners, uplift trees, DragonNet, CEVAE.
  - **Result:** Production-grade meta-learner family with uplift-modeling focus.
  - **Status:** Verified

- **tmle3: The Extensible TMLE Framework** — Jeremy R. Coyle, Nima S. Hejazi, Ivana Malenica, Rachael V. Phillips, Mark J. van der Laan; R package, tlverse ecosystem
  - **Source:** https://tlverse.org/tmle3/
  - **Code:** https://github.com/tlverse/tmle3
  - **Mechanism:** Modular R framework exposing unified TMLE interface — loss-based estimation + targeting/fluctuation for arbitrary parameters.
  - **Result:** Part of tlverse ecosystem; reference implementation of TMLE methodology from van der Laan & Rose textbooks.
  - **Status:** Verified

- **DoubleML — An Object-Oriented Implementation of Double Machine Learning in R** — Bach et al. (2024); Journal of Statistical Software 2024
  - **Source:** https://www.jstatsoft.org/article/view/v108i03
  - **Code:** —
  - **Mechanism:** R implementation of DoubleML based on mlr3.
  - **Result:** Official R companion to the Python DoubleML; both maintained by original Chernozhukov-group developers.
  - **Status:** Verified (no widely-known repo)

- **DoubleML -- An Object-Oriented Implementation of Double Machine Learning in Python** — Bach et al. (2022); Journal of Machine Learning Research 2022
  - **Source:** https://arxiv.org/abs/2104.03220
  - **Code:** —
  - **Mechanism:** Python implementation of DoubleML based on scikit-learn.
  - **Result:** Reference Python implementation of the Chernozhukov et al. DML framework.
  - **Status:** Verified (no widely-known repo)

- **The causalweight package for causal inference in R** — Hugo Bodory, Martin Huber; FSES Working Papers
  - **Source:** https://ideas.repec.org/p/fri/fribow/fribow00493.html
  - **Code:** https://cran.r-project.org/package=causalweight
  - **Mechanism:** Semiparametric IPW and mediation estimators for treatment effects under selection on observables or IV.
  - **Result:** Handles non-random attrition/sample selection; bootstrap-based inference; canonical R implementation for dynamic-DML by Bodory-Huber-Lafférs.
  - **Status:** Verified

- **CausalTune: Automated Tuning and Selection for Causal Estimators** — PyWhy Contributors; Open-source Python package
  - **Source:** https://www.pywhy.org/causaltune/
  - **Code:** https://github.com/py-why/causaltune
  - **Mechanism:** Automated hyperparameter tuning of EconML estimators via energy-score validation and FLAML-driven search.
  - **Result:** Two-stage tuning of first-stage models + second-stage causal estimator; per-row treatment-impact estimates with CIs.
  - **Status:** Verified

- **Machine Learning using Stata/Python** — Giovanni Cerulli; Stata Journal 22(4):772-810
  - **Source:** https://arxiv.org/abs/2103.03122
  - **Code:** —
  - **Mechanism:** Two Stata modules (r_ml_stata, c_ml_stata) wrapping scikit-learn via Stata 16 Python integration with k-fold CV.
  - **Result:** Brings sklearn regression and classification ML into Stata workflows; reduces software fragmentation for econometricians.
  - **Status:** Verified (no widely-known repo)

- **DoWhy: An End-to-End Library for Causal Inference** — Sharma & Kıcıman (2020); arXiv preprint
  - **Source:** https://arxiv.org/abs/2011.04216
  - **Code:** https://github.com/py-why/dowhy
  - **Mechanism:** PyWhy library for graphical-causal-model-based identification + estimation pipelines.
  - **Result:** Provides Model–Identify–Estimate–Refute framework; interoperates with EconML and CausalML.
  - **Status:** Verified

- **ddml: Double/debiased machine learning in Stata** — Achim Ahrens, Christian B. Hansen, Mark E. Schaffer, Thomas Wiemann; Stata Journal 24(1):3-45
  - **Source:** https://arxiv.org/abs/2301.09397
  - **Code:** https://statalasso.github.io/
  - **Mechanism:** Stata package implementing five DDML econometric models with stacking-estimation support and existing-Stata-ML compatibility.
  - **Result:** Monte Carlo evidence recommends stacking; bridges ML and causal inference for endogenous variables in high-dim settings.
  - **Status:** Verified

- **xtdml: Double Machine Learning Estimation to Static Panel Data Models with Fixed Effects in R** — Annalivia Polselli; arXiv preprint
  - **Source:** https://arxiv.org/abs/2512.15965
  - **Code:** https://github.com/POLSEAN/xtdml
  - **Mechanism:** R package for partially-linear panel DML using mlr3 nuisance learners; first-difference / within / correlated-RE transformations.
  - **Result:** Covariate preprocessing (normalization, polynomial expansion); operationalizes block-k-fold cross-fitting for panel DML.
  - **Status:** Verified
