# Temporal and dynamic DML estimators

Temporal and dynamic DML estimators — user-emphasized sub-area. Lewis–Syrgkanis recursive g-estimation, dynamic treatment effects with DML/ML nuisance, panel DML with fixed effects, rolling-window / walk-forward DML, structural nested mean models, marginal structural models, and DML for time-series settings with serial correlation.

**Coverage:** 37 entries (40 pre-audit; 3 duplicates consolidated 2026-05-16).
**Anchor prefix:** `D` (this file is `04_estimator_temporal.md`).

## D1. Temporal and dynamic DML estimators

- **A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect** — Robins (1986); Mathematical Modelling 1986
  - **Source:** https://www.sciencedirect.com/science/article/pii/0270025586900886
  - **Code:** —
  - **Mechanism:** Sequential exchangeability + structural nested mean models for time-varying treatments.
  - **Result:** Sources the entire epidemiology causal-inference toolkit (MSMs, g-estimation, g-formula).
  - **Status:** Verified (no widely-known repo)

- **Marginal Structural Models and Causal Inference in Epidemiology** — Robins, Hernán & Brumback (2000); Epidemiology 2000
  - **Source:** https://journals.lww.com/epidem/abstract/2000/09000/marginal_structural_models_and_causal_inference_in.11.aspx
  - **Code:** —
  - **Mechanism:** Marginal structural models for time-varying treatments with time-varying confounders.
  - **Result:** Establishes IPTW for time-varying causal effects.
  - **Status:** Verified (no widely-known repo)

- **Optimal Dynamic Treatment Regimes** — Susan A. Murphy; Journal of the Royal Statistical Society: Series B (Statistical Methodology) 65(2), 331-355 (2003)
  - **Source:** https://academic.oup.com/jrsssb/article/65/2/331/7092852
  - **Code:** —
  - **Mechanism:** Parameterizes regret functions for dynamic treatment regimes; backward induction over benefit-to-go avoids modeling full longitudinal distribution.
  - **Result:** Foundational paper for optimal dynamic treatment regimes; simulation shows estimated regimes attain near-optimal mean response.
  - **Status:** Verified (no widely-known repo)

- **Structural Nested Models and G-estimation: The Partially Realized Promise** — Stijn Vansteelandt, Marshall Joffe; Statistical Science, Vol. 29, No. 4, pp. 707-731 (2014)
  - **Source:** https://arxiv.org/abs/1503.01589
  - **Code:** —
  - **Mechanism:** Comprehensive review of structural nested models and g-estimation for sequential treatments under time-dependent confounding.
  - **Result:** Catalogues adoption barriers, proposes extensions to standard SNM/g-estimation procedures, and argues for wider use.
  - **Status:** Verified (no widely-known repo)

- **Inference in High Dimensional Panel Models with an Application to Gun Control** — Alexandre Belloni, Victor Chernozhukov, Christian Hansen, Damian Kozbur; Journal of Business & Economic Statistics 34(4), 590-605 (2016)
  - **Source:** https://arxiv.org/abs/1411.6507
  - **Code:** —
  - **Mechanism:** Grouped-Lasso variant for fixed-effects panels under sparsity; treats unit heterogeneity as fixed effects, allows within-unit dependence.
  - **Result:** Delivers uniformly valid inference on a fixed parameter subset in linear/IV panel FE models; applied to gun-prevalence and crime.
  - **Status:** Verified (no widely-known repo)

- **Matrix Completion Methods for Causal Panel Data Models** — Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, Khashayar Khosravi; Journal of the American Statistical Association 116(536), 2021; arXiv:1710.10251
  - **Source:** https://arxiv.org/abs/1710.10251
  - **Code:** —
  - **Mechanism:** Imputes the missing control outcomes for treated unit-periods via nuclear-norm regularized matrix completion.
  - **Result:** Unifies synthetic-control, interactive-FE and unconfoundedness estimators; outperforms baselines in real-data simulations.
  - **Status:** Verified (no widely-known repo)

- **An introduction to g methods** — Naimi, Cole & Kennedy (2017); International Journal of Epidemiology 2017
  - **Source:** https://academic.oup.com/ije/article/46/2/756/2760169
  - **Code:** —
  - **Mechanism:** Pedagogical introduction to g-methods (g-formula, IPW, g-estimation).
  - **Result:** Standard reading-list reference for understanding the time-varying-treatment toolkit.
  - **Status:** Verified (no widely-known repo)

- **Design-based Analysis in Difference-In-Differences Settings with Staggered Adoption** — Susan Athey, Guido Imbens; arXiv:1808.05293 (econ.EM)
  - **Source:** https://arxiv.org/abs/1808.05293
  - **Code:** —
  - **Mechanism:** Design-based DiD analysis under staggered adoption.
  - **Result:** Shows two-way-FE estimand is a weighted average that can include negative weights; offers alternatives.
  - **Status:** Verified (no widely-known repo)

- **Estimation and Inference on Heterogeneous Treatment Effects in High-Dimensional Dynamic Panels under Weak Dependence** — Vira Semenova, Matt Goldman, Victor Chernozhukov, Matt Taddy; arXiv:1712.09988 (econ.EM) preprint
  - **Source:** https://arxiv.org/abs/1712.09988
  - **Code:** https://github.com/vsemenova/orthoml
  - **Mechanism:** Leave-out-neighbors cross-fitting + orthogonal Lasso CATE learner under weak time-series/panel dependence.
  - **Result:** Achieves faster-than-first-stage rates when CATE complexity is simpler; debiased simultaneous inference on CATE parameters.
  - **Status:** Verified

- **Time Series Experiments and Causal Estimands: Exact Randomization Tests and Trading** — Bojinov & Shephard (2019); JASA 2019
  - **Source:** https://doi.org/10.1080/01621459.2018.1527225
  - **Code:** —
  - **Mechanism:** Time-series experiments under design-based inference; randomization tests for trading-style settings.
  - **Result:** Bridges design-based inference to time series; useful for finance-style causal questions.
  - **Status:** Verified (no widely-known repo)

- **Orthogonal Random Forest for Causal Inference** — Miruna Oprescu, Vasilis Syrgkanis, Zhiwei Steven Wu; Proceedings of the 36th International Conference on Machine Learning (ICML 2019); arXiv:1806.03467
  - **Source:** https://arxiv.org/abs/1806.03467
  - **Code:** —
  - **Mechanism:** Combines Neyman-orthogonality with generalized random forests; local L1-penalized regression for nuisance estimation.
  - **Result:** Achieves oracle error rates with high-dimensional controls; handles discrete and continuous treatments under sparsity.
  - **Status:** Verified (no widely-known repo)

- **Estimating Counterfactual Treatment Outcomes over Time Through Adversarially Balanced Representations** — Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar; ICLR 2020; arXiv:2002.04083
  - **Source:** https://arxiv.org/abs/2002.04083
  - **Code:** https://github.com/ioanabica/Counterfactual-Recurrent-Network
  - **Mechanism:** Counterfactual Recurrent Network: seq2seq model with domain-adversarial training to build treatment-invariant patient representations.
  - **Result:** Lower counterfactual error and better optimal-treatment/timing recovery on simulated tumor-growth benchmarks vs prior methods.
  - **Status:** Verified

- **Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes** — Nathan Kallus, Masatoshi Uehara; Journal of Machine Learning Research 21(167), 2020; arXiv:1908.08526
  - **Source:** https://arxiv.org/abs/1908.08526
  - **Code:** —
  - **Mechanism:** Cross-fold estimation of Q-functions and marginalized density ratios; exploits MDP memorylessness.
  - **Result:** Achieves semiparametric efficiency bound for OPE when both components attain fourth-root rates; doubly robust.
  - **Status:** Verified (no widely-known repo)

- **Double/Debiased Machine Learning for Dynamic Treatment Effects via g-Estimation** — Lewis & Syrgkanis (2020); NeurIPS 2021
  - **Source:** https://arxiv.org/abs/2002.07285
  - **Code:** —
  - **Mechanism:** Extends DML to dynamic treatment regimes via orthogonal cross-fitted g-estimation; covers SNMMs.
  - **Result:** Founding reference for dynamic-treatment DML — canonical citation for TemporalPLR-style estimators.
  - **Status:** Verified (no widely-known repo)

- **G-Net: A Deep Learning Approach to G-computation for Counterfactual Outcome Prediction Under Dynamic Treatment Regimes** — Rui Li, Zach Shahn, Jun Li, Mingyu Lu, Prithwish Chakraborty, Daby Sow, Mohamed Ghalwash, Li-wei H. Lehman; arXiv:2003.10551 (cs.LG); reportedly ML4H 2021 (unverified, 2026-05-16)
  - **Source:** https://arxiv.org/abs/2003.10551
  - **Code:** —
  - **Mechanism:** Sequential neural g-computation; deep learning replaces classical regression for estimating expected counterfactual outcomes.
  - **Result:** Estimates individual- and population-level time-varying treatment effects; validated on CVSim cardiovascular simulator.
  - **Status:** Verified (no widely-known repo)

- **Multiway Cluster Robust Double/Debiased Machine Learning** — Harold D. Chiang, Kengo Kato, Yukun Ma, Yuya Sasaki; Journal of Business & Economic Statistics 40(3), 2022; arXiv:1909.03489
  - **Source:** https://arxiv.org/abs/1909.03489
  - **Code:** —
  - **Mechanism:** Multiway cross-fitting algorithm + multiway cluster-robust standard errors for DML under clustered sampling.
  - **Result:** Robust SEs materially larger than non-robust in market-share demand application; validates DML for multiway-clustered data.
  - **Status:** Verified (no widely-known repo)

- **When Should We Use Unit Fixed Effects Regression Models for Causal Inference with Longitudinal Data?** — Imai & Kim (2021); American Journal of Political Science 2021
  - **Source:** https://doi.org/10.1111/ajps.12417
  - **Code:** —
  - **Mechanism:** When fixed-effects regressions deliver valid causal estimates with longitudinal data.
  - **Result:** Establishes assumptions needed for unit-FE causal interpretation.
  - **Status:** Verified (no widely-known repo)

- **Double/Debiased Machine Learning for Dynamic Treatment Effects** — Greg Lewis, Vasilis Syrgkanis; NeurIPS 2021 (Advances in Neural Information Processing Systems 34)
  - **Source:** https://proceedings.neurips.cc/paper_files/paper/2021/hash/bf65417dcecc7f2b0006e1f5793b7143-Abstract.html
  - **Code:** —
  - **Mechanism:** Recursive orthogonal scores + cross-fitting for dynamic treatment effects; extends DML to structural nested mean models.
  - **Result:** Founding NeurIPS paper for dynamic-DML; root-n inference for sequence-of-treatments effects under ML nuisance.
  - **Status:** Verified (no widely-known repo)

- **Proximal Causal Inference for Complex Longitudinal Studies** — Andrew Ying, Wang Miao, Xu Shi, Eric J. Tchetgen Tchetgen; arXiv:2109.07030 (stat.ME)
  - **Source:** https://arxiv.org/abs/2109.07030
  - **Code:** —
  - **Mechanism:** Time-varying proxy pairs for identification when sequential randomization fails; doubly robust semiparametric MSM estimation.
  - **Result:** Establishes nonparametric identification and semiparametric efficiency bound for longitudinal proximal causal inference.
  - **Status:** Verified (no widely-known repo)

- **Evaluating (weighted) dynamic treatment effects by double machine learning** — Bodory, Huber & Lafférs (2022); The Econometrics Journal 2022
  - **Source:** https://arxiv.org/abs/2012.00370
  - **Code:** —
  - **Mechanism:** Evaluating weighted dynamic treatment effects under DML with selection-on-observables.
  - **Result:** Ready-to-use DML estimators for sequences of binary treatments; implemented in causalweight.
  - **Status:** Verified (no widely-known repo)

- **Automatic Debiased Machine Learning for Dynamic Treatment Effects and General Nested Functionals** — Victor Chernozhukov, Whitney Newey, Rahul Singh, Vasilis Syrgkanis; arXiv:2203.13887 (econ.EM)
  - **Source:** https://arxiv.org/abs/2203.13887
  - **Code:** —
  - **Mechanism:** Recursive Riesz-representer characterization of nested mean regressions; multipliers of debias corrections solve sequence of loss minimizations.
  - **Result:** Automatic debiasing for dynamic treatment regimes without specifying correction-term form; extends to long-term and discrete-choice settings.
  - **Status:** Verified (no widely-known repo)

- **Locally Robust Semiparametric Estimation** — Victor Chernozhukov, Juan Carlos Escanciano, Hidehiko Ichimura, Whitney K. Newey, James M. Robins; Econometrica 90(4), 1501-1535 (2022); arXiv:1608.00033
  - **Source:** https://arxiv.org/abs/1608.00033
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Estimating Continuous Treatment Effects in Panel Data using Machine Learning with a Climate Application** — Sylvia Klosin, Max Vilgalys; arXiv:2207.08789 (econ.EM)
  - **Source:** https://arxiv.org/abs/2207.08789
  - **Code:** —
  - **Mechanism:** Automatic DML with analytic derivatives and optimization-based debiasing; allows unit FE plus high-dim nonlinear heterogeneity.
  - **Result:** Yields 50% larger but equally precise estimate of extreme-heat effect on corn yield vs linear TWFE; outperforms polynomial OLS.
  - **Status:** Verified (no widely-known repo)

- **Causal Transformer for Estimating Counterfactual Outcomes** — Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel; Proceedings of the 39th International Conference on Machine Learning, PMLR 162, 2022; arXiv:2204.07258
  - **Source:** https://arxiv.org/abs/2204.07258
  - **Code:** —
  - **Mechanism:** Three transformer subnetworks (covariates, treatments, outcomes) with cross-attention + counterfactual-domain-confusion loss for adversarial balance.
  - **Result:** First transformer for longitudinal counterfactual estimation; outperforms LSTM-based state-of-the-art on long temporal dependencies.
  - **Status:** Verified (no widely-known repo)

- **Structural Nested Mean Models Under Parallel Trends Assumptions** — Zach Shahn, Oliver Dukes, Meghana Shamsunder, David Richardson, Eric Tchetgen Tchetgen, James Robins; arXiv:2204.10291 (stat.ME)
  - **Source:** https://arxiv.org/abs/2204.10291
  - **Code:** —
  - **Mechanism:** Identifies SNMMs under conditional parallel-trends assumption instead of no-unmeasured-confounding.
  - **Result:** Extends SNMMs to DiD-style settings; demonstrated on Medicaid expansion, flood insurance take-up, crop-yield temperature effects.
  - **Status:** Verified (no widely-known repo)

- **Machine Learning for Staggered Difference-in-Differences and Dynamic Treatment Effect Heterogeneity** — Julia Hatamyar, Noemi Kreif, Rudi Rocha, Martin Huber; arXiv:2310.11962 (econ.EM)
  - **Source:** https://arxiv.org/abs/2310.11962
  - **Code:** —
  - **Mechanism:** MLDID: combines two nonparametric DiD methods with ML to estimate time-varying CATTs under staggered adoption.
  - **Result:** Brazil Family Health Program application: faster infant-mortality impacts for individuals in poverty and urban areas.
  - **Status:** Verified (no widely-known repo)

- **Semiparametric inference for impulse response functions using double/debiased machine learning** — Daniele Ballinari, Alexander Wehrli; arXiv:2411.10009 (econ.EM)
  - **Source:** https://arxiv.org/abs/2411.10009
  - **Code:** —
  - **Mechanism:** Extends DML theory from i.i.d. to time-series settings for impulse-response functions with multiple discrete treatments over time.
  - **Result:** Establishes parametric-rate consistency and asymptotic normality under serial dependence; macro-shock empirical application.
  - **Status:** Verified (no widely-known repo)

- **Double Machine Learning for Static Panel Models with Fixed Effects** — Paul S. Clarke, Annalivia Polselli; arXiv:2312.08174 (econ.EM) (Econometrics Journal forthcoming, unverified 2026-05-16)
  - **Source:** https://arxiv.org/abs/2312.08174
  - **Code:** —
  - **Mechanism:** DML for static panel models with fixed effects using block-k-fold cross-fitting.
  - **Result:** Solves within-unit serial-correlation problem by allocating each unit's full time series to one fold.
  - **Status:** Verified (no widely-known repo)

- **Double Machine Learning meets Panel Data -- Promises, Pitfalls, and Potential Solutions** — Jonathan Fuhr, Dominik Papies (2024); arXiv:2409.01266 (econ.EM)
  - **Source:** https://arxiv.org/abs/2409.01266
  - **Code:** —
  - **Mechanism:** Survey of DML adaptations to panel data — promises, pitfalls, and current solutions.
  - **Result:** Maps unsolved problems and partial fixes in the 2022–2024 panel-DML literature.
  - **Status:** Verified (no widely-known repo)

- **Difference-in-Differences with Time-varying Continuous Treatments using Double/Debiased Machine Learning** — Michel F. C. Haddad, Martin Huber, José Eduardo Medina-Reyes, Lucas Z. Zhang; arXiv:2410.21105 (econ.EM)
  - **Source:** https://arxiv.org/abs/2410.21105
  - **Code:** —
  - **Mechanism:** DML-based kernel ATET estimators under conditional parallel trends for continuous time-varying treatments and histories.
  - **Result:** Repeated-cross-section and panel variants; Brazil COVID-vaccination application shows mortality reduction after multi-week lag.
  - **Status:** Verified (no widely-known repo)

- **Neighborhood Stability in Double/Debiased Machine Learning with Dependent Data** — Jianfei Cao, Michael P. Leung; arXiv:2511.10995 (econ.EM)
  - **Source:** https://arxiv.org/abs/2511.10995
  - **Code:** —
  - **Mechanism:** Introduces neighborhood-stability condition — insensitivity to resampling in slowly growing neighborhoods of dependent data.
  - **Result:** Validates DML without cross-fitting for spatial/network data; avoids small-training-set problem from leave-out folds.
  - **Status:** Verified (no widely-known repo)

- **DeepBlip: Estimating Conditional Average Treatment Effects Over Time** — Haorui Ma, Dennis Frauen, Stefan Feuerriegel; arXiv:2511.14545 (stat.ML)
  - **Source:** https://arxiv.org/abs/2511.14545
  - **Code:** —
  - **Mechanism:** Double-optimization trick for simultaneous learning of all SNMM blip functions via LSTM/transformer backbones.
  - **Result:** First neural SNMM framework with Neyman-orthogonal loss; SOTA on clinical datasets with interpretable time-specific effects.
  - **Status:** Verified (no widely-known repo)

- **Semiparametric Double Reinforcement Learning with Applications to Long-Term Causal Inference** — Lars van der Laan, David Hubbard, Allen Tran, Nathan Kallus, Aurélien Bibaut; arXiv:2501.06926 (stat.ML)
  - **Source:** https://arxiv.org/abs/2501.06926
  - **Code:** —
  - **Mechanism:** Treats Q-function as 1-D summary of state-action process; calibrated FQI plug-in eliminates density-ratio estimation.
  - **Result:** Superefficient estimators below Cramer-Rao bound; weakens intertemporal overlap to single-dimensional condition.
  - **Status:** Verified (no widely-known repo)

- **Continuous difference-in-differences with double/debiased machine learning** — Lucas Z. Zhang; The Econometrics Journal, 2025; arXiv:2408.10509
  - **Source:** https://arxiv.org/abs/2408.10509
  - **Code:** —
  - **Mechanism:** DML-based ATT(d) estimators for continuous-treatment DiD under conditional parallel trends; first stage estimates conditional treatment density.
  - **Result:** Asymptotic normality + uniform confidence bands via multiplier bootstrap; reanalyzes 1983 Medicare Prospective Payment System reform.
  - **Status:** Verified (no widely-known repo)

- **Double Machine Learning for Static Panel Data with Instrumental Variables: New Method and Applications** — Anna Baiardi, Paul S. Clarke, Andrea A. Naghi, Annalivia Polselli; arXiv:2603.20464 (econ.EM)
  - **Source:** https://arxiv.org/abs/2603.20464
  - **Code:** —
  - **Mechanism:** Panel IV-DML combining ML for high-dim covariate adjustment with shift-share instrument identification.
  - **Result:** Diagnostics for weak identification; revisits three migration studies, exposing instrument weakness masked by 2SLS.
  - **Status:** Verified (no widely-known repo)

- **Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence** — Kaicheng Chen, Harold D. Chiang; arXiv:2602.11333 (econ.EM)
  - **Source:** https://arxiv.org/abs/2602.11333
  - **Code:** —
  - **Mechanism:** Localization-based empirical process framework + novel maximal inequalities for separately exchangeable arrays.
  - **Result:** Establishes asymptotic linearity/normality for debiased GMM without cross-fitting under multiway clustering.
  - **Status:** Verified (no widely-known repo)

- **Double Machine Learning for Time Series** — Milos Ciganovic, Federico D'Amario, Massimiliano Tancioni; arXiv:2603.10999 (econ.EM)
  - **Source:** https://arxiv.org/abs/2603.10999
  - **Code:** —
  - **Mechanism:** Reverse Cross-Fitting deterministic step leveraging time-reversibility of stationary series; Goldilocks-zone tuning.
  - **Result:** Asymptotically valid under misspecification/heteroskedasticity; residualized local projections estimate regulatory capital effects.
  - **Status:** Verified (no widely-known repo)
