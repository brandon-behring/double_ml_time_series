# Benchmark Datasets + Synthetic DGPs

Methodology benchmarks and synthetic data-generating processes — ACIC competition data (2016-2022), IHDP semi-synthetic (Hill 2011), Twins (Almond-Chay-Lee / Louizos CEVAE), LaLonde-NSW, JOBS II, plus the simulated DGPs shipped in DoubleML, EconML, CausalML, GRF, DoWhy, CATENets, CSuite, and RealCause. These let researchers validate DML estimators against known ground-truth ATE/CATE.

---

## D1. ACIC 2017 Competition Data (Targeted Selection / CATE Challenge)

- **ACIC 2017 Competition Data (Targeted Selection / CATE Challenge)** — GitHub (2017).
  - **Source:** https://github.com/vdorie/aciccomp/tree/master/2017
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 32 parameter settings x 250 replications, task_family: regression
  - **Size+License:** license: open source (R package aciccomp2017)
  - **Tasks:** Hahn, P. R., Dorie, V., & Murray, J. S. (2019). Atlantic Causal Inference Conference (ACIC) Data Analysis Challenge 2017. arXiv:1905.09515.
  - **Status:** Verified.

## D2. ACIC 2019 Data Challenge

- **ACIC 2019 Data Challenge** — institutional / other (2019).
  - **Source:** https://sites.google.com/view/acic2019datachallenge/home
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 3200 datasets per track (low-dim 500x20 and high-dim 1000x200), task_family: regression
  - **Size+License:** license: open access for research
  - **Tasks:** ACIC 2019 Data Challenge. Atlantic Causal Inference Conference, McGill University.
  - **Status:** Verified.

## D3. ACIC 2022 Data Challenge (Mathematica)

- **ACIC 2022 Data Challenge (Mathematica)** — institutional / other (2022).
  - **Source:** https://acic2022.mathematica.org/
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: open access for research
  - **Tasks:** ACIC 2022 Data Challenge. Atlantic Causal Inference Conference, hosted by Mathematica Policy Research.
  - **Status:** Unverified.

## D4. Atlantic Causal Inference Conference (ACIC) 2016 Competition Data

- **Atlantic Causal Inference Conference (ACIC) 2016 Competition Data** — GitHub (2016).
  - **Source:** https://github.com/vdorie/aciccomp/tree/master/2016
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 77 simulation settings x 100 replications x 4302 obs, task_family: regression
  - **Size+License:** license: open source (R package aciccomp2016; check LICENSE file)
  - **Tasks:** Dorie, V., Hill, J., Shalit, U., Scott, M., & Cervone, D. (2019). Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition. Statistical Science, 34(1), 43-68.
  - **Status:** Verified.

## D5. CATENets Treatment-Effect Benchmark Suite (ACIC 2016, IHDP, Twins loaders)

- **CATENets Treatment-Effect Benchmark Suite (ACIC 2016, IHDP, Twins loaders)** — GitHub (2021).
  - **Source:** https://github.com/AliciaCurth/CATENets
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: BSD-3-Clause
  - **Tasks:** Curth, A., & van der Schaar, M. (2021). Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms. AISTATS 2021. Also: Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in Treatment Effect Estimation (NeurIPS 2021 Datasets & Benchmarks track). Repo includes catenets/datasets/ with dataset_acic2016.py, dataset_ihdp.py, dataset_twins.py.
  - **Status:** Verified.

## D6. CausalML Dataset Module (synthetic, regression, classification, semi-synthetic generators)

- **CausalML Dataset Module (synthetic, regression, classification, semi-synthetic generators)** — GitHub (2019).
  - **Source:** https://github.com/uber/causalml/tree/master/causalml/dataset
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: Apache-2.0
  - **Tasks:** Chen, H., Harinen, T., Lee, J.-Y., Yung, M., & Zhao, Z. (2020). CausalML: Python package for causal machine learning. arXiv:2002.11631. Module causalml/dataset/ provides synthetic.py, regression.py, classification.py, semiSynthetic.py.
  - **Status:** Verified.

## D7. CSuite: Benchmark Datasets for Causality (Microsoft Research)

- **CSuite: Benchmark Datasets for Causality (Microsoft Research)** — GitHub (2022).
  - **Source:** https://github.com/microsoft/csuite
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 15 datasets x 4000 train + 2000 test + interventional environments, task_family: regression
  - **Size+License:** license: MIT
  - **Tasks:** Geffner, T., Antoran, J., Foster, A., Gong, W., Ma, C., Kiciman, E., Sharma, A., Lamb, A., Kukla, M., Pawlowski, N., Allamanis, M., & Zhang, C. (2022). Deep End-to-end Causal Inference. arXiv:2202.02195.
  - **Status:** Verified.

## D8. DoubleML fetch_401K Dataset (SIPP 1991 financial wealth + 401(k) participation)

- **DoubleML fetch_401K Dataset (SIPP 1991 financial wealth + 401(k) participation)** — GitHub (2018).
  - **Source:** https://docs.doubleml.org/stable/api/datasets.html
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 9915, task_family: regression
  - **Size+License:** license: open source (BSD-3, packaged within DoubleML)
  - **Tasks:** Chernozhukov, V., Hansen, C., Spindler, M., & Bach, P. (2018). DoubleML: Double Machine Learning in Python. JMLR (with fetch_401K convenience wrapper). Data source: Survey of Income and Program Participation (SIPP) 1991.
  - **Status:** Verified.

## D9. DoubleML fetch_bonus: Pennsylvania Reemployment Bonus Experiment

- **DoubleML fetch_bonus: Pennsylvania Reemployment Bonus Experiment** — GitHub (2000).
  - **Source:** https://docs.doubleml.org/stable/api/datasets.html
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 13913, task_family: regression
  - **Size+License:** license: open source (data ships within DoubleML; original from Bilias 2000 JAE Data Archive)
  - **Tasks:** Bilias, Y. (2000). Sequential testing of duration data: The case of the Pennsylvania reemployment bonus experiment. Journal of Applied Econometrics, 15(6), 575-594. Convenience wrapper: doubleml.datasets.fetch_bonus.
  - **Status:** Verified.

## D10. DoubleML synthetic DGP suite (make_plr_CCDDHNR2018, make_irm_data, make_iivm_data, etc.)

- **DoubleML synthetic DGP suite (make_plr_CCDDHNR2018, make_irm_data, make_iivm_data, etc.)** — GitHub (2018).
  - **Source:** https://docs.doubleml.org/stable/api/datasets.html
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: open source (BSD-3)
  - **Tasks:** DoubleML team. doubleml.datasets module: synthetic DGP generators including make_plr_CCDDHNR2018 (Chernozhukov-Chetverikov-Demirer-Duflo-Hansen-Newey-Robins 2018), make_plr_turrell2018, make_pliv_CHS2015, make_irm_data, make_iivm_data, make_heterogeneous_data, make_did_SZ2020, make_did_CS2021, make_simple_rdd_data, make_pliv_multiway_cluster_CKMS2021. Implementation in DoubleML-for-py.
  - **Status:** Verified.

## D11. DoWhy Example Datasets (linear_dataset generator, IHDP, hotel-booking case studies)

- **DoWhy Example Datasets (linear_dataset generator, IHDP, hotel-booking case studies)** — GitHub (2019).
  - **Source:** https://github.com/py-why/dowhy
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: MIT (DoWhy repository)
  - **Tasks:** Sharma, A., & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference. arXiv:2011.04216. PyWhy organization. Includes dowhy.datasets.linear_dataset() generator with configurable beta, num_common_causes, num_instruments, num_samples.
  - **Status:** Verified.

## D12. EconML Example Notebooks (Customer Scenarios, AutomatedML, Solutions)

- **EconML Example Notebooks (Customer Scenarios, AutomatedML, Solutions)** — GitHub (2019).
  - **Source:** https://github.com/microsoft/EconML/tree/main/notebooks
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: MIT (EconML repository)
  - **Tasks:** Microsoft Research / PyWhy. EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation. Customer Scenarios (segmentation, multi-investment attribution, AB testing, LaLonde training) and AutomatedML demonstrations.
  - **Status:** Verified.

## D13. GRF (Generalized Random Forests) Test Datasets

- **GRF (Generalized Random Forests) Test Datasets** — GitHub (2018).
  - **Source:** https://github.com/grf-labs/grf/tree/master/r-package/grf/tests/testthat
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: GPL-3
  - **Tasks:** Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. Annals of Statistics, 47(2), 1148-1178. Test data in grf/r-package/grf/tests/testthat/data/ covers causal_forest, instrumental_forest, multi_arm_causal_forest, survival_forest test cases.
  - **Status:** Verified.

## D14. Huber-Lechner-Wunsch Empirical Monte Carlo Design (German UI placebo)

- **Huber-Lechner-Wunsch Empirical Monte Carlo Design (German UI placebo)** — institutional / other (2013).
  - **Source:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1696892
  - **Access:** signup wall; auth_required: Y
  - **Schema:** task_family: regression
  - **Size+License:** license: replication archive; underlying German IAB data restricted
  - **Tasks:** Huber, M., Lechner, M., & Wunsch, C. (2013). The performance of estimators based on the propensity score. Journal of Econometrics, 175(1), 1-21. Empirical Monte Carlo design using placebo treatments on German jobseeker data.
  - **Status:** Verified.

## D15. IHDP (Infant Health and Development Program) Semi-Synthetic Benchmark

- **IHDP (Infant Health and Development Program) Semi-Synthetic Benchmark** — institutional / other (2011).
  - **Source:** https://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 747 obs per replication x 1000 replications; 25 covariates, task_family: regression
  - **Size+License:** license: open access (research use; redistribution by Fredrik Johansson)
  - **Tasks:** Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. Journal of Computational and Graphical Statistics, 20(1), 217-240. Semi-synthetic IHDP NPCI dataset assembled from Brooks-Gunn et al. IHDP RCT (1990-1991).
  - **Status:** Verified.

## D16. JOBS II Preventive Intervention for Unemployed Job Seekers (Vinokur-Price)

- **JOBS II Preventive Intervention for Unemployed Job Seekers (Vinokur-Price)** — ICPSR / openICPSR (1991).
  - **Source:** https://www.icpsr.umich.edu/web/ICPSR/studies/2739
  - **Access:** signup wall; auth_required: Y
  - **Schema:** rows: 1801 (671 received the JOBS II intervention), task_family: regression
  - **Size+License:** license: ICPSR member-institution access; available via Researcher Passport for non-members
  - **Tasks:** Vinokur, A. D., & Price, R. H. (2000). JOBS II Preventive Intervention for Unemployed Job Seekers, 1991-1993: [Southeast Michigan]. ICPSR02739-v1. Ann Arbor, MI: Inter-university Consortium for Political and Social Research.
  - **Status:** Verified.

## D17. Knaus-Lechner-Strittmatter Empirical Monte Carlo DGP Framework (Swiss UI data)

- **Knaus-Lechner-Strittmatter Empirical Monte Carlo DGP Framework (Swiss UI data)** — institutional / other (2021).
  - **Source:** https://arxiv.org/abs/1810.13237
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** license: replication code on GitHub; underlying Swiss labor data restricted
  - **Tasks:** Knaus, M. C., Lechner, M., & Strittmatter, A. (2021). Machine learning estimation of heterogeneous causal effects: empirical Monte Carlo evidence. The Econometrics Journal, 24(1), 134-161. 24 DGPs x 11 estimators x 3 aggregation levels.
  - **Status:** Verified.

## D18. LaLonde NSW (National Supported Work) Dataset (Dehejia-Wahba 1999 version)

- **LaLonde NSW (National Supported Work) Dataset (Dehejia-Wahba 1999 version)** — institutional / other (1986).
  - **Source:** https://users.nber.org/~rdehejia/nswdata2.html
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 722 treated/control (NSW experimental) + PSID/CPS observational comparison samples, task_family: regression
  - **Size+License:** license: CC BY-NC 2.0 (attributable non-commercial use; citation of Dehejia & Wahba 1999/2002 and LaLonde 1986 required)
  - **Tasks:** LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental data. AER, 76(4), 604-620. Dehejia, R. H., & Wahba, S. (1999). Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs. JASA, 94(448), 1053-1062.
  - **Status:** Verified.

## D19. RealCause: Realistic Causal Inference Benchmarking (LaLonde PSID/CPS + Twins generative)

- **RealCause: Realistic Causal Inference Benchmarking (LaLonde PSID/CPS + Twins generative)** — GitHub (2020).
  - **Source:** https://github.com/bradyneal/realcause
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 100 samples per realistic dataset x 3 base datasets (LaLonde PSID, LaLonde CPS, Twins), task_family: regression
  - **Size+License:** license: MIT
  - **Tasks:** Neal, B., Huang, C.-W., & Raghupathi, S. (2020). RealCause: Realistic Causal Inference Benchmarking. arXiv:2011.15007.
  - **Status:** Verified.

## D20. Twins Birth-Weight Dataset (Almond-Chay-Lee, augmented by Louizos et al.)

- **Twins Birth-Weight Dataset (Almond-Chay-Lee, augmented by Louizos et al.)** — GitHub (2017).
  - **Source:** https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS
  - **Access:** direct download; auth_required: N
  - **Schema:** rows: 11,984 twin pairs; 46 covariates, task_family: regression
  - **Size+License:** license: open access (research use; original NBER linked birth-infant death data)
  - **Tasks:** Almond, D., Chay, K. Y., & Lee, D. S. (2005). The costs of low birth weight. Quarterly Journal of Economics, 120(3), 1031-1083. Augmented for causal inference by Louizos, C., et al. (2017). Causal Effect Inference with Deep Latent-Variable Models. NeurIPS 2017.
  - **Status:** Verified.

## D21. Twins Dataset (Zenodo Mirror, Doutreligne 2025)

- **Twins Dataset (Zenodo Mirror, Doutreligne 2025)** — Zenodo (2025).
  - **Source:** https://zenodo.org/records/14674618
  - **Access:** direct download; auth_required: N
  - **Schema:** task_family: regression
  - **Size+License:** 4.1 MB; license: CC0 (Creative Commons Zero v1.0 Universal / public domain)
  - **Tasks:** Doutreligne, M. (2025). TWINS dataset used for experiment in the paper How to select predictive models for causal inference? Zenodo. DOI: 10.5281/zenodo.14674618.
  - **Status:** Verified.

