# Real-world replications and applied DML

Real-world replications and applied DML — user-emphasized sub-area. 401(k) eligibility/participation literature, LaLonde / Dehejia–Wahba program-evaluation reanalysis, narrative macro-shock identification, real-time vintage data, FRED macro applications, and recent applied-DML replication studies.

**Coverage:** 34 entries.
**Anchor prefix:** `F` (this file is `06_replication_applied.md`).

## F1. Real-world replications and applied DML

- **Federal Reserve Economic Data (FRED)** — Federal Reserve Bank of St. Louis (n.d., accessed 2026); Online data service
  - **Source:** https://fred.stlouisfed.org/
  - **Code:** —
  - **Mechanism:** Federal Reserve Economic Data — public macroeconomic time-series API.
  - **Result:** Canonical source of US macro data; used by manuscript's FRED-integration chapter.
  - **Status:** Verified (no widely-known repo)

- **Evaluating the econometric evaluations of training programs with experimental data** — LaLonde (1986); American Economic Review 1986
  - **Source:** https://www.jstor.org/stable/1806062
  - **Code:** —
  - **Mechanism:** Compares experimental and non-experimental estimates of training-program effects.
  - **Result:** Originated the LaLonde dataset — founding causal-inference benchmark.
  - **Status:** Verified (no widely-known repo)

- **Do 401(k) Contributions Crowd Out Other Personal Saving?** — Poterba, Venti & Wise (1995); Journal of Public Economics 1995
  - **Source:** https://www.sciencedirect.com/science/article/pii/004727279401462W
  - **Code:** —
  - **Mechanism:** Tests whether 401(k) contributions crowd out other personal saving.
  - **Result:** Foundational 401(k) crowd-out study; spawned the multi-decade debate the DML 401(k) examples replicate.
  - **Status:** Verified (no widely-known repo)

- **The Illusory Effects of Saving Incentives on Saving** — Eric M. Engen, William G. Gale, John Karl Scholz; Journal of Economic Perspectives 10(4):113-138
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/jep.10.4.113
  - **Code:** —
  - **Mechanism:** Re-examines IRA/401(k) saving-effect estimates accounting for saver-preference selection, asset substitution, and pre-tax vs after-tax comparison.
  - **Result:** Argues prior studies overstate impact; accounting for the three issues essentially eliminates the estimated effect of saving incentives.
  - **Status:** Verified (no widely-known repo)

- **Personal Retirement Saving Programs and Asset Accumulation: Reconciling the Evidence** — Poterba, Venti & Wise (1996); NBER Working Paper 5599
  - **Source:** https://www.nber.org/papers/w5599
  - **Code:** —
  - **Mechanism:** Reconciles competing evidence on personal-retirement-saving programs' impact on wealth accumulation.
  - **Result:** Brings divergent 1990s 401(k) studies together; lays out eligibility-vs-participation distinction.
  - **Status:** Verified (no widely-known repo)

- **Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs** — Dehejia & Wahba (1999); JASA 1999
  - **Source:** https://doi.org/10.1080/01621459.1999.10473858
  - **Code:** —
  - **Mechanism:** Re-analyzes LaLonde data with propensity-score matching; recovers experimental estimates from non-experimental data.
  - **Result:** Established propensity-score matching as a serious tool by demonstrating ATT recovery.
  - **Status:** Verified (no widely-known repo)

- **The economics and econometrics of active labor market programs** — James J. Heckman, Robert J. LaLonde, Jeffrey A. Smith; Handbook of Labor Economics 3:1865-2097
  - **Source:** https://ideas.repec.org/h/eee/labchp/3-31.html
  - **Code:** —
  - **Mechanism:** Handbook chapter comparing experimental and non-experimental evaluation methods for active labor market programs.
  - **Result:** Argues data quality dominates technique; documents substantial heterogeneity in program impacts across population subgroups.
  - **Status:** Verified (no widely-known repo)

- **A Real-Time Data Set for Macroeconomists** — Dean Croushore, Tom Stark; Journal of Econometrics 105(1):111-130
  - **Source:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=244554
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Semiparametric instrumental variable estimation of treatment response models** — Alberto Abadie; Journal of Econometrics 113(2):231-263
  - **Source:** https://ideas.repec.org/a/eee/econom/v113y2003i2p231-263.html
  - **Code:** —
  - **Mechanism:** Semiparametric IV estimation for treatment response under a local average response (LARF) framework with continuous outcomes.
  - **Result:** Provides root-N consistent and asymptotically normal estimators using kappa weighting; founding reference for IV-based heterogeneous treatment effect estimation.
  - **Status:** Verified (no widely-known repo)

- **Does 401(k) eligibility increase saving? Evidence from propensity score subclassification** — Daniel J. Benjamin; Journal of Public Economics 87(5-6):1259-1290
  - **Source:** https://ideas.repec.org/a/eee/pubeco/v87y2003i5-6p1259-1290.html
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Sensitivity to Exogeneity Assumptions in Program Evaluation** — Guido W. Imbens; American Economic Review 93(2):126-132
  - **Source:** https://www.semanticscholar.org/paper/Sensitivity-to-Exogeneity-Assumptions-in-Program-Imbens/f71d13e53828097cf8cd718255fc5b6393809eb9
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **The Effects of 401(K) Participation on the Wealth Distribution: An Instrumental Quantile Regression Analysis** — Chernozhukov & Hansen (2004); Review of Economics and Statistics 2004
  - **Source:** https://doi.org/10.1162/0034653041811734
  - **Code:** —
  - **Mechanism:** IV-quantile regression analysis of 401(k) participation effects on wealth distribution.
  - **Result:** Introduces the 401(k) eligibility instrument that is now the canonical applied-DML benchmark.
  - **Status:** Verified (no widely-known repo)

- **A New Measure of Monetary Shocks: Derivation and Implications** — Christina D. Romer, David H. Romer; American Economic Review 94(4):1055-1084
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/0002828042002651
  - **Code:** —
  - **Mechanism:** Narrative + quantitative identification of intended federal funds rate changes, residualizing against Fed internal forecasts.
  - **Result:** Documents large, rapid, statistically significant policy effects on output and inflation; canonical narrative-shock series.
  - **Status:** Verified (no widely-known repo)

- **Does matching overcome LaLonde's critique of nonexperimental estimators?** — Jeffrey A. Smith, Petra E. Todd; Journal of Econometrics 125(1-2):305-353
  - **Source:** https://ideas.repec.org/a/eee/econom/v125y2005i1-2p305-353.html
  - **Code:** —
  - **Mechanism:** Applies cross-sectional and longitudinal propensity-score matching to NSW data revisited by LaLonde.
  - **Result:** Difference-in-differences matching performs best; matching is sensitive to score covariates and not a general solution to LaLonde's critique.
  - **Status:** Verified (no widely-known repo)

- **Comparing Greenbook and Reduced Form Forecasts Using a Large Realtime Dataset** — Jon Faust, Jonathan H. Wright; Journal of Business & Economic Statistics 27(4):468-479
  - **Source:** https://www.tandfonline.com/doi/abs/10.1198/jbes.2009.07214
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Recent Developments in the Econometrics of Program Evaluation** — Imbens & Wooldridge (2009); Journal of Economic Literature 2009
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/jel.47.1.5
  - **Code:** —
  - **Mechanism:** JEL survey of program-evaluation econometrics: matching, IV, RDD, DiD.
  - **Result:** Standard one-stop reference for the pre-DML program-evaluation toolkit.
  - **Status:** Verified (no widely-known repo)

- **How Do 401(k)s Affect Saving? Evidence from Changes in 401(k) Eligibility** — Alexander M. Gelber; American Economic Journal: Economic Policy 3(4):103-122
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/pol.3.4.103
  - **Code:** —
  - **Mechanism:** Exploits tenure-based 401(k) eligibility changes to identify saving effects.
  - **Result:** Eligibility raises 401(k) balances; IRA assets rise while car-asset accumulation falls, supporting crowd-in interpretation.
  - **Status:** Verified (no widely-known repo)

- **Sparse Models and Methods for Optimal Instruments with an Application to Eminent Domain** — Belloni, Chen, Chernozhukov & Hansen (2012); Econometrica 2012
  - **Source:** https://arxiv.org/abs/1010.4345
  - **Code:** —
  - **Mechanism:** Lasso-based optimal-instrument selection for IV regression with many instruments.
  - **Result:** Established Lasso as valid IV-instrument selector under approximate sparsity.
  - **Status:** Verified (no widely-known repo)

- **Discretionary Tax Changes and the Macroeconomy: New Narrative Evidence from the United Kingdom** — James Cloyne; American Economic Review 103(4):1507-1528
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1507
  - **Code:** —
  - **Mechanism:** Romer-Romer narrative identification applied to UK exogenous tax policy shocks.
  - **Result:** 1% tax cut raises GDP by 0.6% on impact and 2.5% over three years; tax shocks drive major UK cycle episodes.
  - **Status:** Verified (no widely-known repo)

- **Genetic Matching for Estimating Causal Effects: A General Multivariate Matching Method for Achieving Balance in Observational Studies** — Alexis Diamond, Jasjeet S. Sekhon; Review of Economics and Statistics 95(3):932-945
  - **Source:** https://direct.mit.edu/rest/article/95/3/932/58101/Genetic-Matching-for-Estimating-Causal-Effects-A
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States** — Karel Mertens, Morten O. Ravn; American Economic Review 103(4):1212-1247
  - **Source:** https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1212
  - **Code:** —
  - **Mechanism:** Uses narratively-identified tax-liability changes as proxy SVAR instruments for structural tax shocks.
  - **Result:** Documents large short-run output effects; personal vs corporate income tax shocks differ in labor and spending channels.
  - **Status:** Verified (no widely-known repo)

- **Inference in High Dimensional Panel Models with an Application to Gun Control** — Alexandre Belloni, Victor Chernozhukov, Christian Hansen, Damian Kozbur; Journal of Business & Economic Statistics 34(4):590-605
  - **Source:** https://arxiv.org/abs/1411.6507
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **FRED-MD: A Monthly Database for Macroeconomic Research** — Michael W. McCracken, Serena Ng; Journal of Business & Economic Statistics 34(4):574-589
  - **Source:** https://ideas.repec.org/a/taf/jnlbes/v34y2016i4p574-589.html
  - **Code:** —
  - **Mechanism:** Curated monthly FRED-based macroeconomic database with automated updates and standardized transformations.
  - **Result:** Public replicable benchmark for big-data macro analysis; factors match predictive power of established alternatives.
  - **Status:** Verified (no widely-known repo)

- **Program Evaluation and Causal Inference With High-Dimensional Data** — Alexandre Belloni, Victor Chernozhukov, Iván Fernández-Val, Christian Hansen; Econometrica 85(1):233-298
  - **Source:** https://arxiv.org/abs/1311.2645
  - **Code:** —
  - **Mechanism:** Orthogonal/doubly-robust moments for LATE, LQTE under approximate sparsity for nuisance regressions.
  - **Result:** Honest confidence bands uniformly valid across regularization choices; demonstrated on 401(k) eligibility benchmark.
  - **Status:** Verified (no widely-known repo)

- **Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments** — James H. Stock, Mark W. Watson; The Economic Journal 128(610):917-948
  - **Source:** https://onlinelibrary.wiley.com/doi/abs/10.1111/ecoj.12593
  - **Code:** —
  - **Mechanism:** External-instruments / proxy-SVAR framework for identifying dynamic causal effects in macro time series.
  - **Result:** Standard reference for proxy-SVAR / LP-IV macro identification; pairs narrative or high-frequency instruments with structural VARs.
  - **Status:** Verified (no widely-known repo)

- **Estimating Treatment Effects with Causal Forests: An Application** — Susan Athey, Stefan Wager; Observational Studies (2019; vol/pages unverified 2026-05-16)
  - **Source:** https://arxiv.org/abs/1902.07409
  - **Code:** —
  - **Mechanism:** Causal forest applied to National Study of Learning Mindsets with estimated propensity scores and clustered errors.
  - **Result:** Demonstrates causal-forest implementation challenges in education research; canonical worked example for practitioners.
  - **Status:** Verified (no widely-known repo)

- **Machine learning estimation of heterogeneous causal effects: Empirical Monte Carlo evidence** — Michael C. Knaus, Michael Lechner, Anthony Strittmatter; The Econometrics Journal 24(1):134-161
  - **Source:** https://academic.oup.com/ectj/article-abstract/24/1/134/5854188
  - **Code:** https://github.com/MCKnaus/CATEs
  - **Mechanism:** Empirical Monte Carlo benchmark of 11 causal-ML estimators across 24 realistic DGPs at three aggregation levels.
  - **Result:** Four estimators (multi-step, modeling treatment and outcome processes) perform consistently well across all DGPs.
  - **Status:** Verified

- **Heterogeneous Employment Effects of Job Search Programmes: A Machine Learning Approach** — Michael C. Knaus, Michael Lechner, Anthony Strittmatter; Journal of Human Resources 57(2):597-636
  - **Source:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3029832
  - **Code:** —
  - **Mechanism:** Causal-ML CATE estimation applied to Swiss job-search programme administrative data.
  - **Result:** Documents substantial heterogeneity in employment effects of job-search programmes; canonical applied causal-ML reference.
  - **Status:** Verified (no widely-known repo)

- **Double Machine Learning based Program Evaluation under Unconfoundedness** — Michael C. Knaus; The Econometrics Journal 25(3):602-627
  - **Source:** https://arxiv.org/abs/2003.03191
  - **Code:** https://github.com/MCKnaus/causalDML
  - **Mechanism:** DML review for ATE/CATE/optimal-policy targets; introduces normalised DR-learner (NDR-learner) to stabilise individual effect estimates.
  - **Result:** Comprehensive program-evaluation pipeline applied to Swiss Active Labour Market Policy data.
  - **Status:** Verified

- **What Is the Value Added by Using Causal Machine Learning Methods in a Welfare Experiment Evaluation?** — Anthony Strittmatter; Labour Economics 84:102412
  - **Source:** https://arxiv.org/abs/1812.06533
  - **Code:** —
  - **Mechanism:** Re-evaluates Connecticut Jobs First welfare experiment using causal-ML CATE methods.
  - **Result:** Causal-ML validates theoretical labor-supply predictions where conventional CATE estimators failed; identifies limitations of both approaches.
  - **Status:** Verified (no widely-known repo)

- **The effect of plough agriculture on gender roles: A machine learning approach** — Anna Baiardi, Andrea A. Naghi; Journal of Applied Econometrics 39(7):1396-1402
  - **Source:** https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3083
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **The value added of machine learning to causal inference: evidence from revisited studies** — Anna Baiardi, Andrea A. Naghi; The Econometrics Journal 27(2):213-234
  - **Source:** https://academic.oup.com/ectj/article/27/2/213/7602388
  - **Code:** —
  - **Mechanism:** Replication study comparing causal-ML to traditional econometric estimators across published applied papers.
  - **Result:** ML methods more robust to nonlinear confounders than OLS; benefits largest when balancing many controls against precision.
  - **Status:** Verified (no widely-known repo)

- **Double Machine Learning for Sample Selection Models** — Michela Bia, Martin Huber, Lukáš Lafférs; Journal of Business & Economic Statistics (2024; vol/pages unverified 2026-05-16)
  - **Source:** https://arxiv.org/abs/2012.00745
  - **Code:** —
  - **Mechanism:** Neyman-orthogonal score for treatment + selection with cross-fitting; selection identified via observables or IV.
  - **Result:** Asymptotically normal root-n estimators; Job Corps application estimates training effects on wages conditional on employment.
  - **Status:** Verified (no widely-known repo)

- **Model Averaging and Double Machine Learning** — Achim Ahrens, Christian B. Hansen, Mark E. Schaffer, Thomas Wiemann; Journal of Applied Econometrics
  - **Source:** https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3103
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)
