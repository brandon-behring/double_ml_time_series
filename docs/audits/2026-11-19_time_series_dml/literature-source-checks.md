# Paper ledger source-identity checks

On November 19, 2026, 115 of the 163 ledger rows returned machine-readable title metadata. Of these, 111 matched the ledger after lowercasing and reducing punctuation and whitespace to token boundaries. Two showed minor title variations, and two resolved to different works. A further 48 entries did not yield usable title metadata in this pass. The unit is a ledger row, so duplicate identifiers count more than once. A matching title establishes source identity only at this level of comparison. It does not verify an attribution, publication status, simulation result or the accuracy of a chapter's interpretation. [Original ledger](../../research/bib_ledger.yml) · [Substantive review](05-literature.md)

The check retrieved arXiv titles by identifier, used DOI-deposit titles from Crossref, and inspected citation-title or Dublin Core title metadata on remaining reachable pages. For a missing entry below, the link may still be usable by a human reader. Rate limiting, absent title metadata, PDF-only pages, and non-paper resources all occur in this corpus.

- **Ding on FWL standard errors.** The stored arXiv link returns a software-engineering paper. Use Ding's article and correct the ledger and its generated indexes together. [Correct paper](https://doi.org/10.1016/j.spl.2020.108945)
- **Kiefer and Vogelsang on fixed-bandwidth HAC.** The stored DOI ends in 5050516 and returns a paper about controlling for too many factors. The intended journal article ends in 5050565. This affects Chapter 5's path to the inference literature. [Publisher record](https://doi.org/10.1017/S0266466605050565)

| Ledger key and stored source | Result | Returned title |
|---|---|---|
| [frisch1933partial](https://www.jstor.org/stable/1907330) | No machine-readable title obtained | — |
| [lovell1963seasonal](https://doi.org/10.1080/01621459.1963.10480682) | No machine-readable title obtained | — |
| [robinson1988rootn](https://www.econometricsociety.org/publications/econometrica/1988/07/01/root-n-consistent-semiparametric-regression) | No machine-readable title obtained | — |
| [lovell2008simple](https://doi.org/10.3200/JECE.39.1.88-91) | Exact normalized title · Crossref | A Simple Proof of the FWL Theorem |
| [ding2021frisch](https://arxiv.org/abs/2106.02087) | Different work · arXiv | Unsupervised Learning of General-Purpose Embeddings for Code Changes |
| [basu2023yule](https://arxiv.org/abs/2307.00369) | Exact normalized title · arXiv | The Yule-Frisch-Waugh-Lovell Theorem |
| [chernozhukov2018double](https://academic.oup.com/ectj/article/21/1/C1/5056401) | No machine-readable title obtained | — |
| [chernozhukov2017double](https://arxiv.org/abs/1608.00060) | Exact normalized title · arXiv | Double/Debiased Machine Learning for Treatment and Causal Parameters |
| [chernozhukov2017aer](https://www.aeaweb.org/articles?id=10.1257/aer.p20171038) | Exact normalized title · Crossref | Double/Debiased/Neyman Machine Learning of Treatment Effects |
| [chernozhukov2022locally](https://arxiv.org/abs/1608.00033) | Exact normalized title · arXiv | Locally Robust Semiparametric Estimation |
| [ichimura2022influence](https://arxiv.org/abs/1508.01378) | Exact normalized title · arXiv | The Influence Function of Semiparametric Estimators |
| [newey1994asymptotic](https://www.jstor.org/stable/2951752) | No machine-readable title obtained | — |
| [newey1990semiparametric](https://doi.org/10.1002/jae.3950050202) | Exact normalized title · Crossref | Semiparametric efficiency bounds |
| [foster2023orthogonal](https://arxiv.org/abs/1901.09036) | Exact normalized title · arXiv | Orthogonal Statistical Learning |
| [neyman1959optimal](https://www.jstor.org/stable/i285003) | No machine-readable title obtained | — |
| [ahrens2025introduction](https://arxiv.org/abs/2504.08324) | Exact normalized title · arXiv | An Introduction to Double/Debiased Machine Learning |
| [vandervaart2000asymptotic](https://doi.org/10.1017/CBO9780511802256) | Exact normalized title · Crossref | Asymptotic Statistics |
| [wager2018estimation](https://arxiv.org/abs/1510.04342) | Exact normalized title · arXiv | Estimation and Inference of Heterogeneous Treatment Effects using Random Forests |
| [nie2021quasi](https://arxiv.org/abs/1712.04912) | Exact normalized title · arXiv | Quasi-Oracle Estimation of Heterogeneous Treatment Effects |
| [kuenzel2019metalearners](https://arxiv.org/abs/1706.03461) | Exact normalized title · arXiv | Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning |
| [athey2019grf](https://arxiv.org/abs/1610.01271) | Exact normalized title · arXiv | Generalized Random Forests |
| [athey2016recursive](https://doi.org/10.1073/pnas.1510489113) | Exact normalized title · Crossref | Recursive partitioning for heterogeneous causal effects |
| [hill2011bayesian](https://doi.org/10.1198/jcgs.2010.08162) | Exact normalized title · Crossref | Bayesian Nonparametric Modeling for Causal Inference |
| [shi2019adapting](https://arxiv.org/abs/1906.02120) | Exact normalized title · arXiv | Adapting Neural Networks for the Estimation of Treatment Effects |
| [belloni2014inference](https://arxiv.org/abs/1201.0224) | Exact normalized title · arXiv | Inference on Treatment Effects After Selection Amongst High-Dimensional Controls |
| [hahn2020bayesian](https://arxiv.org/abs/1706.09523) | Exact normalized title · arXiv | Bayesian regression tree models for causal inference: regularization, confounding, and heterogeneous effects |
| [athey2019machine](https://doi.org/10.1146/annurev-economics-080217-053433) | Exact normalized title · Crossref | Machine Learning Methods That Economists Should Know About |
| [lewis2020dynamic](https://arxiv.org/abs/2002.07285) | Exact normalized title · arXiv | Double/Debiased Machine Learning for Dynamic Treatment Effects via g-Estimation |
| [bodory2022evaluating](https://arxiv.org/abs/2012.00370) | Exact normalized title · arXiv | Evaluating (weighted) dynamic treatment effects by double machine learning |
| [clarke2024doublemlpanel](https://arxiv.org/abs/2312.08174) | Exact normalized title · arXiv | Double Machine Learning for Static Panel Models with Fixed Effects |
| [fuhr2024dmlpaneldata](https://arxiv.org/abs/2409.01266) | Exact normalized title · arXiv | Double Machine Learning meets Panel Data -- Promises, Pitfalls, and Potential Solutions |
| [robins1986newapproach](https://www.sciencedirect.com/science/article/pii/0270025586900886) | No machine-readable title obtained | — |
| [robins2000marginal](https://journals.lww.com/epidem/abstract/2000/09000/marginal_structural_models_and_causal_inference_in.11.aspx) | No machine-readable title obtained | — |
| [naimi2017introduction](https://academic.oup.com/ije/article/46/2/756/2760169) | No machine-readable title obtained | — |
| [bojinov2019time](https://doi.org/10.1080/01621459.2018.1554455) | No machine-readable title obtained | — |
| [imai2021should](https://doi.org/10.1111/ajps.12417) | Exact normalized title · Crossref | When Should We Use Unit Fixed Effects Regression Models for Causal Inference with Longitudinal Data? |
| [athey2022design](https://arxiv.org/abs/1808.05293) | Exact normalized title · arXiv | Design-based Analysis in Difference-In-Differences Settings with Staggered Adoption |
| [newey1987simple](https://www.jstor.org/stable/1913610) | No machine-readable title obtained | — |
| [andrews1991hac](https://www.jstor.org/stable/2938229) | No machine-readable title obtained | — |
| [driscoll1998consistent](https://doi.org/10.1162/003465398557825) | Exact normalized title · Crossref | Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data |
| [kiefer2005newasym](https://doi.org/10.1017/S0266466605050516) | Different work · Crossref | VIOLATING IGNORABILITY OF TREATMENT BY CONTROLLING FOR TOO MANY FACTORS |
| [bergmeir2018note](https://doi.org/10.1016/j.csda.2017.11.003) | Exact normalized title · Crossref | A note on the validity of cross-validation for evaluating autoregressive time series prediction |
| [deprado2018advances](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) | No machine-readable title obtained | — |
| [lalonde1986evaluating](https://www.jstor.org/stable/1806062) | No machine-readable title obtained | — |
| [dehejia1999causal](https://doi.org/10.1080/01621459.1999.10473858) | Exact normalized title · Crossref | Causal Effects in Nonexperimental Studies: Reevaluating the Evaluation of Training Programs |
| [chernozhukov2004effects](https://doi.org/10.1162/0034653041811734) | Exact normalized title · Crossref | The Effects of 401(K) Participation on the Wealth Distribution: An Instrumental Quantile Regression Analysis |
| [poterba1995crowdout](https://www.sciencedirect.com/science/article/pii/004727279401462W) | No machine-readable title obtained | — |
| [poterba1996reconciling](https://www.nber.org/papers/w5599) | Exact normalized title · Page metadata | Personal Retirement Saving Programs and Asset Accumulation: Reconciling the Evidence |
| [belloni2012sparse](https://arxiv.org/abs/1010.4345) | Exact normalized title · arXiv | Sparse Models and Methods for Optimal Instruments with an Application to Eminent Domain |
| [imbens2009recent](https://www.aeaweb.org/articles?id=10.1257/jel.47.1.5) | Exact normalized title · Crossref | Recent Developments in the Econometrics of Program Evaluation |
| [fred](https://fred.stlouisfed.org/) | No machine-readable title obtained | — |
| [econml](https://github.com/py-why/EconML) | No machine-readable title obtained | — |
| [dowhy2022](https://arxiv.org/abs/2011.04216) | Exact normalized title · arXiv | DoWhy: An End-to-End Library for Causal Inference |
| [chen2020causalml](https://arxiv.org/abs/2002.11631) | Exact normalized title · arXiv | CausalML: Python Package for Causal Machine Learning |
| [bach2022doubleml](https://www.jstatsoft.org/article/view/v108i03) | Exact normalized title · Page metadata | DoubleML: An Object-Oriented Implementation of Double Machine Learning in R |
| [bach2022doublemlpy](https://arxiv.org/abs/2104.03220) | Exact normalized title · arXiv | DoubleML -- An Object-Oriented Implementation of Double Machine Learning in Python |
| [spindler2016hdm](https://arxiv.org/abs/1603.01700) | Title variation · arXiv | High-Dimensional Metrics in R |
| [athey2019grfpkg](https://github.com/grf-labs/grf) | No machine-readable title obtained | — |
| [lewis2021dynamicneurips](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bf65417dcecc7f2b0006e1f5793b7143-Abstract.html) | Exact normalized title · Page metadata | Double/Debiased Machine Learning for Dynamic Treatment Effects |
| [chernozhukov2022adml](https://arxiv.org/abs/2203.13887) | Exact normalized title · arXiv | Automatic Debiased Machine Learning for Dynamic Treatment Effects and General Nested Functionals |
| [vansteelandt2014snmm](https://arxiv.org/abs/1503.01589) | Exact normalized title · arXiv | Structural Nested Models and G-estimation: The Partially Realized Promise |
| [shahn2022snmmparalleltrends](https://arxiv.org/abs/2204.10291) | Exact normalized title · arXiv | Structural Nested Mean Models Under Parallel Trends Assumptions |
| [semenova2018orthoml](https://arxiv.org/abs/1712.09988) | Exact normalized title · arXiv | Estimation and Inference on Heterogeneous Treatment Effects in High-Dimensional Dynamic Panels under Weak Dependence |
| [belloni2016highdimpanel](https://arxiv.org/abs/1411.6507) | Exact normalized title · arXiv | Inference in High Dimensional Panel Models with an Application to Gun Control |
| [clarke2024dmlpanel](https://arxiv.org/abs/2312.08174) | Exact normalized title · arXiv | Double Machine Learning for Static Panel Models with Fixed Effects |
| [fuhr2024panelpitfalls](https://arxiv.org/abs/2409.01266) | Exact normalized title · arXiv | Double Machine Learning meets Panel Data -- Promises, Pitfalls, and Potential Solutions |
| [klosin2022panelapd](https://arxiv.org/abs/2207.08789) | Exact normalized title · arXiv | Estimating Continuous Treatment Effects in Panel Data using Machine Learning with a Climate Application |
| [ciganovic2026dmltimeseries](https://arxiv.org/abs/2603.10999) | Exact normalized title · arXiv | Double Machine Learning for Time Series |
| [ballinari2024dmlirf](https://arxiv.org/abs/2411.10009) | Exact normalized title · arXiv | Semiparametric inference for impulse response functions using double/debiased machine learning |
| [chiang2021multiwaydml](https://arxiv.org/abs/1909.03489) | Exact normalized title · arXiv | Multiway Cluster Robust Double/Debiased Machine Learning |
| [chen2026crossfittingfree](https://arxiv.org/abs/2602.11333) | Exact normalized title · arXiv | Cross-Fitting-Free Debiased Machine Learning with Multiway Dependence |
| [cao2025neighborhoodstability](https://arxiv.org/abs/2511.10995) | Exact normalized title · arXiv | Neighborhood Stability in Double/Debiased Machine Learning with Dependent Data |
| [zhang2025contdid](https://arxiv.org/abs/2408.10509) | Exact normalized title · arXiv | Continuous difference-in-differences with double/debiased machine learning |
| [haddad2024tvcontdid](https://arxiv.org/abs/2410.21105) | Exact normalized title · arXiv | Difference-in-Differences with Time-varying Continuous Treatments using Double/Debiased Machine Learning |
| [hatamyar2023mldid](https://arxiv.org/abs/2310.11962) | Exact normalized title · arXiv | Machine Learning for Staggered Difference-in-Differences and Dynamic Treatment Effect Heterogeneity |
| [athey2018staggeredadoption](https://arxiv.org/abs/1808.05293) | Exact normalized title · arXiv | Design-based Analysis in Difference-In-Differences Settings with Staggered Adoption |
| [athey2017matrixcompletion](https://arxiv.org/abs/1710.10251) | Exact normalized title · arXiv | Matrix Completion Methods for Causal Panel Data Models |
| [oprescu2019orthorf](https://arxiv.org/abs/1806.03467) | Exact normalized title · arXiv | Orthogonal Random Forest for Causal Inference |
| [chernozhukov2022locallyrobust](https://arxiv.org/abs/1608.00033) | Exact normalized title · arXiv | Locally Robust Semiparametric Estimation |
| [kallus2020doublerl](https://arxiv.org/abs/1908.08526) | Exact normalized title · arXiv | Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes |
| [murphy2003dtr](https://academic.oup.com/jrsssb/article/65/2/331/7092852) | No machine-readable title obtained | — |
| [bica2020crn](https://arxiv.org/abs/2002.04083) | Exact normalized title · arXiv | Estimating Counterfactual Treatment Outcomes over Time Through Adversarially Balanced Representations |
| [melnychuk2022causaltransformer](https://arxiv.org/abs/2204.07258) | Exact normalized title · arXiv | Causal Transformer for Estimating Counterfactual Outcomes |
| [li2020gnet](https://arxiv.org/abs/2003.10551) | Exact normalized title · arXiv | G-Net: A Deep Learning Approach to G-computation for Counterfactual Outcome Prediction Under Dynamic Treatment Regimes |
| [ying2021proximallong](https://arxiv.org/abs/2109.07030) | Exact normalized title · arXiv | Proximal Causal Inference for Complex Longitudinal Studies |
| [baiardi2026panelivdml](https://arxiv.org/abs/2603.20464) | Exact normalized title · arXiv | Double Machine Learning for Static Panel Data with Instrumental Variables: New Method and Applications |
| [ma2025deepblip](https://arxiv.org/abs/2511.14545) | Exact normalized title · arXiv | DeepBlip: Estimating Conditional Average Treatment Effects Over Time |
| [vanderlaan2025semiparametricdrl](https://arxiv.org/abs/2501.06926) | Exact normalized title · arXiv | Semiparametric Double Reinforcement Learning with Applications to Long-Term Causal Inference |
| [sun2008optimal](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.0012-9682.2008.00822.x) | Exact normalized title · Crossref | Optimal Bandwidth Selection in Heteroskedasticity–Autocorrelation Robust Testing |
| [lazarus2018har](https://scholar.harvard.edu/files/stock/files/lazarus_lewis_stock_watson_har_inference_recommendations_for_practice_jbes_2018.pdf) | No machine-readable title obtained | — |
| [andrews1992improved](https://elischolar.library.yale.edu/cowles-discussion-paper-series/1185/) | No machine-readable title obtained | — |
| [mueller2014hac](https://www.princeton.edu/~umueller/HACtest.pdf) | No machine-readable title obtained | — |
| [newey1994automatic](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1994.pdf) | No machine-readable title obtained | — |
| [kiefer2002bartlett](https://ideas.repec.org/p/ecl/corcae/01-13.html) | Exact normalized title · Page metadata | Heteroskedasticity-Autocorrelation Robust Standard Errors Using the Bartlett Kernel without Truncation |
| [kwiatkowski1992stationarity](https://www.sciencedirect.com/science/article/abs/pii/030440769290104Y) | No machine-readable title obtained | — |
| [dickey1979distribution](https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10482531) | Exact normalized title · Crossref | Distribution of the Estimators for Autoregressive Time Series with a Unit Root |
| [phillips1988testing](https://academic.oup.com/biomet/article-abstract/75/2/335/292919) | No machine-readable title obtained | — |
| [racine2000hvblock](https://econpapers.repec.org/RePEc:eee:econom:v:99:y:2000:i:1:p:39-61) | Exact normalized title · Page metadata | Consistent cross-validatory model-selection for dependent data: hv-block cross-validation |
| [romano2005stepwise](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2005.00615.x) | Exact normalized title · Crossref | Stepwise Multiple Testing as Formalized Data Snooping |
| [vogelsang2012heteroskedasticity](https://ideas.repec.org/a/eee/econom/v166y2012i2p303-319.html) | Exact normalized title · Page metadata | Heteroskedasticity, autocorrelation, and spatial correlation robust inference in linear panel models with fixed-effects |
| [engen1996illusory](https://www.aeaweb.org/articles?id=10.1257/jep.10.4.113) | Exact normalized title · Crossref | The Illusory Effects of Saving Incentives on Saving |
| [benjamin2003does](https://ideas.repec.org/a/eee/pubeco/v87y2003i5-6p1259-1290.html) | Exact normalized title · Page metadata | Does 401(k) eligibility increase saving?: Evidence from propensity score subclassification |
| [abadie2003semiparametric](https://ideas.repec.org/a/eee/econom/v113y2003i2p231-263.html) | Exact normalized title · Page metadata | Semiparametric instrumental variable estimation of treatment response models |
| [gelber2011how](https://www.aeaweb.org/articles?id=10.1257/pol.3.4.103) | Exact normalized title · Crossref | How do 401(k)s Affect Saving? Evidence from Changes in 401(k) Eligibility |
| [smith2005does](https://ideas.repec.org/a/eee/econom/v125y2005i1-2p305-353.html) | Exact normalized title · Page metadata | Does matching overcome LaLonde's critique of nonexperimental estimators? |
| [imbens2003sensitivity](https://www.semanticscholar.org/paper/Sensitivity-to-Exogeneity-Assumptions-in-Program-Imbens/f71d13e53828097cf8cd718255fc5b6393809eb9) | No machine-readable title obtained | — |
| [diamond2013genetic](https://direct.mit.edu/rest/article/95/3/932/58101/Genetic-Matching-for-Estimating-Causal-Effects-A) | No machine-readable title obtained | — |
| [mertens2013dynamic](https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1212) | Exact normalized title · Crossref | The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States |
| [romer2004new](https://www.aeaweb.org/articles?id=10.1257/0002828042002651) | Exact normalized title · Crossref | A New Measure of Monetary Shocks: Derivation and Implications |
| [cloyne2013discretionary](https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1507) | Exact normalized title · Crossref | Discretionary Tax Changes and the Macroeconomy: New Narrative Evidence from the United Kingdom |
| [stock2018identification](https://onlinelibrary.wiley.com/doi/abs/10.1111/ecoj.12593) | No machine-readable title obtained | — |
| [mccracken2016fredmd](https://ideas.repec.org/a/taf/jnlbes/v34y2016i4p574-589.html) | Exact normalized title · Page metadata | FRED-MD: A Monthly Database for Macroeconomic Research |
| [croushore2001realtime](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=244554) | No machine-readable title obtained | — |
| [faust2009comparing](https://www.tandfonline.com/doi/abs/10.1198/jbes.2009.07214) | Exact normalized title · Crossref | Comparing Greenbook and Reduced Form Forecasts Using a Large Realtime Dataset |
| [knaus2022programme](https://arxiv.org/abs/2003.03191) | Exact normalized title · arXiv | Double Machine Learning based Program Evaluation under Unconfoundedness |
| [knaus2021heterogeneous](https://academic.oup.com/ectj/article-abstract/24/1/134/5854188) | No machine-readable title obtained | — |
| [knaus2022heterogeneous](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3029832) | No machine-readable title obtained | — |
| [strittmatter2023value](https://arxiv.org/abs/1812.06533) | Exact normalized title · arXiv | What Is the Value Added by Using Causal Machine Learning Methods in a Welfare Experiment Evaluation? |
| [baiardi2024plough](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3083) | Exact normalized title · Crossref | The effect of plough agriculture on gender roles: A machine learning approach |
| [baiardi2024valueadded](https://academic.oup.com/ectj/article/27/2/213/7602388) | No machine-readable title obtained | — |
| [ahrens2025modelaveraging](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3103) | No machine-readable title obtained | — |
| [bia2024doubleml](https://arxiv.org/abs/2012.00745) | Exact normalized title · arXiv | Double machine learning for sample selection models |
| [belloni2017program](https://arxiv.org/abs/1311.2645) | Exact normalized title · arXiv | Program Evaluation and Causal Inference with High-Dimensional Data |
| [belloni2016inference](https://arxiv.org/abs/1411.6507) | Exact normalized title · arXiv | Inference in High Dimensional Panel Models with an Application to Gun Control |
| [athey2019estimating](https://arxiv.org/abs/1902.07409) | Exact normalized title · arXiv | Estimating Treatment Effects with Causal Forests: An Application |
| [heckman1999economics](https://ideas.repec.org/h/eee/labchp/3-31.html) | Exact normalized title · Page metadata | The economics and econometrics of active labor market programs |
| [ahrens2024ddml](https://arxiv.org/abs/2301.09397) | Exact normalized title · arXiv | ddml: Double/debiased machine learning in Stata |
| [clarke2025xtdml](https://arxiv.org/abs/2512.15965) | Exact normalized title · arXiv | xtdml: Double Machine Learning Estimation to Static Panel Data Models with Fixed Effects in R |
| [coyle2020tmle3](https://tlverse.org/tmle3/) | No machine-readable title obtained | — |
| [bodory2022causalweight](https://ideas.repec.org/p/fri/fribow/fribow00493.html) | Exact normalized title · Page metadata | The causalweight package for causal inference in R |
| [causaltune2022](https://www.pywhy.org/causaltune/) | No machine-readable title obtained | — |
| [cerulli2022mlstata](https://arxiv.org/abs/2103.03122) | Exact normalized title · arXiv | Machine Learning using Stata/Python |
| [econmldynamicdml](https://econml.azurewebsites.net/spec/estimation/dynamic_dml.html) | No machine-readable title obtained | — |
| [brodersen2015causalimpact](https://arxiv.org/abs/1506.00356) | Exact normalized title · arXiv | Inferring causal impact using Bayesian structural time-series models |
| [basu2023fwliv](https://arxiv.org/abs/2307.12731) | Exact normalized title · arXiv | The Yule-Frisch-Waugh-Lovell Theorem for Linear Instrumental Variables Estimation |
| [cattaneo2018alternative](https://arxiv.org/abs/1507.02493) | Exact normalized title · arXiv | Inference in Linear Regression Models with Many Covariates and Heteroskedasticity |
| [davidson2003econometric](http://qed.econ.queensu.ca/ETM/ETM-davidson-mackinnon-2021.pdf) | No machine-readable title obtained | — |
| [vandergeer2014asymptotically](https://arxiv.org/abs/1303.0518) | Exact normalized title · arXiv | On asymptotically optimal confidence regions and tests for high-dimensional models |
| [zhang2014confidence](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12026) | No machine-readable title obtained | — |
| [javanmard2014confidence](https://arxiv.org/abs/1306.3171) | Exact normalized title · arXiv | Confidence Intervals and Hypothesis Testing for High-Dimensional Regression |
| [belloni2014jephighdim](https://www.aeaweb.org/articles?id=10.1257/jep.28.2.29) | Exact normalized title · Crossref | High-Dimensional Methods and Inference on Structural and Treatment Effects |
| [kennedy2022semiparametric](https://arxiv.org/abs/2203.06469) | Exact normalized title · arXiv | Semiparametric doubly robust targeted double machine learning: a review |
| [hines2022demystifying](https://arxiv.org/abs/2107.00681) | Exact normalized title · arXiv | Demystifying statistical learning based on efficient influence functions |
| [chernozhukov2022locallyriesz](https://arxiv.org/abs/1802.08667) | Title variation · arXiv | De-Biased Machine Learning of Global and Local Parameters Using Regularized Riesz Representers |
| [chernozhukov2022automatic](https://arxiv.org/abs/1809.05224) | Exact normalized title · arXiv | Automatic Debiased Machine Learning of Causal and Structural Effects |
| [vanderlaan2006targeted](https://biostats.bepress.com/ucbbiostat/paper213/) | No machine-readable title obtained | — |
| [robins1994estimation](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476818) | No machine-readable title obtained | — |
| [bickel1998efficient](https://link.springer.com/book/9780387984735) | No machine-readable title obtained | — |
| [robins2008higher](https://arxiv.org/abs/0805.3040) | Exact normalized title · arXiv | Higher order influence functions and minimax estimation of nonlinear functionals |
| [kennedy2023optimal](https://arxiv.org/abs/2004.14497) | Exact normalized title · arXiv | Towards optimal doubly robust estimation of heterogeneous causal effects |
| [oprescu2019orthogonal](https://arxiv.org/abs/1806.03467) | Exact normalized title · arXiv | Orthogonal Random Forest for Causal Inference |
| [curth2021nonparametric](https://arxiv.org/abs/2101.10943) | Exact normalized title · arXiv | Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms |
| [curth2021inductive](https://arxiv.org/abs/2106.03765) | Exact normalized title · arXiv | On Inductive Biases for Heterogeneous Treatment Effect Estimation |
| [shalit2017estimating](https://arxiv.org/abs/1606.03976) | Exact normalized title · arXiv | Estimating individual treatment effect: generalization bounds and algorithms |
| [yoon2018ganite](https://openreview.net/forum?id=ByKWUeWA-) | No machine-readable title obtained | — |
| [caron2022shrinkage](https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2067549) | No machine-readable title obtained | — |
| [athey2021policy](https://arxiv.org/abs/1702.02896) | Exact normalized title · arXiv | Policy Learning with Observational Data |
| [chernozhukov2022riesznet](https://arxiv.org/abs/2110.03031) | Exact normalized title · arXiv | RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests |
| [wager2024causalinf](https://web.stanford.edu/~swager/causal_inf_book.pdf) | No machine-readable title obtained | — |
| [cinelli2020omitted](https://academic.oup.com/jrsssb/article-abstract/82/1/39/7056023) | No machine-readable title obtained | — |
| [rubin1974estimating](https://psycnet.apa.org/doi/10.1037/h0037350) | No machine-readable title obtained | — |
| [holland1986statistics](https://www.tandfonline.com/doi/abs/10.1080/01621459.1986.10478354) | No machine-readable title obtained | — |
| [vanderweele2017evalue](https://www.acpjournals.org/doi/10.7326/M16-2607) | No machine-readable title obtained | — |

## Second pass on the 48 entries without title metadata

The 48 rows yielded **38 scholarly matches, 5 project or data resources, 2 different works, 2 HTTP 404s and 1 alternate-only confirmation**. A scholarly match requires evidence for the stored page or its identifier. This includes a publisher-indexed page at the same article ID, a registered DOI, the title page of the file, or the paper under the same repository ID. The alternate-only row has an independent publication record but lacks confirmation for its literal stored path. The first-pass table above retains its original outcomes.

Two newly identified destinations need substantive corrections. The Neyman entry points to an issue on teaching English. The Croushore–Stark entry points to a related forecasting study by the same authors. The two 404s also have identified replacements, one of which is the Bojinov–Shephard paper on time-series randomization. The sites for FRED, EconML, GRF, tmle3 and DynamicDML document data or software. These five pages are appropriate for identifying those resources.

Across both passes, four stored links demonstrably lead to a different work. Two more returned 404, with independent confirmation of the intended source. Several accessible URLs lead to earlier or later versions than the corresponding ledger citation. The table records source versions where the difference was visible.

| Stored ledger entry | Second-pass result | Evidence |
|---|---|---|
| [frisch1933partial](https://www.jstor.org/stable/1907330) | Scholarly match | JSTOR article 1907330 matches Frisch and Waugh. |
| [lovell1963seasonal](https://doi.org/10.1080/01621459.1963.10480682) | Scholarly match | Crossref’s title search returned the ledger title and the exact stored DOI. [DOI record](https://doi.org/10.1080/01621459.1963.10480682) |
| [robinson1988rootn](https://www.econometricsociety.org/publications/econometrica/1988/07/01/root-n-consistent-semiparametric-regression) | Scholarly match | The Econometric Society page returned Robinson’s title. [Article identifier](https://doi.org/10.2307/1912705) |
| [chernozhukov2018double](https://academic.oup.com/ectj/article/21/1/C1/5056401) | Scholarly match | Oxford’s indexed article has the same title and article ID 5056401. [Journal DOI](https://doi.org/10.1111/ectj.12097) |
| [newey1994asymptotic](https://www.jstor.org/stable/2951752) | Scholarly match | JSTOR article 2951752 matches Newey. |
| [neyman1959optimal](https://www.jstor.org/stable/i285003) | Different work | Stable ID i285003 is the March 1972 *TESOL Quarterly* issue. Neyman cites his 1959 Cramér volume chapter, pp. 213–234, in his later bibliography, reference 26. [Neyman’s own citation](https://statmodeling.stat.columbia.edu/wp-content/uploads/2024/07/Neyman-FrequentistProbabilityFrequentist-1977.pdf#page=35) |
| [robins1986newapproach](https://www.sciencedirect.com/science/article/pii/0270025586900886) | Scholarly match | Elsevier indexes Robins’s title under the stored PII 0270025586900886. [Journal DOI](https://doi.org/10.1016/0270-0255%2886%2990088-6) |
| [robins2000marginal](https://journals.lww.com/epidem/abstract/2000/09000/marginal_structural_models_and_causal_inference_in.11.aspx) | Alternate only | PubMed 10955408 and the publisher’s DOI identify Robins, Hernán and Brumback. The literal LWW path did not return a verifiable article page. [PubMed](https://pubmed.ncbi.nlm.nih.gov/10955408/) · [Publisher DOI](https://doi.org/10.1097/00001648-200009000-00011) |
| [naimi2017introduction](https://academic.oup.com/ije/article/46/2/756/2760169) | Scholarly match | Oxford indexes Naimi et al. at the stored article ID 2760169. [Oxford article](https://academic.oup.com/ije/article/46/2/756/2760169?searchresult=1) |
| [bojinov2019time](https://doi.org/10.1080/01621459.2018.1554455) | 404 | The DOI resolver and registry returned 404 for the stored DOI ending in 1554455. The authors and journal identify the correct ending as 1527225. [Article DOI](https://doi.org/10.1080/01621459.2018.1527225) · [Shephard’s publication page](https://shephard.scholars.harvard.edu/publications/time-series-experiments-and-causal-estimands-exact-randomization-tests-and) |
| [newey1987simple](https://www.jstor.org/stable/1913610) | Scholarly match | JSTOR article 1913610 matches Newey and West. |
| [andrews1991hac](https://www.jstor.org/stable/2938229) | Scholarly match | JSTOR article 2938229 matches Andrews. |
| [deprado2018advances](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) | Scholarly match | The publisher identifies López de Prado’s book under the stored product ISBN 9781119482086. [Wiley ISBN record](https://www.wiley.com/en-in/Advances%20in%20Financial%20Machine%20Learning-p-9781119482086) |
| [lalonde1986evaluating](https://www.jstor.org/stable/1806062) | Scholarly match | JSTOR article 1806062 matches LaLonde. |
| [poterba1995crowdout](https://www.sciencedirect.com/science/article/pii/004727279401462W) | Scholarly match | Elsevier’s PII 0047-2727(94)01462-W matches the stored compact PII and Poterba et al.’s paper. [Journal DOI](https://doi.org/10.1016/0047-2727%2894%2901462-W) |
| [fred](https://fred.stlouisfed.org/) | Project or data | Federal Reserve Bank of St. Louis economic data service. |
| [econml](https://github.com/py-why/EconML) | Project or data | PyWhy’s EconML source repository and software citation. |
| [athey2019grfpkg](https://github.com/grf-labs/grf) | Project or data | GRF project’s R package source repository. |
| [murphy2003dtr](https://academic.oup.com/jrsssb/article/65/2/331/7092852) | Scholarly match | Oxford indexes Murphy’s title and article ID 7092852. |
| [lazarus2018har](https://scholar.harvard.edu/files/stock/files/lazarus_lewis_stock_watson_har_inference_recommendations_for_practice_jbes_2018.pdf) | Scholarly match | Harvard indexes the stored PDF under Stock’s author site. The migrated author page also supplies the article. [Author’s publication page](https://stock.scholars.harvard.edu/publications/har-inference-recommendations-practice) |
| [andrews1992improved](https://elischolar.library.yale.edu/cowles-discussion-paper-series/1185/) | Scholarly match | Yale identifies the 1990 Andrews–Monahan Cowles discussion paper 942. The journal citation refers to the 1992 publication. |
| [mueller2014hac](https://www.princeton.edu/~umueller/HACtest.pdf) | Scholarly match | The Princeton PDF’s first page identifies Müller and the article title. |
| [newey1994automatic](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1994.pdf) | Scholarly match | The Wisconsin PDF’s scanned first page identifies Whitney K. Newey and Kenneth D. West, *Review of Economic Studies* 61 (1994), 631–653. |
| [kwiatkowski1992stationarity](https://www.sciencedirect.com/science/article/abs/pii/030440769290104Y) | Scholarly match | Elsevier’s PII 030440769290104Y matches the article by Kwiatkowski et al. [Journal DOI](https://doi.org/10.1016/0304-4076%2892%2990104-Y) |
| [phillips1988testing](https://academic.oup.com/biomet/article-abstract/75/2/335/292919) | Scholarly match | Oxford indexes Phillips and Perron at this article ID. [Oxford article](https://academic.oup.com/biomet/article-abstract/75/2/335/292919?login=false) |
| [imbens2003sensitivity](https://www.semanticscholar.org/paper/Sensitivity-to-Exogeneity-Assumptions-in-Program-Imbens/f71d13e53828097cf8cd718255fc5b6393809eb9) | Scholarly match | Semantic Scholar’s API maps the stored record ID to the exact Imbens title and journal DOI. [Registry record](https://api.semanticscholar.org/graph/v1/paper/f71d13e53828097cf8cd718255fc5b6393809eb9?fields=title,externalIds) · [AEA article](https://doi.org/10.1257/000282803321946921) |
| [diamond2013genetic](https://direct.mit.edu/rest/article/95/3/932/58101/Genetic-Matching-for-Estimating-Causal-Effects-A) | Scholarly match | The Diamond–Sekhon DOI redirects to MIT Press article ID 58101, the same ID as the stored URL. [Publisher DOI](https://doi.org/10.1162/REST_a_00318) · [Author PDF](https://www.jsekhon.com/papers/GenMatch.pdf) |
| [stock2018identification](https://onlinelibrary.wiley.com/doi/abs/10.1111/ecoj.12593) | Scholarly match | Wiley’s article page matches Stock and Watson. |
| [croushore2001realtime](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=244554) | Different work | SSRN 244554 is the forecasting study subtitled ‘Does Data Vintage Matter for Forecasting?’ The 2001 article listed in the ledger appears in *Journal of Econometrics* 105, 111–130. [Stored working paper DOI](https://doi.org/10.2139/ssrn.244554) · [Intended journal article](https://www.sciencedirect.com/science/article/pii/S0304407601000720) |
| [knaus2021heterogeneous](https://academic.oup.com/ectj/article-abstract/24/1/134/5854188) | Scholarly match | The Oxford article record matches Knaus et al.’s empirical Monte Carlo paper. |
| [knaus2022heterogeneous](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3029832) | Scholarly match | SSRN’s download for this exact abstract ID is the working paper with ‘Programmes’ in its title. The *Journal of Human Resources* version uses ‘Programs’. [SSRN document](https://papers.ssrn.com/sol3/Delivery.cfm/dp10961.pdf?abstractid=3029832) · [IZA working paper](https://www.iza.org/en/publications/dp/10961/heterogeneous-employment-effects-of-job-search-programmes-a-machine-learning-approach) |
| [baiardi2024valueadded](https://academic.oup.com/ectj/article/27/2/213/7602388) | Scholarly match | Oxford indexes the titled paper at the stored ID 7602388. |
| [ahrens2025modelaveraging](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3103) | Scholarly match | Wiley’s article page matches Ahrens et al. |
| [coyle2020tmle3](https://tlverse.org/tmle3/) | Project or data | The project’s R package reference site documents the extensible TMLE framework. |
| [causaltune2022](https://www.pywhy.org/causaltune/) | 404 | The stored PyWhy documentation path returned HTTP 404. The maintainers identify the software and link its current docs. [Official repository](https://github.com/py-why/causaltune) · [Maintainer-linked documentation](https://causaltune.readthedocs.io/) |
| [econmldynamicdml](https://econml.azurewebsites.net/spec/estimation/dynamic_dml.html) | Project or data | The old Azure site points to the PyWhy EconML DynamicDML documentation. [Current project page](https://www.pywhy.org/EconML/spec/estimation/dynamic_dml.html) |
| [davidson2003econometric](http://qed.econ.queensu.ca/ETM/ETM-davidson-mackinnon-2021.pdf) | Scholarly match | Queen’s indexes the same PDF path on HTTPS. The 2021 filename identifies the downloadable revision of Davidson and MacKinnon’s book. [Queen’s copy](https://econ.queensu.ca/ETM/ETM-davidson-mackinnon-2021.pdf) |
| [zhang2014confidence](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12026) | Scholarly match | Wiley’s journal record confirms Zhang and Zhang’s title. |
| [vanderlaan2006targeted](https://biostats.bepress.com/ucbbiostat/paper213/) | Scholarly match | The Berkeley working-paper record identifies Mark van der Laan and Daniel Rubin. |
| [robins1994estimation](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476818) | Scholarly match | The journal’s indexed DOI identifies the Robins, Rotnitzky and Zhao paper. [Article DOI](https://doi.org/10.1080/01621459.1994.10476818) |
| [bickel1998efficient](https://link.springer.com/book/9780387984735) | Scholarly match | Springer identifies the Bickel et al. book under the stored ISBN. |
| [yoon2018ganite](https://openreview.net/forum?id=ByKWUeWA-) | Scholarly match | The OpenReview PDF indexed for the same forum ID identifies GANITE as an ICLR 2018 paper. [Conference PDF](https://openreview.net/pdf?id=ByKWUeWA-) |
| [caron2022shrinkage](https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2067549) | Scholarly match | The authors’ institutional record carries the same title and DOI as the Taylor & Francis URL. [UCL record](https://discovery.ucl.ac.uk/id/eprint/10149945/) |
| [wager2024causalinf](https://web.stanford.edu/~swager/causal_inf_book.pdf) | Scholarly match | The Stanford PDF identifies Wager and the correct title. The retrieved draft is dated September 8, 2026. The ledger credits Wager (2024). |
| [cinelli2020omitted](https://academic.oup.com/jrsssb/article-abstract/82/1/39/7056023) | Scholarly match | Oxford’s article record matches Cinelli and Hazlett. |
| [rubin1974estimating](https://psycnet.apa.org/doi/10.1037/h0037350) | Scholarly match | The APA article identifier matches Rubin’s 1974 paper. [Article DOI](https://doi.org/10.1037/h0037350) |
| [holland1986statistics](https://www.tandfonline.com/doi/abs/10.1080/01621459.1986.10478354) | Scholarly match | The journal’s DOI record identifies Holland’s article. [Article DOI](https://doi.org/10.1080/01621459.1986.10478354) |
| [vanderweele2017evalue](https://www.acpjournals.org/doi/10.7326/M16-2607) | Scholarly match | PubMed matches the DOI in the publisher URL, the title and authors. [PubMed](https://pubmed.ncbi.nlm.nih.gov/28693043/) |
