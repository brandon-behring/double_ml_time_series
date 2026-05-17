# Time-series inference machinery for DML

HAC / Newey-West / Driscoll-Kraay covariance estimators, fixed-b asymptotics (Kiefer-Vogelsang), HAR / long-run-variance inference (Lazarus–Lewis–Stock–Watson), time-series cross-validation (Bergmeir–Hyndman; purged CV), stationarity diagnostics (KPSS, ADF, Phillips–Perron), and Romano–Wolf step-down testing.

**Coverage:** 20 entries.
**Anchor prefix:** `E` (this file is `05_inference_ts.md`).

## E1. Time-series inference machinery for DML

- **Distribution of the Estimators for Autoregressive Time Series with a Unit Root** — David A. Dickey, Wayne A. Fuller; Journal of the American Statistical Association 74(366a):427-431
  - **Source:** https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10482531
  - **Code:** —
  - **Mechanism:** Derives non-standard limiting distribution for OLS coefficient in AR(1) under unit-root null.
  - **Result:** Tabulates critical values for the canonical ADF test of stationarity vs unit root; founding unit-root test.
  - **Status:** Verified (no widely-known repo)

- **A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix** — Newey & West (1987); Econometrica 1987
  - **Source:** https://www.jstor.org/stable/1913610
  - **Code:** —
  - **Mechanism:** Newey–West HAC: positive-semi-definite covariance estimator robust to heteroskedasticity and autocorrelation.
  - **Result:** Canonical HAC formula — Bartlett-kernel-weighted sum of sample autocovariances.
  - **Status:** Verified (no widely-known repo)

- **Testing for a unit root in time series regression** — Peter C. B. Phillips, Pierre Perron; Biometrika 75(2):335-346
  - **Source:** https://academic.oup.com/biomet/article-abstract/75/2/335/292919
  - **Code:** —
  - **Mechanism:** Nonparametric correction for nuisance parameters allowing weakly dependent, heterogeneously distributed data.
  - **Result:** PP test extends Dickey-Fuller to a broader class of error structures; distinguishes unit root from stationary trending behavior.
  - **Status:** Verified (no widely-known repo)

- **Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation** — Andrews (1991); Econometrica 1991
  - **Source:** https://www.jstor.org/stable/2938229
  - **Code:** —
  - **Mechanism:** Optimal kernel and bandwidth selection for HAC under MSE criterion.
  - **Result:** Data-dependent automatic bandwidth selection (QS kernel).
  - **Status:** Verified (no widely-known repo)

- **An Improved Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimator** — Donald W. K. Andrews, J. Christopher Monahan; Econometrica 60(4):953-966
  - **Source:** https://elischolar.library.yale.edu/cowles-discussion-paper-series/1185/
  - **Code:** —
  - **Mechanism:** Prewhitened kernel HAC estimator using VAR prewhitening before applying kernel smoothing.
  - **Result:** Reduces bias and rescues over-rejection of kernel-HAC t-statistics; improves CI coverage at cost of higher variance.
  - **Status:** Verified (no widely-known repo)

- **Testing the null hypothesis of stationarity against the alternative of a unit root: How sure are we that economic time series have a unit root?** — Denis Kwiatkowski, Peter C. B. Phillips, Peter Schmidt, Yongcheol Shin; Journal of Econometrics 54(1-3):159-178
  - **Source:** https://www.sciencedirect.com/science/article/abs/pii/030440769290104Y
  - **Code:** —
  - **Mechanism:** LM-style test reversing the null/alternative of Dickey-Fuller — tests stationarity against unit-root alternative.
  - **Result:** Founding KPSS test; complement to ADF for confirmatory stationarity diagnostics.
  - **Status:** Verified (no widely-known repo)

- **Automatic Lag Selection in Covariance Matrix Estimation** — Whitney K. Newey, Kenneth D. West; Review of Economic Studies 61(4):631-653
  - **Source:** https://users.ssc.wisc.edu/~behansen/718/NeweyWest1994.pdf
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Consistent Covariance Matrix Estimation With Spatially Dependent Panel Data** — Driscoll & Kraay (1998); Review of Economics and Statistics 1998
  - **Source:** https://doi.org/10.1162/003465398557825
  - **Code:** —
  - **Mechanism:** Spatial-correlation-consistent SE for panel data via Newey-West-style cross-section averaging.
  - **Result:** Standard HAC robust to both serial and cross-sectional dependence under large-T asymptotics.
  - **Status:** Verified (no widely-known repo)

- **Consistent cross-validatory model-selection for dependent data: hv-block cross-validation** — Jeffrey S. Racine; Journal of Econometrics 99(1):39-61
  - **Source:** https://econpapers.repec.org/RePEc:eee:econom:v:99:y:2000:i:1:p:39-61
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Heteroskedasticity-Autocorrelation Robust Standard Errors Using The Bartlett Kernel Without Truncation** — Nicholas M. Kiefer, Timothy J. Vogelsang; Econometrica 70(5):2093-2095
  - **Source:** https://ideas.repec.org/p/ecl/corcae/01-13.html
  - **Code:** —
  - **Mechanism:** Bartlett-kernel HAC estimator without truncation; inconsistent but asymptotically valid for testing.
  - **Result:** Tests equivalent to prior Kiefer-Vogelsang-Bunzel HAR tests; founding fixed-b HAC reference.
  - **Status:** Verified (no widely-known repo)

- **A New Asymptotic Theory for Heteroskedasticity-Autocorrelation Robust Tests** — Kiefer & Vogelsang (2005); Econometric Theory 2005
  - **Source:** https://doi.org/10.1017/S0266466605050516
  - **Code:** —
  - **Mechanism:** Fixed-b asymptotics: bandwidth held proportional to sample size yields better finite-sample inference.
  - **Result:** Distribution depending on kernel and bandwidth, vs small-b asymptotic point mass.
  - **Status:** Verified (no widely-known repo)

- **Stepwise Multiple Testing as Formalized Data Snooping** — Joseph P. Romano, Michael Wolf; Econometrica 73(4):1237-1282
  - **Source:** https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2005.00615.x
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Optimal Bandwidth Selection in Heteroskedasticity-Autocorrelation Robust Testing** — Yixiao Sun, Peter C. B. Phillips, Sainan Jin; Econometrica 76(1):175-194
  - **Source:** https://onlinelibrary.wiley.com/doi/abs/10.1111/j.0012-9682.2008.00822.x
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Heteroskedasticity, autocorrelation, and spatial correlation robust inference in linear panel models with fixed-effects** — Timothy J. Vogelsang; Journal of Econometrics 166(2):303-319
  - **Source:** https://ideas.repec.org/a/eee/econom/v166y2012i2p303-319.html
  - **Code:** —
  - **Mechanism:** Fixed-b asymptotic theory for cluster vs Driscoll-Kraay panel SEs, including individual and time fixed effects.
  - **Result:** Fixed-b approximation much better than normal/chi-square for Driscoll-Kraay; practical for panels with two-way fixed effects.
  - **Status:** Verified (no widely-known repo)

- **HAC Corrections for Strongly Autocorrelated Time Series** — Ulrich K. Müller; Journal of Business & Economic Statistics 32(3):311-322
  - **Source:** https://www.princeton.edu/~umueller/HACtest.pdf
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Sensitivity Analysis in Observational Research: Introducing the E-Value** — VanderWeele and Ding (2017); Annals of Internal Medicine, 167(4), 268-274
  - **Source:** https://www.acpjournals.org/doi/10.7326/M16-2607
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **A note on the validity of cross-validation for evaluating autoregressive time series prediction** — Bergmeir, Hyndman & Koo (2018); Computational Statistics & Data Analysis 2018
  - **Source:** https://doi.org/10.1016/j.csda.2017.11.003
  - **Code:** —
  - **Mechanism:** Validity of K-fold cross-validation for AR time series with uncorrelated errors.
  - **Result:** Standard k-fold CV is valid for autoregressive models when errors are uncorrelated.
  - **Status:** Verified (no widely-known repo)

- **Advances in Financial Machine Learning** — de Prado (2018); Wiley 2018
  - **Source:** https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
  - **Code:** —
  - **Mechanism:** Textbook on ML for finance, including purged k-fold CV and combinatorial purged CV.
  - **Result:** Purging/embargo conventions for backtesting financial ML to prevent lookahead bias.
  - **Status:** Verified (no widely-known repo)

- **HAR Inference: Recommendations for Practice** — Eben Lazarus, Daniel J. Lewis, James H. Stock, Mark W. Watson; Journal of Business & Economic Statistics 36(4):541-559
  - **Source:** https://scholar.harvard.edu/files/stock/files/lazarus_lewis_stock_watson_har_inference_recommendations_for_practice_jbes_2018.pdf
  - **Code:** —
  - **Mechanism:** See abstract for details.
  - **Result:** See abstract for details.
  - **Status:** Verified (no widely-known repo)

- **Making sense of sensitivity: extending omitted variable bias** — Cinelli and Hazlett (2020); Journal of the Royal Statistical Society Series B, 82(1), 39-67
  - **Source:** https://academic.oup.com/jrsssb/article-abstract/82/1/39/7056023
  - **Code:** https://github.com/carloscinelli/sensemakr
  - **Mechanism:** Reparameterizes omitted-variable-bias formula via partial R-squared with treatment and outcome (robustness value).
  - **Result:** Scale-free OVB sensitivity reporting handling multiple confounders and nonlinearity; implemented in sensemakr R package.
  - **Status:** Verified
