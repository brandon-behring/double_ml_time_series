Time Series DML
===============

Standard cross-fitting assumes i.i.d. data. Time series introduces autocorrelation,
non-stationarity, and temporal confounding — each requiring methodological adaptations.
This tutorial covers the full suite of time series extensions.

.. contents:: Contents
   :local:
   :depth: 2

----

1. Why Standard Cross-Fitting Fails
-------------------------------------

1.1 The Temporal Leakage Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard K-fold cross-validation randomly assigns observations to folds.
With autocorrelated data, this creates **information leakage**: training folds
contain observations temporally adjacent to test observations, allowing the
model to "peek" at near-future information.

Consider an AR(1) process :math:`X_t = \rho X_{t-1} + \epsilon_t` with
:math:`\rho = 0.8`. Under random K-fold:

.. math::

   \text{Train fold contains } X_{t-1}, X_{t+1} \text{ when predicting } X_t

Since :math:`\text{Corr}(X_t, X_{t+1}) = \rho = 0.8`, the model can exploit
temporal neighbors in the training set, producing overly optimistic in-sample
nuisance estimates. This inflates first-stage :math:`R^2` and biases
:math:`\hat\theta`.

1.2 Bias Under Temporal Leakage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DML estimator under temporal leakage:

.. math::

   \hat\theta_{\text{leaked}} - \theta_0 \approx
   \frac{\sum_t \tilde{T}_t [g_0(X_t) - \hat{g}_{\text{leaked}}(X_t)]}
   {\sum_t \tilde{T}_t^2}

With leakage, :math:`\hat{g}_{\text{leaked}}(X_t)` is too accurate — it captures
autocorrelation structure rather than the true conditional expectation. The
nuisance estimation error is correlated with the score, reintroducing the
regularization bias that cross-fitting was designed to eliminate.

.. tip::

   **Key insight**: Cross-fitting eliminates regularization bias only when
   :math:`\hat{g}^{(-k)}(X_i)` is independent of :math:`\epsilon_i` for
   :math:`i \in I_k`. Temporal leakage violates this independence because
   autocorrelated :math:`\epsilon_t` makes :math:`\epsilon_{t-1}` (in the
   training set) informative about :math:`\epsilon_t` (in the test set).

----

2. Temporal Cross-Validation
-----------------------------

2.1 TimeSeriesCrossValidator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~src.dml.cross_fitting.TimeSeriesCrossValidator` enforces temporal
ordering with three safeguards:

**Blocking**: Folds respect temporal order — all training observations precede
all test observations. No future information leaks backward.

**Purging**: Remove :math:`p` observations at the train-test boundary:

.. math::

   \text{Train: } \{1, \ldots, t_{\text{split}} - p\}, \quad
   \text{Test: } \{t_{\text{split}}, \ldots, t_{\text{split}} + h\}

This eliminates contamination from observations where :math:`X_t` and
:math:`X_{t+k}` (for small :math:`k`) share autocorrelation.

**Gap**: An additional gap :math:`g` between purged training data and test data,
useful when the autocorrelation horizon exceeds the purge window.

.. code-block:: python

   from src.dml import TimeSeriesCrossValidator

   cv = TimeSeriesCrossValidator(
       n_splits=5,
       test_size=50,
       gap=10,       # 10-period gap
       purge=5,      # purge 5 boundary observations
   )

   # Fold structure (expanding window):
   # Fold 0: Train [0, ..., 140] | Gap [141-150] | Purge [146-150] | Test [151-200]
   # Fold 1: Train [0, ..., 190] | Gap [191-200] | Purge [196-200] | Test [201-250]
   # ...

2.2 BlockedTimeSeriesCV
~~~~~~~~~~~~~~~~~~~~~~~~

For settings where you want non-overlapping, fixed-size blocks rather than
an expanding window:

.. code-block:: python

   from src.dml import BlockedTimeSeriesCV

   cv_blocked = BlockedTimeSeriesCV(
       n_splits=5,
       gap=10,
   )

   # Each fold uses a fixed-size training block rather than all prior data

2.3 PurgedGroupTimeSeriesCV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For grouped/panel data where observations within the same group (entity)
must not be split across train and test:

.. code-block:: python

   from src.dml import PurgedGroupTimeSeriesCV

   cv_panel = PurgedGroupTimeSeriesCV(
       n_splits=5,
       group_gap=2,   # purge 2 group-periods at boundary
   )

----

3. HAC Standard Errors
-----------------------

3.1 Why Naive SE Fails
~~~~~~~~~~~~~~~~~~~~~~~

The standard DML variance estimator assumes i.i.d. scores:

.. math::

   \hat\sigma^2_{\text{naive}} = \frac{1}{n} \sum_{t=1}^n \hat\psi_t^2

Under autocorrelation, :math:`\text{Cov}(\hat\psi_t, \hat\psi_{t-j}) \neq 0`
for :math:`j > 0`. The naive estimator ignores these cross-terms, understating
the true variance and producing anti-conservative confidence intervals.

3.2 Newey-West HAC Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The HAC (Heteroskedasticity and Autocorrelation Consistent) estimator includes
autocovariance terms:

.. math::

   \hat\Sigma_{\text{HAC}} = \hat\Gamma_0
   + \sum_{j=1}^{b_n} w\!\left(\frac{j}{b_n}\right)
   (\hat\Gamma_j + \hat\Gamma_j')

where:

- :math:`\hat\Gamma_j = \frac{1}{n}\sum_{t=j+1}^n \hat\psi_t \hat\psi_{t-j}'`
  is the :math:`j`-th autocovariance
- :math:`w(x) = 1 - |x|` is the **Bartlett kernel** (ensures positive
  semi-definiteness)
- :math:`b_n` is the **bandwidth** (truncation lag)

3.3 Bandwidth Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

The bandwidth :math:`b_n` controls the bias-variance tradeoff:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Method
     - Formula
     - When to Use
   * - Fixed rule
     - :math:`b_n = \lfloor n^{1/3} \rfloor`
     - Quick default
   * - Andrews (1991)
     - :math:`b_n = 1.1447 (\hat\alpha n)^{1/3}`
     - Automatic, data-driven (recommended)
   * - Newey-West (1994)
     - Lag selection via information criteria
     - Short panels

.. code-block:: python

   from src.dml import newey_west_se, HACEstimator

   # Automatic bandwidth
   se_hac = newey_west_se(residuals, bandwidth="auto")

   # Fixed bandwidth
   se_fixed = newey_west_se(residuals, bandwidth=10)

   # Full HAC estimator with diagnostics
   hac = HACEstimator(kernel="bartlett", bandwidth="auto")
   result = hac.fit(residuals)
   print(f"HAC SE: {result.se:.4f}")
   print(f"Selected bandwidth: {result.bandwidth}")
   print(f"Effective df: {result.effective_df:.1f}")

3.4 Naive vs HAC Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from src.dml import newey_west_se

   # AR(1) residuals with rho = 0.7
   rho = 0.7
   n = 500
   eps = np.random.randn(n)
   residuals = np.zeros(n)
   for t in range(1, n):
       residuals[t] = rho * residuals[t - 1] + eps[t]

   se_naive = np.std(residuals) / np.sqrt(n)
   se_hac = newey_west_se(residuals, bandwidth="auto")

   print(f"Naive SE:  {se_naive:.4f}")
   print(f"HAC SE:    {se_hac:.4f}")
   print(f"Ratio:     {se_hac / se_naive:.2f}x")
   # HAC SE is typically 1.5-3x larger — naive SE is anti-conservative

----

4. DynamicDML
--------------

4.1 Time-Varying Treatment Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When treatment effects change over time, a single :math:`\hat\theta` is
misleading. **DynamicDML** estimates period-specific effects:

.. math::

   Y_t = \theta_t T_t + g(X_t, T_{t-1}, T_{t-2}, \ldots) + \epsilon_t

where :math:`\theta_t` varies with :math:`t` or with observable characteristics.
The key challenge is distinguishing time-varying effects from time-varying
confounding.

4.2 Sequential G-Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DynamicDML uses sequential g-estimation:

1. Start from the final period :math:`T`
2. Estimate :math:`\hat\theta_T` via DML
3. Subtract :math:`\hat\theta_T T_T` from :math:`Y_T`
4. Move backward: estimate :math:`\hat\theta_{T-1}` on adjusted outcomes
5. Continue until all periods estimated

This backward induction avoids the curse of dimensionality in the history
:math:`(T_{t-1}, T_{t-2}, \ldots)`.

.. code-block:: python

   from src.dml import DynamicDML
   from sklearn.ensemble import GradientBoostingRegressor

   dynamic = DynamicDML(
       model_y=GradientBoostingRegressor(n_estimators=100),
       model_t=GradientBoostingRegressor(n_estimators=100),
       n_folds=5,
       cv_type="temporal",   # use TimeSeriesCrossValidator
   )

   result = dynamic.fit(Y, T, X, time_index=time)
   print(f"Time-varying effects: {result.theta_t}")
   print(f"HAC-robust SE: {result.se_t}")

----

5. RollingWindowDML
--------------------

5.1 Local Estimation for Non-Stationarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the DGP itself is non-stationary (structural breaks, regime changes),
global estimation averages across regimes. **RollingWindowDML** estimates
:math:`\hat\theta` locally within a sliding window:

.. math::

   \hat\theta_t = \text{DML}(\{Y_s, T_s, X_s : s \in [t - w, t]\})

where :math:`w` is the window size.

5.2 Window Size Tradeoff
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Window
     - Advantage
     - Disadvantage
   * - Small (:math:`w < 100`)
     - Adapts quickly to regime changes
     - High variance, unstable estimates
   * - Large (:math:`w > 500`)
     - Low variance, stable
     - Slow adaptation, averages across regimes
   * - Moderate (:math:`w \approx 200`)
     - Balanced bias-variance
     - May miss sharp breaks

.. code-block:: python

   from src.dml import RollingWindowDML
   from sklearn.ensemble import RandomForestRegressor

   rolling = RollingWindowDML(
       model_y=RandomForestRegressor(n_estimators=100),
       model_t=RandomForestRegressor(n_estimators=100),
       window_size=200,
       step_size=50,     # evaluate every 50 periods
       n_folds=3,
   )

   result = rolling.fit(Y, T, X, time_index=time)
   print(f"Window centers: {result.window_centers}")
   print(f"Rolling estimates: {result.theta_windows}")

----

6. PanelDML
------------

6.1 Fixed Effects + Temporal Blocking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Panel data with entity :math:`i` and time :math:`t` introduces fixed effects:

.. math::

   Y_{it} = \theta T_{it} + g(X_{it}) + \alpha_i + \epsilon_{it}

where :math:`\alpha_i` captures time-invariant entity heterogeneity. PanelDML
combines within-entity demeaning with temporal cross-validation:

1. **Demean**: :math:`\ddot{Y}_{it} = Y_{it} - \bar{Y}_i`,
   :math:`\ddot{T}_{it} = T_{it} - \bar{T}_i`
2. **Temporal CV**: Split time periods (not observations) into folds
3. **DML**: Cross-fitted estimation on demeaned data

.. code-block:: python

   from src.dml import PanelDML
   from sklearn.ensemble import RandomForestRegressor

   panel = PanelDML(
       model_y=RandomForestRegressor(n_estimators=100),
       model_t=RandomForestRegressor(n_estimators=100),
       n_folds=5,
   )

   result = panel.fit(
       Y, T, X,
       entity_index=entities,
       time_index=times,
   )
   print(f"Panel DML estimate: {result.theta:.3f}")
   print(f"Entity fixed effects absorbed: {result.n_entities}")

----

7. Stationarity Diagnostics
-----------------------------

Before applying time series DML, verify stationarity of key variables.
Non-stationary series require differencing or detrending.

7.1 Test Suite
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 30 30 25

   * - Test
     - Null Hypothesis
     - Alternative
     - Interpretation
   * - ADF
     - Unit root (:math:`\rho = 1`)
     - Stationary
     - Reject → stationary
   * - KPSS
     - Trend-stationary
     - Unit root
     - Reject → non-stationary
   * - PP
     - Unit root (HAC-corrected)
     - Stationary
     - Reject → stationary

**Recommended practice**: Run ADF and KPSS jointly. If both agree, conclusion
is clear. If they disagree, the series may be near-integrated — consider
first-differencing to be safe.

.. code-block:: python

   from src.validation import StationarityDiagnostic

   diag = StationarityDiagnostic()
   result = diag.test(series, test="all")

   print(f"ADF p-value: {result.adf_pvalue:.4f}")
   print(f"KPSS p-value: {result.kpss_pvalue:.4f}")
   print(f"PP p-value: {result.pp_pvalue:.4f}")
   print(f"Conclusion: {result.conclusion}")

----

8. Worked Example: Full Time Series DML Pipeline
--------------------------------------------------

End-to-end example: generate autocorrelated data, validate stationarity,
estimate with temporal CV and HAC standard errors, compare to naive approach.

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import RandomForestRegressor
   from src.validation import TimeSeriesDGPGenerator
   from src.dml import double_ml, TimeSeriesCrossValidator, newey_west_se

   # 1. Generate AR(1) DGP with known treatment effect
   dgp = TimeSeriesDGPGenerator(
       n_obs=500,
       n_features=5,
       ar_coef=0.7,          # autocorrelation coefficient
       true_theta=1.5,
       noise_std=0.5,
       seed=42,
   )
   data = dgp.generate()
   Y, T, X = data.Y, data.T, data.X

   # 2. Verify stationarity
   from src.validation import StationarityDiagnostic
   diag = StationarityDiagnostic()
   for j in range(X.shape[1]):
       result = diag.test(X[:, j])
       assert result.adf_pvalue < 0.05, f"Feature {j} is non-stationary"

   # 3. DML with temporal cross-validation
   cv = TimeSeriesCrossValidator(n_splits=5, test_size=50, gap=10)
   rf = lambda: RandomForestRegressor(n_estimators=100, random_state=42)

   result = double_ml(Y, T, X, model_y=rf(), model_t=rf(), cv=cv)

   # 4. HAC standard errors
   se_hac = newey_west_se(result.residuals, bandwidth="auto")

   print(f"DML estimate: {result.theta:.3f} (true: 1.5)")
   print(f"Naive SE: {result.se:.4f}")
   print(f"HAC SE:   {se_hac:.4f}")
   print(f"Ratio:    {se_hac / result.se:.2f}x")

----

See Also
--------

- :doc:`fwl_to_dml` — Foundation: FWL → Robinson → DML
- :doc:`production_pipeline` — Deploying time series DML in production
- :doc:`/api/dml` — Full API reference
