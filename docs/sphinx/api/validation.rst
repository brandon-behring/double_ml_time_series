Validation Infrastructure
=========================

Data generating processes (DGPs) and diagnostic tools for validating
causal estimator performance under known ground truth.

.. contents:: Module Index
   :local:
   :depth: 1

----

Cross-Sectional DGP (``dml_ts.validation.dgp_generator``)
---------------------------------------------------------

Configurable data generating processes for i.i.d. cross-sectional
causal inference validation.

.. automodule:: dml_ts.validation.dgp_generator
   :members:
   :undoc-members: False
   :show-inheritance:

----

Dynamic-Treatment DGP (``dml_ts.validation.dynamic_dgp``)
------------------------------------------------------------

Panel and single-series data with known per-period treatment blips, for
validating recursive dynamic g-estimation against ground truth.

.. automodule:: dml_ts.validation.dynamic_dgp
   :members:
   :undoc-members: False
   :show-inheritance:

----

Time Series DGP (``dml_ts.validation.dgp_generator_ts``)
--------------------------------------------------------

Autocorrelated data generating processes with temporal confounding
for time series DML validation.

.. automodule:: dml_ts.validation.dgp_generator_ts
   :members:
   :undoc-members: False
   :show-inheritance:

----

Insurance DGP (``dml_ts.validation.insurance_dgp``)
---------------------------------------------------

Parameterized insurance pricing DGP with configurable realism levels
for competitor pricing causal inference.

.. automodule:: dml_ts.validation.insurance_dgp
   :members:
   :undoc-members: False
   :show-inheritance:

----

Bias Validation (``dml_ts.validation.bias_validation``)
-------------------------------------------------------

Bias measurement and validation tools for comparing estimator
performance against known treatment effects.

.. automodule:: dml_ts.validation.bias_validation
   :members:
   :undoc-members: False
   :show-inheritance:

----

Parallel Execution (``dml_ts.validation.parallel``)
---------------------------------------------------

Parallel execution utilities optimized for Monte Carlo simulations
on multi-core systems.

.. automodule:: dml_ts.validation.parallel
   :members:
   :undoc-members: False
   :show-inheritance:

----

OLS Baselines (``dml_ts.validation.ols_baseline``)
--------------------------------------------------

OLS baseline estimators: ``NaiveOLS`` (``Y ~ T`` only, exhibits confounding
bias) and ``OLSWithControls`` (``Y ~ T + X``), the standard-econometrics
reference points for DML.

.. automodule:: dml_ts.validation.ols_baseline
   :members:
   :undoc-members: False
   :show-inheritance:

----

IPW Baselines (``dml_ts.validation.ipw_baseline``)
--------------------------------------------------

Inverse-propensity-weighting baselines: ``IPWEstimator`` and the
doubly-robust ``AugmentedIPW``.

.. automodule:: dml_ts.validation.ipw_baseline
   :members:
   :undoc-members: False
   :show-inheritance:

----

ML Baselines (``dml_ts.validation.ml_baseline``)
------------------------------------------------

Machine-learning outcome-model baselines: ``RandomForestEstimator`` and
``XGBoostEstimator``.

.. automodule:: dml_ts.validation.ml_baseline
   :members:
   :undoc-members: False
   :show-inheritance:

----

Baseline Comparison (``dml_ts.validation.baseline_comparison``)
---------------------------------------------------------------

Unified harness that runs the OLS / IPW / ML baselines side by side on
shared data, for contrast against the DML estimators.

.. automodule:: dml_ts.validation.baseline_comparison
   :members:
   :undoc-members: False
   :show-inheritance:

----

Bootstrap Configuration (``dml_ts.validation.bootstrap_config``)
----------------------------------------------------------------

Standardized bootstrap configuration shared across bias validation and the
baseline estimators, for consistent resampling parameters.

.. automodule:: dml_ts.validation.bootstrap_config
   :members:
   :undoc-members: False
   :show-inheritance:

----

Bootstrap Diagnostics (``dml_ts.validation.bootstrap_diagnostics``)
-------------------------------------------------------------------

Bootstrap quality diagnostics: convergence as ``n_bootstrap`` grows,
distribution normality / symmetry, and Monte-Carlo variance stability.

.. automodule:: dml_ts.validation.bootstrap_diagnostics
   :members:
   :undoc-members: False
   :show-inheritance:

----

Enhanced DGP (``dml_ts.validation.enhanced_dgp``)
-------------------------------------------------

Robustness-testing DGP extending the basic generator with heterogeneous
treatment effects, misspecification scenarios, and propensity extremeness.

.. automodule:: dml_ts.validation.enhanced_dgp
   :members:
   :undoc-members: False
   :show-inheritance:

----

Empirical Replication (``dml_ts.validation.empirical_replication``)
-------------------------------------------------------------------

Empirical replication of the Chernozhukov et al. (2018) 401(k) DML analysis,
for real-data validation of the implementation.

.. automodule:: dml_ts.validation.empirical_replication
   :members:
   :undoc-members: False
   :show-inheritance:

----

Lasso Diagnostic (``dml_ts.validation.lasso_diagnostic``)
---------------------------------------------------------

Diagnostic tooling investigating the 401(k) Lasso ATE mismatch against
published results (bootstrap convergence, seed sensitivity, hyperparameter
analysis).

.. automodule:: dml_ts.validation.lasso_diagnostic
   :members:
   :undoc-members: False
   :show-inheritance:

----

Validation Result (``dml_ts.validation.validation_result``)
------------------------------------------------------------

Standardized ``ValidationResult`` container shared across all validation
methods, ensuring consistent reporting and storage.

.. automodule:: dml_ts.validation.validation_result
   :members:
   :undoc-members: False
   :show-inheritance:

----

Result Storage (``dml_ts.validation.storage``)
-----------------------------------------------

Storage and caching for ``ValidationResult`` objects, DGP configurations,
and simulation outputs.

.. automodule:: dml_ts.validation.storage
   :members:
   :undoc-members: False
   :show-inheritance:

----

Plotting Utilities (``dml_ts.validation.plotting``)
---------------------------------------------------

Consistent styling utilities (shared palette, fonts, layout) for validation
result plots.

.. automodule:: dml_ts.validation.plotting
   :members:
   :undoc-members: False
   :show-inheritance:
