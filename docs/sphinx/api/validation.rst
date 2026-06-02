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

Stationarity Diagnostics (``dml_ts.validation.stationarity``)
-------------------------------------------------------------

ADF, KPSS, and Phillips-Perron stationarity tests with
comprehensive diagnostic reporting.

.. automodule:: dml_ts.validation.stationarity
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
