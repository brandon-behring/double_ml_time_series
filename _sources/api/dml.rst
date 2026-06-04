DML Estimators
==============

Core causal inference estimators implementing the pedagogical progression from
Frisch-Waugh-Lovell through Double Machine Learning, plus temporal PLR helpers.

.. contents:: Module Index
   :local:
   :depth: 1

----

Frisch-Waugh-Lovell (``dml_ts.dml.fwl``)
----------------------------------------

The FWL theorem provides the algebraic foundation for all residualization-based
causal estimators. See :doc:`/user_guide/fwl_to_dml` for the mathematical derivation.

.. automodule:: dml_ts.dml.fwl
   :members:
   :undoc-members: False
   :show-inheritance:

----

Robinson Estimator (``dml_ts.dml.robinson``)
--------------------------------------------

Semiparametric partially linear model estimator. Extends FWL to nonparametric
nuisance functions :math:`g(X)` and :math:`m(X)`.

.. automodule:: dml_ts.dml.robinson
   :members:
   :undoc-members: False
   :show-inheritance:

----

Double Machine Learning (``dml_ts.dml.double_ml``)
--------------------------------------------------

Cross-fitted DML with Neyman orthogonal scores. Resolves the overfitting bias
of the Robinson estimator via sample splitting.

.. automodule:: dml_ts.dml.double_ml
   :members:
   :undoc-members: False
   :show-inheritance:

----

Temporal PLR DML (``dml_ts.dml.temporal_plr_dml``)
--------------------------------------------------

Scalar temporal partially linear DML with lagged treatment controls,
time-series cross-fitting, and HAC inference. This module also contains
rolling-window and panel companion estimators. It is not a true dynamic
g-estimation implementation.

.. automodule:: dml_ts.dml.temporal_plr_dml
   :members:
   :undoc-members: False
   :show-inheritance:

----

Time Series Cross-Validation (``dml_ts.dml.cross_fitting``)
-----------------------------------------------------------

Temporal cross-validation with blocking, purging, and gap controls to prevent
information leakage under autocorrelation.

.. automodule:: dml_ts.dml.cross_fitting
   :members:
   :undoc-members: False
   :show-inheritance:

----

HAC Standard Errors (``dml_ts.dml.hac``)
----------------------------------------

Heteroskedasticity and autocorrelation consistent (HAC) covariance estimation
via Newey-West with automatic bandwidth selection.

.. automodule:: dml_ts.dml.hac
   :members:
   :undoc-members: False
   :show-inheritance:

----

Recursive Dynamic G-Estimation (``dml_ts.dml.dynamic_g_estimation``)
-------------------------------------------------------------------------

Neyman-orthogonal recursive dynamic g-estimation (Lewis-Syrgkanis): recovers
period-specific treatment blips for linear structural nested mean models via
cross-fitted backward peeling, with the joint coupled-stage sandwich variance.

.. automodule:: dml_ts.dml.dynamic_g_estimation
   :members:
   :undoc-members: False
   :show-inheritance:
