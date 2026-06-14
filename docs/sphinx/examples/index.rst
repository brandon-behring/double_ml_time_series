Examples
========

Runnable example scripts demonstrating the current companion-code workflows.
Each script is self-contained and can be executed directly.

.. contents:: Example Index
   :local:
   :depth: 1

----

FWL to DML Progression
-----------------------

Demonstrates the pedagogical progression: FWL fails on nonlinear confounding,
Robinson overfits without cross-fitting, DML recovers the true effect.

.. literalinclude:: ../../../examples/example_fwl_to_dml.py
   :language: python
   :caption: examples/example_fwl_to_dml.py

----

Time Series DML
---------------

TemporalPLRDML with temporal cross-validation and HAC standard errors
on autocorrelated data.

.. literalinclude:: ../../../examples/example_time_series_dml.py
   :language: python
   :caption: examples/example_time_series_dml.py

----

Dynamic G-Estimation
--------------------

Recursive dynamic g-estimation (``DynamicGEstimationDML``) recovering
period-specific blips ``theta_1..theta_m`` for a linear SNMM, with a gated
EconML ``DynamicDML`` cross-check.

.. literalinclude:: ../../../examples/example_dynamic_g_estimation.py
   :language: python
   :caption: examples/example_dynamic_g_estimation.py

----

Rolling-Window DML
------------------

``RollingWindowDML`` refits a partially-linear DML estimate on a sliding
window of the series to trace a time-varying treatment effect.

.. literalinclude:: ../../../examples/example_rolling_window_dml.py
   :language: python
   :caption: examples/example_rolling_window_dml.py

----

Panel DML
---------

``PanelDML`` with an individual fixed-effects (within) transformation and
cluster-robust standard errors on a synthetic panel.

.. literalinclude:: ../../../examples/example_panel_dml.py
   :language: python
   :caption: examples/example_panel_dml.py

----

Insurance DGP
-------------

Insurance pricing DGP at multiple realism levels — from simple linear
to fully nonlinear with temporal dynamics.

.. literalinclude:: ../../../examples/example_insurance_dgp.py
   :language: python
   :caption: examples/example_insurance_dgp.py

----

Sensitivity Analysis
--------------------

Rosenbaum bounds sensitivity analysis applied to DML results.

.. literalinclude:: ../../../examples/example_sensitivity.py
   :language: python
   :caption: examples/example_sensitivity.py

----

Research Pipeline Utilities
---------------------------

Registry and causal-monitoring utilities used as reproducible research/demo
code. This example is not a production deployment claim.

.. literalinclude:: ../../../examples/example_production_pipeline.py
   :language: python
   :caption: examples/example_production_pipeline.py
