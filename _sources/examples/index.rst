Examples
========

Runnable example scripts demonstrating key library workflows. Each script
is self-contained and can be executed directly.

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

DynamicDML with temporal cross-validation and HAC standard errors
on autocorrelated data.

.. literalinclude:: ../../../examples/example_time_series_dml.py
   :language: python
   :caption: examples/example_time_series_dml.py

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

Production Pipeline
-------------------

End-to-end production workflow: model registry, causal monitoring,
and retraining decisions.

.. literalinclude:: ../../../examples/example_production_pipeline.py
   :language: python
   :caption: examples/example_production_pipeline.py
