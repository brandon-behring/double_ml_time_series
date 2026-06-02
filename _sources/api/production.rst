Research Pipeline Utilities
===========================

Research/demo utilities for causal inference model versioning, monitoring, and
retraining concepts. The module path is still ``dml_ts.production`` for now, but
these APIs are not deployment-grade infrastructure.

.. contents:: Module Index
   :local:
   :depth: 1

----

Model Registry (``dml_ts.production.model_registry``)
-----------------------------------------------------

Version control for DML models with per-fold nuisance model storage and a
demo promotion workflow.

.. automodule:: dml_ts.production.model_registry
   :members:
   :undoc-members: False
   :show-inheritance:

----

Causal Monitor (``dml_ts.production.causal_monitor``)
-----------------------------------------------------

Monitoring helpers for causal inference assumptions: overlap, treatment
distribution, nuisance quality, and effect stability.

.. automodule:: dml_ts.production.causal_monitor
   :members:
   :undoc-members: False
   :show-inheritance:

----

Retrain Pipeline (``dml_ts.production.retrain_pipeline``)
---------------------------------------------------------

Causal-specific retraining triggers and scheduling.

.. automodule:: dml_ts.production.retrain_pipeline
   :members:
   :undoc-members: False
   :show-inheritance:

----

Insurance DML Pipeline (``dml_ts.production.dml_pipeline``)
-----------------------------------------------------------

End-to-end research/demo pipeline for insurance competitor pricing causal
inference.

.. automodule:: dml_ts.production.dml_pipeline
   :members:
   :undoc-members: False
   :show-inheritance:
