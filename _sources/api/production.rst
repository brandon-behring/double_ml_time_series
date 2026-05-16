Research Pipeline Utilities
===========================

Research/demo utilities for causal inference model versioning, monitoring, and
retraining concepts. The module path is still ``src.production`` for now, but
these APIs are not deployment-grade infrastructure.

.. contents:: Module Index
   :local:
   :depth: 1

----

Model Registry (``src.production.model_registry``)
---------------------------------------------------

Version control for DML models with per-fold nuisance model storage and a
demo promotion workflow.

.. automodule:: src.production.model_registry
   :members:
   :undoc-members: False
   :show-inheritance:

----

Causal Monitor (``src.production.causal_monitor``)
---------------------------------------------------

Monitoring helpers for causal inference assumptions: overlap, treatment
distribution, nuisance quality, and effect stability.

.. automodule:: src.production.causal_monitor
   :members:
   :undoc-members: False
   :show-inheritance:

----

Retrain Pipeline (``src.production.retrain_pipeline``)
------------------------------------------------------

Causal-specific retraining triggers and scheduling.

.. automodule:: src.production.retrain_pipeline
   :members:
   :undoc-members: False
   :show-inheritance:

----

Insurance DML Pipeline (``src.production.dml_pipeline``)
---------------------------------------------------------

End-to-end research/demo pipeline for insurance competitor pricing causal
inference.

.. automodule:: src.production.dml_pipeline
   :members:
   :undoc-members: False
   :show-inheritance:
