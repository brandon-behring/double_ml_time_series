Research Pipeline Utilities
===========================

The ``src.production`` package contains research/demo utilities for registry,
monitoring, and retraining examples used in the book companion. It should not be
described as production deployment infrastructure without a separate hardening pass.

Current Scope
-------------

Implemented companion utilities include:

- :class:`~src.production.model_registry.DMLModelRegistry`
- :class:`~src.production.model_registry.DMLModelVersion`
- :class:`~src.production.causal_monitor.CausalMonitor`
- :class:`~src.production.retrain_pipeline.RetrainScheduler`
- :class:`~src.production.dml_pipeline.InsuranceDMLPipeline`

These utilities are useful for demonstrating how causal diagnostics might be
organized. They do not establish service reliability, model governance, security,
observability, deployment, rollback, or live data guarantees.

Registry Example
----------------

.. code-block:: python

   import tempfile

   import numpy as np
   from sklearn.linear_model import Ridge

   from src.production import DMLModelRegistry, DMLModelVersion

   rng = np.random.default_rng(42)
   X = rng.normal(size=(100, 3))
   T = X[:, 0] + rng.normal(size=100)
   Y = 2.0 * T + X[:, 1] + rng.normal(size=100)

   nuisance = {
       0: (
           Ridge().fit(X, T),
           Ridge().fit(X, Y),
       )
   }

   version = DMLModelVersion.create(
       model_type="double_ml",
       nuisance_models=nuisance,
       feature_names=["x0", "x1", "x2"],
       treatment_name="treatment",
       outcome_name="outcome",
       hyperparameters={"n_folds": 5},
       metrics={"theta": 2.0, "se": 0.1},
       metadata={"example": True},
   )

   with tempfile.TemporaryDirectory() as tmpdir:
       registry = DMLModelRegistry(base_path=tmpdir)
       version_id = registry.register(version)
       loaded = registry.get(version_id)
       print(loaded.model_type)

Monitoring Example
------------------

Use monitoring helpers as diagnostics, not as proof that a model is deployment-ready.

.. code-block:: python

   import numpy as np

   from src.production import CausalMonitor

   rng = np.random.default_rng(42)
   treatment_train = rng.normal(size=200)
   treatment_new = rng.normal(loc=0.2, size=200)

   monitor = CausalMonitor()

   shift = abs(treatment_train.mean() - treatment_new.mean()) / treatment_train.std()
   print(f"standardized treatment shift: {shift:.3f}")
   print(type(monitor).__name__)

Guardrails
----------

Before using these modules beyond examples, add:

- explicit data contracts and schema validation
- persistent registry/version-store tests
- model artifact compatibility checks
- online/offline metric definitions
- alert thresholds and retraining policies tied to real requirements
- deployment, rollback, and audit logging infrastructure

Until that work exists, docs and chapters should call this package "research
pipeline utilities" or "demo utilities," not production software.
