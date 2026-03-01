Production Causal Inference
===========================

Deploying causal models in production differs fundamentally from standard ML:
you cannot monitor treatment effect accuracy because counterfactuals are
unobservable. Instead, you must monitor the **conditions for identification**.

.. contents:: Contents
   :local:
   :depth: 2

----

1. The Fundamental Challenge
-----------------------------

1.1 Why Standard ML Monitoring Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In standard ML, you monitor prediction accuracy:

.. math::

   \text{RMSE}_t = \sqrt{\frac{1}{n_t} \sum_{i=1}^{n_t} (Y_i - \hat{Y}_i)^2}

When RMSE degrades, you retrain. This works because :math:`Y_i` is observable.

In causal inference, the quantity of interest is:

.. math::

   \theta = \E{Y(1) - Y(0)}

But you **never** observe both :math:`Y(1)` and :math:`Y(0)` for the same unit.
You cannot compute prediction error for the treatment effect. A model that
produces :math:`\hat\theta = 5.0` might be correct or wildly wrong — and you
cannot tell from the data alone.

1.2 What You Can Monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead, monitor the **conditions under which** :math:`\hat\theta` is valid:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Condition
     - What It Means
     - What Violation Implies
   * - **Overlap/Positivity**
     - :math:`0 < P(T=1 \mid X) < 1` for all :math:`X`
     - Some subgroups never receive treatment — extrapolation required
   * - **Treatment stability**
     - Distribution of :math:`T \mid X` is stable over time
     - Treatment assignment mechanism has changed
   * - **Nuisance quality**
     - First-stage models maintain :math:`R^2`
     - Confounding structure has shifted
   * - **Effect stability**
     - :math:`\hat\theta_t` does not drift
     - True effect may be changing (non-stationarity)

----

2. Four Monitoring Dimensions
-------------------------------

2.1 Overlap Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

Overlap (positivity) requires that all covariate values have non-degenerate
treatment probability. In practice, monitor the propensity score distribution:

.. math::

   \hat{e}(X_i) = \hat{P}(T_i = 1 \mid X_i)

**Alert conditions**:

- :math:`\hat{e}(X_i) < 0.02` or :math:`\hat{e}(X_i) > 0.98` for any :math:`i`
  → **practical positivity violation**
- Proportion of near-boundary scores increasing over time
  → **overlap degradation**
- For continuous treatments: density of :math:`T \mid X` approaching zero
  in some region

.. code-block:: python

   from src.production import CausalMonitor

   monitor = CausalMonitor(
       overlap_threshold=0.05,    # flag if P(T|X) < 0.05 or > 0.95
       overlap_fraction_limit=0.10,  # alert if >10% of obs violate
   )

   result = monitor.check_overlap(propensity_scores)
   if result.has_violations:
       print(f"Overlap violations: {result.n_violations} obs "
             f"({result.violation_fraction:.1%})")

2.2 Treatment Distribution Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even with good overlap, the treatment assignment mechanism may shift.
Monitor the marginal and conditional distribution of :math:`T`:

.. code-block:: python

   result = monitor.check_treatment_stability(
       T_train=T_historical,
       T_new=T_current,
       X_train=X_historical,
       X_new=X_current,
   )

   print(f"KS statistic: {result.ks_statistic:.4f}")
   print(f"Distribution shift detected: {result.shift_detected}")

2.3 Nuisance Model Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first-stage models :math:`\hat{g}(X)` and :math:`\hat{m}(X)` must
maintain predictive quality. If the covariate-outcome relationship changes,
nuisance models produce poor residuals, biasing :math:`\hat\theta`.

.. code-block:: python

   result = monitor.check_nuisance_quality(
       Y_new=Y_current,
       T_new=T_current,
       X_new=X_current,
       model_y=fitted_model_y,
       model_t=fitted_model_t,
   )

   print(f"R² (outcome model): {result.r2_y:.3f}")
   print(f"R² (treatment model): {result.r2_t:.3f}")
   if result.r2_y < 0.1:
       print("WARNING: Outcome model has degraded")

2.4 Effect Stability Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track :math:`\hat\theta` over time using rolling or expanding windows.
A persistent drift signals non-stationarity in the treatment effect:

.. code-block:: python

   result = monitor.check_effect_stability(
       theta_history=[1.5, 1.4, 1.6, 1.3, 0.8, 0.5],  # declining
       window_size=3,
   )

   print(f"Trend detected: {result.trend_detected}")
   print(f"Recent mean: {result.recent_mean:.3f}")
   print(f"Baseline mean: {result.baseline_mean:.3f}")

----

3. Model Versioning
--------------------

3.1 Why Causal Models Need Special Versioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard ML versioning stores one model artifact per version. DML requires
storing **per-fold nuisance models** because cross-fitting produces :math:`K`
distinct :math:`\hat{g}^{(k)}` and :math:`\hat{m}^{(k)}` models.

The :class:`~src.production.model_registry.DMLModelRegistry` stores:

- Fold-specific nuisance models :math:`\{(\hat{g}^{(k)}, \hat{m}^{(k)})\}_{k=1}^K`
- Cross-validation fold indices
- Treatment effect estimate :math:`\hat\theta` and standard error
- Training data hash (for reproducibility)
- Metadata: timestamp, hyperparameters, diagnostics

3.2 Version Lifecycle
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   candidate → validated → promoted → archived
       ↓           ↓          ↓
     failed     rejected   superseded

.. code-block:: python

   from src.production import DMLModelRegistry, DMLModelVersion

   registry = DMLModelRegistry(storage_path="./model_registry")

   # Register a new model version
   version = registry.register(
       theta=result.theta,
       se=result.se,
       nuisance_models=result.fold_models,
       fold_indices=result.fold_indices,
       metadata={"n_obs": len(Y), "n_folds": 5},
   )
   print(f"Registered: {version.version_id}")

   # Validate and promote
   registry.validate(version.version_id, validation_result)
   registry.promote(version.version_id)

   # Get current production model
   prod = registry.get_production_model()
   print(f"Production model: {prod.version_id}")
   print(f"θ = {prod.theta:.3f} ± {prod.se:.3f}")

----

4. Alert Interpretation
------------------------

4.1 When Alerts Fire
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 35 25 20

   * - Alert
     - Mathematical Condition
     - False Positive Risk
     - Action
   * - Overlap violation
     - :math:`\hat{e}(X_i) < \epsilon` for :math:`> p\%` of :math:`i`
     - Low if :math:`\epsilon` well-calibrated
     - Trim or reweight
   * - Treatment shift
     - KS test :math:`p < 0.01`
     - Moderate (multiple testing)
     - Investigate mechanism
   * - Nuisance degradation
     - :math:`R^2_{\text{new}} < 0.5 \cdot R^2_{\text{train}}`
     - Low
     - Retrain nuisance models
   * - Effect drift
     - :math:`|\hat\theta_{\text{recent}} - \hat\theta_{\text{baseline}}| > 2\sigma`
     - Moderate (noise)
     - Re-estimate with more data

4.2 False Positive Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple monitoring checks create a multiple testing problem. With 4 checks
run daily, the probability of at least one false alarm in a month:

.. math::

   P(\text{any false alarm}) = 1 - (1 - \alpha)^{4 \times 30}
   \approx 1 - (0.95)^{120} \approx 0.998

At :math:`\alpha = 0.05`, you will get false alarms almost daily.

**Mitigations**:

- Use stricter per-test :math:`\alpha` (Bonferroni: :math:`\alpha/4 = 0.0125`)
- Require consecutive violations before alerting (e.g., 3 of 5 days)
- Use adaptive thresholds calibrated to historical variation

----

5. Retraining Triggers
------------------------

5.1 Causal-Specific vs Standard ML Triggers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard ML retraining triggers (accuracy drop, data drift) are necessary
but not sufficient for causal models. Causal-specific triggers monitor the
identification conditions:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Trigger Type
     - Standard ML
     - Causal-Specific
   * - Accuracy-based
     - :math:`\text{RMSE} > \tau`
     - Not applicable (no ground truth)
   * - Data drift
     - :math:`P(X)` shift
     - :math:`P(T \mid X)` shift specifically
   * - Model degradation
     - Any model quality drop
     - **Nuisance model** quality drop
   * - Concept drift
     - :math:`P(Y \mid X)` change
     - :math:`\theta` stability over time
   * - Overlap-specific
     - N/A
     - Positivity violations increasing

.. code-block:: python

   from src.production import RetrainScheduler, RetrainTrigger

   scheduler = RetrainScheduler(
       triggers=[
           RetrainTrigger(
               name="overlap_degradation",
               check_fn=lambda m: m.check_overlap(scores).violation_fraction > 0.10,
               severity="high",
           ),
           RetrainTrigger(
               name="nuisance_degradation",
               check_fn=lambda m: m.check_nuisance_quality(Y, T, X, g, h).r2_y < 0.15,
               severity="high",
           ),
           RetrainTrigger(
               name="effect_drift",
               check_fn=lambda m: m.check_effect_stability(history).trend_detected,
               severity="medium",
           ),
       ],
       min_retrain_interval_days=7,
   )

   # Check if retraining is needed
   decision = scheduler.evaluate(monitor)
   if decision.should_retrain:
       print(f"Retrain triggered by: {decision.triggered_by}")
       print(f"Severity: {decision.severity}")

----

6. Worked Example: End-to-End Production Pipeline
---------------------------------------------------

Complete production workflow: configure, fit, monitor, and make retraining
decisions using the :class:`~src.production.dml_pipeline.InsuranceDMLPipeline`.

.. code-block:: python

   from src.production import (
       InsuranceDMLPipeline,
       PipelineConfig,
       DMLModelRegistry,
       CausalMonitor,
   )
   from src.validation import create_insurance_dgp, InsuranceDGPParams
   from sklearn.ensemble import GradientBoostingRegressor

   # 1. Generate training data
   params = InsuranceDGPParams(
       n_obs=1000,
       n_competitors=3,
       true_effect=1.5,
       realism_level="moderate",
   )
   train_data = create_insurance_dgp(params)

   # 2. Configure pipeline
   config = PipelineConfig(
       model_y=GradientBoostingRegressor(n_estimators=200),
       model_t=GradientBoostingRegressor(n_estimators=200),
       n_folds=5,
       cv_type="temporal",
       hac_bandwidth="auto",
   )

   # 3. Fit pipeline
   pipeline = InsuranceDMLPipeline(config)
   result = pipeline.fit(
       Y=train_data.Y,
       T=train_data.T,
       X=train_data.X,
   )

   print(f"Training estimate: θ = {result.theta:.3f} ± {result.se:.3f}")
   print(f"True effect: {params.true_effect}")

   # 4. Register model
   registry = DMLModelRegistry(storage_path="./model_registry")
   version = registry.register(
       theta=result.theta,
       se=result.se,
       nuisance_models=result.fold_models,
       fold_indices=result.fold_indices,
   )

   # 5. Monitor on new data
   new_data = create_insurance_dgp(params)
   monitor = CausalMonitor()

   overlap_check = monitor.check_overlap(result.propensity_scores)
   nuisance_check = monitor.check_nuisance_quality(
       new_data.Y, new_data.T, new_data.X,
       result.model_y, result.model_t,
   )

   print(f"\nMonitoring Results:")
   print(f"  Overlap violations: {overlap_check.violation_fraction:.1%}")
   print(f"  Nuisance R² (Y): {nuisance_check.r2_y:.3f}")
   print(f"  Nuisance R² (T): {nuisance_check.r2_t:.3f}")

   # 6. Retrain decision
   if nuisance_check.r2_y < 0.15 or overlap_check.violation_fraction > 0.10:
       print("\n⚠ Retraining recommended")
   else:
       print("\n✓ Model stable — no retraining needed")

----

7. Architecture Summary
-------------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────┐
   │                Production Pipeline                   │
   │                                                      │
   │  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
   │  │ Pipeline │───→│ Model        │───→│ Causal     │ │
   │  │ Config   │    │ Registry     │    │ Monitor    │ │
   │  └──────────┘    └──────────────┘    └────────────┘ │
   │       │               │                    │         │
   │       │          ┌────┴─────┐         ┌────┴─────┐  │
   │       │          │ Version  │         │ Retrain  │  │
   │       │          │ Control  │         │ Scheduler│  │
   │       │          └──────────┘         └──────────┘  │
   │       │                                              │
   │  ┌────┴──────────────────────────────────────────┐  │
   │  │  DML Estimator (cross-fitted, temporal CV)    │  │
   │  │  ├── Fold 1: ĝ₁, m̂₁                         │  │
   │  │  ├── Fold 2: ĝ₂, m̂₂                         │  │
   │  │  └── Fold K: ĝ_K, m̂_K                       │  │
   │  └───────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────┘

----

See Also
--------

- :doc:`fwl_to_dml` — Foundation: understanding DML estimation
- :doc:`time_series_dml` — Time series extensions for temporal data
- :doc:`/api/production` — Full API reference
