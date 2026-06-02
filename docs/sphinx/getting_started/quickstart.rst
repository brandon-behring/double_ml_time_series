Quickstart
==========

A short tour of the current executable public contract.

Cross-Sectional PLR DML
-----------------------

``double_ml`` estimates a scalar partially linear treatment effect with ordinary
cross-fitting. In this remediation milestone it should be treated as an
i.i.d.-style helper, not a temporal-CV estimator.

.. code-block:: python

   import numpy as np

   from dml_ts import double_ml

   rng = np.random.default_rng(42)
   n = 500
   X = rng.normal(size=(n, 5))
   T = X[:, 0] + rng.normal(size=n)
   Y = 2.0 * T + X[:, 1] ** 2 + rng.normal(size=n)

   result = double_ml(Y, T, X, n_folds=5, model="ridge", random_state=42)

   print(f"theta: {result.theta:.3f}")
   print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

Temporal PLR DML
----------------

``TemporalPLRDML`` estimates a scalar temporal PLR effect with lagged treatment
controls, temporal cross-fitting, and HAC inference. It does not estimate
period-specific ``theta_t`` effects.

.. code-block:: python

   import numpy as np

   from dml_ts import TemporalPLRDML

   rng = np.random.default_rng(42)
   n = 240
   time_index = np.arange(n)
   X = np.column_stack([rng.normal(size=n), np.sin(time_index / 12)])
   T = 0.4 * X[:, 0] + rng.normal(size=n)
   Y = 1.5 * T + X[:, 1] + rng.normal(size=n)

   model = TemporalPLRDML(
       n_lags=2,
       model_y="ridge",
       model_t="ridge",
       n_splits=4,
       gap=2,
       hac_bandwidth=6,
       random_state=42,
   )
   result = model.fit(Y, T, X, time_index=time_index)

   print(f"theta: {result.theta:.3f}")
   print(f"HAC SE: {result.se:.3f}")
   print(f"CV rows dropped: {result.dropped_initial_rows}")

Time-Series Cross-Validation
----------------------------

Use temporal blocking and purging directly when inspecting train/test geometry.

.. code-block:: python

   import numpy as np

   from dml_ts.dml import TimeSeriesCrossValidator

   X = np.arange(100).reshape(-1, 1)
   cv = TimeSeriesCrossValidator(n_splits=5, gap=3, purge_length=2)

   for train_idx, test_idx in cv.split(X):
       assert train_idx[-1] + 3 + 2 <= test_idx[0]

Synthetic Macro Controls
------------------------

.. code-block:: python

   from dml_ts.data import create_synthetic_fred_data

   macro = create_synthetic_fred_data(
       start_date="2018-01-01",
       end_date="2020-12-31",
       frequency="M",
       seed=42,
   )

   print(macro.data.shape)
   print(macro.data.columns.tolist())

Insurance DGP
-------------

.. code-block:: python

   from dml_ts.validation import create_insurance_dgp

   data = create_insurance_dgp(
       realism="moderate",
       n_periods=120,
       n_products=10,
       true_tau=-0.8,
       seed=42,
   )

   print(f"true effect: {data.true_params.tau}")
   print(f"observations: {len(data.Y)}")

Next Steps
----------

- :doc:`/user_guide/fwl_to_dml` - Mathematical tutorial: FWL to Robinson to DML
- :doc:`/user_guide/time_series_dml` - Temporal PLR, HAC, and temporal CV
- :doc:`/api/dml` - API reference
