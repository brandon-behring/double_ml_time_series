Quickstart
==========

A 5-minute tour of the core library capabilities.

Basic DML Estimation
--------------------

Estimate a causal treatment effect using cross-fitted Double Machine Learning:

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import RandomForestRegressor
   from src.dml import double_ml

   # Generate data with nonlinear confounding
   np.random.seed(42)
   n = 1000
   X = np.random.randn(n, 5)
   T = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.randn(n) * 0.5
   Y = 2.0 * T + np.cos(X[:, 0]) + X[:, 1] ** 2 + np.random.randn(n) * 0.5

   # Estimate with DML (true effect = 2.0)
   result = double_ml(
       Y, T, X,
       model_y=RandomForestRegressor(n_estimators=100, random_state=42),
       model_t=RandomForestRegressor(n_estimators=100, random_state=42),
       n_folds=5,
   )

   print(f"Estimate: {result.theta:.3f} (true: 2.0)")
   print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

Time Series Cross-Validation
-----------------------------

Use temporal blocking to prevent information leakage with autocorrelated data:

.. code-block:: python

   from src.dml import TimeSeriesCrossValidator

   cv = TimeSeriesCrossValidator(
       n_splits=5,
       test_size=50,
       gap=10,        # 10-period gap between train and test
       purge=5,        # purge 5 periods at boundary
   )

   for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
       print(f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
       # Train and test indices respect temporal ordering

HAC Standard Errors
-------------------

Newey-West standard errors for autocorrelated residuals:

.. code-block:: python

   from src.dml import newey_west_se

   # Compute HAC-robust standard errors
   se = newey_west_se(residuals, bandwidth="auto")
   print(f"HAC SE: {se:.4f}")

FRED Macroeconomic Controls
----------------------------

Load macroeconomic time series for use as controls:

.. code-block:: python

   from src.data import create_synthetic_fred_data

   # Synthetic data for offline development
   macro_data = create_synthetic_fred_data(n_periods=120)
   print(f"Series: {list(macro_data.columns)}")
   print(f"Shape: {macro_data.shape}")

Insurance DGP
-------------

Generate insurance pricing data with known causal effects:

.. code-block:: python

   from src.validation import create_insurance_dgp, InsuranceDGPParams

   params = InsuranceDGPParams(
       n_obs=500,
       n_competitors=3,
       true_effect=1.5,
       realism_level="moderate",
   )
   result = create_insurance_dgp(params)

   print(f"True effect: {params.true_effect}")
   print(f"Y shape: {result.Y.shape}")
   print(f"T shape: {result.T.shape}")

Next Steps
----------

- :doc:`/user_guide/fwl_to_dml` — Mathematical tutorial: FWL → Robinson → DML
- :doc:`/user_guide/time_series_dml` — Time series extensions with HAC and temporal CV
- :doc:`/api/dml` — Full API reference
