Temporal PLR DML
================

This guide describes the current implemented time-series workflow. The central
class is :class:`~dml_ts.dml.temporal_plr_dml.TemporalPLRDML`, which estimates a
scalar partially linear treatment effect using lagged treatment controls,
temporal cross-fitting, and HAC inference.

It is intentionally not documented as true Lewis-Syrgkanis dynamic g-estimation.
The current result object has a scalar ``theta`` and does not expose
period-specific ``theta_t`` effects.

Implemented Scope
-----------------

The current implementation supports:

- lagged treatment controls through ``n_lags``
- expanding-window temporal cross-fitting
- gap handling through ``gap``
- HAC/Newey-West-style inference via ``temporalcv.newey_west_se``
- explicit reporting of early rows excluded because temporal CV cannot produce
  out-of-fold predictions without training on future observations

The current implementation does not automatically enforce stationarity,
cointegration, overlap, or low-variance treatment checks. It may warn on cheap
diagnostics, but users must still justify those assumptions in the chapter or
analysis.

Temporal Cross-Validation
-------------------------

.. code-block:: python

   import numpy as np

   from dml_ts.dml import TimeSeriesCrossValidator

   X = np.arange(120).reshape(-1, 1)
   cv = TimeSeriesCrossValidator(n_splits=4, gap=3, purge_length=2)

   for train_idx, test_idx in cv.split(X):
       assert train_idx[-1] + 3 + 2 <= test_idx[0]

Expanding-window temporal CV naturally leaves early observations without
out-of-fold predictions. ``TemporalPLRDML`` excludes those rows from the final
residual regression and reports the count as ``result.dropped_initial_rows``.

TemporalPLRDML
--------------

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

   print(result.theta)
   print(result.se)
   print(result.dropped_initial_rows)

The fitted residual regression uses only rows with valid temporal out-of-fold
predictions. ``result.n_periods`` is therefore the number of rows used after
lag construction and temporal-CV filtering, not simply ``len(Y) - n_lags``.

Rolling Windows
---------------

``RollingWindowDML`` estimates a local scalar effect per window. It is useful as
a descriptive stability diagnostic; it should not be described as a formal
period-specific dynamic treatment-effect estimator.

.. code-block:: python

   from dml_ts import RollingWindowDML

   rolling = RollingWindowDML(
       window_size=100,
       step_size=20,
       model_y="ridge",
       model_t="ridge",
       random_state=42,
   )
   rolling.fit(Y, T, X, time_index=time_index)
   centers, theta_series, se_series = rolling.get_effects()

Panel DML
---------

``PanelDML`` applies fixed-effect transformations and then uses the same scalar
PLR DML machinery.

.. code-block:: python

   import numpy as np

   from dml_ts import PanelDML

   n_products = 10
   n_periods = 24
   product_id = np.repeat(np.arange(n_products), n_periods)
   time_id = np.tile(np.arange(n_periods), n_products)
   X_panel = np.random.default_rng(42).normal(size=(n_products * n_periods, 3))
   T_panel = X_panel[:, 0] + np.random.default_rng(43).normal(size=n_products * n_periods)
   Y_panel = 1.2 * T_panel + X_panel[:, 1] + np.random.default_rng(44).normal(
       size=n_products * n_periods
   )

   panel = PanelDML(
       fixed_effects="individual",
       cluster_se=True,
       model_y="ridge",
       model_t="ridge",
       random_state=42,
   )
   result = panel.fit(Y_panel, T_panel, X_panel, product_id, time_id)

Methodology Checklist
---------------------

Before making chapter or README claims from a temporal estimate, record:

- whether the treatment and outcome series are stationary or transformed
- whether nonstationary series are plausibly cointegrated or otherwise justified
- whether temporal CV geometry avoids look-ahead leakage for the target estimand
- whether treatment residual variation is sufficient after controls and lags
- whether nuisance diagnostics are reasonable under temporal CV
- which rows were dropped by lag construction and temporal-CV coverage

Deferred Work
-------------

The following are future work, not current implemented claims:

- recursive dynamic g-estimation
- period-specific ``theta_t`` effects from a DynamicDML class
- automatic stationarity or cointegration blocking gates
- temporal CV support inside ``double_ml``
- full optional EconML heterogeneity workflows
