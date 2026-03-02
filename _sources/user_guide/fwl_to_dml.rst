From FWL to Double Machine Learning
====================================

This tutorial traces the pedagogical progression from the Frisch-Waugh-Lovell
theorem through Robinson's semiparametric estimator to Double Machine Learning.
Each step resolves a limitation of the previous approach.

.. contents:: Contents
   :local:
   :depth: 2

----

1. The FWL Theorem
------------------

1.1 Statement
~~~~~~~~~~~~~

Consider the linear model:

.. math::

   Y = X\beta + T\theta + \epsilon

where :math:`T` is the treatment of interest and :math:`X` is a matrix of controls.
The **Frisch-Waugh-Lovell theorem** states that :math:`\hat\theta` can be obtained
equivalently by:

1. Regress :math:`Y` on :math:`X`, obtain residuals :math:`\tilde{Y} = M_X Y`
2. Regress :math:`T` on :math:`X`, obtain residuals :math:`\tilde{T} = M_X T`
3. Regress :math:`\tilde{Y}` on :math:`\tilde{T}`: the coefficient equals :math:`\hat\theta_{\text{OLS}}`

where :math:`M_X = I - X(X'X)^{-1}X'` is the annihilator (residual-maker) matrix.

1.2 Proof Sketch
~~~~~~~~~~~~~~~~

The OLS normal equations for the full model give:

.. math::

   \begin{pmatrix} X'X & X'T \\ T'X & T'T \end{pmatrix}
   \begin{pmatrix} \hat\beta \\ \hat\theta \end{pmatrix}
   =
   \begin{pmatrix} X'Y \\ T'Y \end{pmatrix}

From the second block equation:

.. math::

   T'T\hat\theta + T'X\hat\beta = T'Y

Substituting :math:`\hat\beta` from the first block and simplifying:

.. math::

   \hat\theta = (T'M_X T)^{-1}(T'M_X Y) = (\tilde{T}'\tilde{T})^{-1}(\tilde{T}'\tilde{Y})

This is precisely OLS of :math:`\tilde{Y}` on :math:`\tilde{T}`. The algebraic
equivalence holds **exactly** — not approximately.

.. tip::

   **Why FWL matters for causal inference**: Residualization isolates the variation
   in :math:`T` that is orthogonal to :math:`X`. If :math:`X` contains all
   confounders, this variation is "as-if random" — the identification strategy
   for selection-on-observables designs.

1.3 Worked Example: Linear DGP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the data generating process is truly linear, FWL recovers the exact treatment effect:

.. code-block:: python

   import numpy as np
   from src.dml.fwl import fwl_estimate, fwl_vs_ols_comparison

   np.random.seed(42)
   n = 1000
   X = np.random.randn(n, 3)
   T = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)
   Y = 2.0 * T + X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(n)

   result = fwl_estimate(Y, T, X)
   print(f"FWL estimate: {result.theta:.4f}")  # ≈ 2.0
   print(f"OLS estimate: {result.theta:.4f}")  # Identical (FWL theorem)

   # Verify algebraic equivalence
   comparison = fwl_vs_ols_comparison(Y, T, X)
   print(f"Max difference: {comparison.max_diff:.2e}")  # ~1e-14

----

2. When Linear Fails
---------------------

2.1 Nonlinear Confounding
~~~~~~~~~~~~~~~~~~~~~~~~~~

FWL assumes the nuisance relationship :math:`\E{Y \mid X}` is linear in :math:`X`.
When confounders enter nonlinearly, FWL is biased:

.. math::

   Y = \theta_0 T + \underbrace{\cos(X_1) + X_2^2}_{\text{nonlinear } g(X)} + \epsilon

Linear residualization cannot capture :math:`\cos(X_1)` or :math:`X_2^2`,
leaving confounding variation in the residuals. The resulting :math:`\hat\theta`
is biased because :math:`\tilde{Y}` still contains confounding signal.

.. code-block:: python

   # Nonlinear DGP — FWL will be biased
   X = np.random.randn(n, 5)
   T = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.randn(n) * 0.5
   Y = 2.0 * T + np.cos(X[:, 0]) + X[:, 1] ** 2 + np.random.randn(n) * 0.5

   result_fwl = fwl_estimate(Y, T, X)
   print(f"FWL estimate: {result_fwl.theta:.3f}")  # Biased — not 2.0

----

3. Robinson's Semiparametric Estimator
--------------------------------------

3.1 The Partially Linear Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robinson (1988) proposed the **partially linear model**:

.. math::

   Y = \theta_0 T + g_0(X) + \epsilon, \quad \E{\epsilon \mid X, T} = 0

   T = m_0(X) + \eta, \quad \E{\eta \mid X} = 0

where :math:`g_0(\cdot)` and :math:`m_0(\cdot)` are unknown functions estimated
nonparametrically. The key insight: replace linear projections with flexible
ML estimators.

3.2 Estimation Procedure
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Estimate :math:`\hat{g}(X) \approx \E{Y \mid X}` using ML (random forest, etc.)
2. Estimate :math:`\hat{m}(X) \approx \E{T \mid X}` using ML
3. Form residuals: :math:`\tilde{Y} = Y - \hat{g}(X)`, :math:`\tilde{T} = T - \hat{m}(X)`
4. Regress :math:`\tilde{Y}` on :math:`\tilde{T}` to obtain :math:`\hat\theta`

.. code-block:: python

   from src.dml.robinson import robinson_estimator
   from sklearn.ensemble import RandomForestRegressor

   result_robinson = robinson_estimator(
       Y, T, X,
       model_y=RandomForestRegressor(n_estimators=200, random_state=42),
       model_t=RandomForestRegressor(n_estimators=200, random_state=42),
   )
   print(f"Robinson estimate: {result_robinson.theta:.3f}")

3.3 The Overfitting Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robinson's estimator uses **in-sample** predictions :math:`\hat{g}(X_i)` where
observation :math:`i` was used to fit :math:`\hat{g}`. This creates
**regularization bias**:

.. math::

   \hat\theta_{\text{Robinson}} - \theta_0 =
   \underbrace{\frac{1}{n}\sum_i \tilde{T}_i \epsilon_i}_{\text{noise (vanishes)}}
   + \underbrace{\frac{1}{n}\sum_i \tilde{T}_i [g_0(X_i) - \hat{g}(X_i)]}_{\text{regularization bias}}

The second term does **not** vanish at :math:`\sqrt{n}` rate because in-sample
predictions :math:`\hat{g}(X_i)` overfit: they capture some of the noise
:math:`\epsilon_i`, creating a correlation between the residuals
:math:`\tilde{T}_i` and the estimation error :math:`g_0(X_i) - \hat{g}(X_i)`.

.. tip::

   **Intuition**: A random forest predicting :math:`Y` in-sample partially
   memorizes :math:`\epsilon_i`. When you subtract :math:`\hat{g}(X_i)` from
   :math:`Y_i`, you remove too much — biasing :math:`\tilde{Y}` and hence
   :math:`\hat\theta`.

----

4. Double Machine Learning
--------------------------

4.1 The Cross-Fitting Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chernozhukov et al. (2018) resolved the overfitting bias with **cross-fitting**
(sample splitting):

**Algorithm** (K-fold DML):

1. Partition :math:`\{1, \ldots, n\}` into :math:`K` folds :math:`I_1, \ldots, I_K`
2. For each fold :math:`k`:

   a. Fit :math:`\hat{g}^{(-k)}` on :math:`\{I_j : j \neq k\}` (all data except fold :math:`k`)
   b. Fit :math:`\hat{m}^{(-k)}` on :math:`\{I_j : j \neq k\}`
   c. For :math:`i \in I_k`: compute :math:`\tilde{Y}_i = Y_i - \hat{g}^{(-k)}(X_i)`,
      :math:`\tilde{T}_i = T_i - \hat{m}^{(-k)}(X_i)`

3. Pool all residuals: :math:`\hat\theta = (\sum_i \tilde{T}_i^2)^{-1} \sum_i \tilde{T}_i \tilde{Y}_i`

Because :math:`\hat{g}^{(-k)}` never sees observation :math:`i \in I_k`, the
regularization bias term vanishes.

4.2 Neyman Orthogonal Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DML uses the **Neyman orthogonal** moment condition:

.. math::

   \psi(W; \theta, \hat\eta) = (Y - g(X) - \theta T)(T - m(X))

where :math:`\hat\eta = (\hat{g}, \hat{m})` are the nuisance parameters.
Orthogonality means:

.. math::

   \left.\frac{\partial}{\partial t} \E{\psi(W; \theta_0, \eta_0 + t(\hat\eta - \eta_0))}\right|_{t=0} = 0

This ensures that first-order errors in nuisance estimation :math:`\hat\eta`
do not affect :math:`\hat\theta`. Combined with cross-fitting, DML achieves:

.. math::

   \sqrt{n}(\hat\theta - \theta_0) \overset{d}{\to} \N(0, \sigma^2)

where :math:`\sigma^2 = J^{-1} \Sigma J^{-1}` with :math:`J = \E{\tilde{T}^2}`
and :math:`\Sigma = \E{\psi^2}`.

4.3 Worked Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.dml import double_ml

   result_dml = double_ml(
       Y, T, X,
       model_y=RandomForestRegressor(n_estimators=200, random_state=42),
       model_t=RandomForestRegressor(n_estimators=200, random_state=42),
       n_folds=5,
   )

   print(f"DML estimate: {result_dml.theta:.3f}")
   print(f"SE: {result_dml.se:.3f}")
   print(f"95% CI: [{result_dml.ci_lower:.3f}, {result_dml.ci_upper:.3f}]")
   print(f"Covers true (2.0): {result_dml.ci_lower <= 2.0 <= result_dml.ci_upper}")

4.4 Influence Function Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DML standard errors come from the **influence function**:

.. math::

   \hat\sigma^2 = \frac{1}{n} \sum_{i=1}^n \hat\psi_i^2

where :math:`\hat\psi_i = \hat{J}^{-1}(Y_i - \hat{g}(X_i) - \hat\theta T_i)(T_i - \hat{m}(X_i))`.

This is the semiparametrically efficient variance — it achieves the
Cramér-Rao lower bound for the partially linear model.

----

5. Diagnostic Checklist
------------------------

After running DML, verify these conditions:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Diagnostic
     - What to Check
     - API
   * - First-stage :math:`R^2`
     - :math:`R^2_Y > 0.1`, :math:`R^2_T > 0.1`
     - ``result.r2_y``, ``result.r2_t``
   * - Residual independence
     - :math:`\text{Corr}(\tilde{Y}, \tilde{T})` should be near :math:`\hat\theta \cdot \text{Var}(\tilde{T})`
     - Check residual plots
   * - Coverage
     - 95% CI should cover true :math:`\theta_0` in ~95% of MC runs
     - Monte Carlo simulation
   * - Stability across folds
     - Per-fold :math:`\hat\theta_k` should not vary excessively
     - ``result.fold_estimates``
   * - Sensitivity
     - How large must unmeasured confounding be to overturn result?
     - :class:`~src.sensitivity.rosenbaum.RosenbaumBounds`

----

6. Full Worked Example: The Complete Progression
-------------------------------------------------

Putting it all together — one DGP, three estimators, clear winner:

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import RandomForestRegressor
   from src.dml.fwl import fwl_estimate
   from src.dml.robinson import robinson_estimator
   from src.dml import double_ml

   # DGP with nonlinear confounding
   np.random.seed(42)
   n = 2000
   X = np.random.randn(n, 5)
   T = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.randn(n) * 0.5
   Y = 2.0 * T + np.cos(X[:, 0]) + X[:, 1] ** 2 + np.random.randn(n) * 0.5

   TRUE_THETA = 2.0
   rf = lambda: RandomForestRegressor(n_estimators=200, random_state=42)

   # 1. FWL — biased (can't handle nonlinearity)
   r_fwl = fwl_estimate(Y, T, X)
   print(f"FWL:      θ = {r_fwl.theta:.3f}  (bias: {r_fwl.theta - TRUE_THETA:+.3f})")

   # 2. Robinson — less biased but overfits
   r_rob = robinson_estimator(Y, T, X, model_y=rf(), model_t=rf())
   print(f"Robinson: θ = {r_rob.theta:.3f}  (bias: {r_rob.theta - TRUE_THETA:+.3f})")

   # 3. DML — unbiased with valid inference
   r_dml = double_ml(Y, T, X, model_y=rf(), model_t=rf(), n_folds=5)
   print(f"DML:      θ = {r_dml.theta:.3f}  (bias: {r_dml.theta - TRUE_THETA:+.3f})")
   print(f"          SE = {r_dml.se:.3f}")
   print(f"          CI = [{r_dml.ci_lower:.3f}, {r_dml.ci_upper:.3f}]")

Expected output pattern: FWL shows substantial bias, Robinson shows moderate
bias, DML estimate is close to 2.0 with a confidence interval that covers it.

----

See Also
--------

- :doc:`/api/dml` — Full API reference for all estimators
- :doc:`time_series_dml` — Time series extensions (temporal CV, HAC SE)
- :doc:`production_pipeline` — Production deployment of DML models
