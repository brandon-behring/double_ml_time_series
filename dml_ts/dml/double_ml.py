"""Double Machine Learning (DML) with Cross-Fitting.

This module implements the Chernozhukov et al. (2018) estimator of a
partially linear regression slope from independent observations.

Mathematical Foundation
-----------------------
The problem with naive ML in causal inference:

    1. Regularization bias: ML models shrink coefficients, biasing treatment effects
    2. Overfitting bias: In-sample predictions overfit, creating spurious correlations
    3. Standard errors: Don't account for nuisance estimation error

The estimator combines three steps:

    1. Neyman orthogonality makes the population score locally insensitive to
       first-order errors in the conditional means.
    2. Cross-fitting evaluates each observation using models trained on other folds.
    3. The sample variability of the estimated influence scores gives the
       standard error for independent observations under the DML assumptions.

The Algorithm
-------------
For the partially linear model Y = θ·T + g(X) + ε:

1. Split data into K folds
2. For each fold k:
   a. Train outcome model on all folds except k: m̂₋ₖ(X) ≈ E[Y|X]
   b. Train treatment model on all folds except k: ℓ̂₋ₖ(X) ≈ E[T|X]
   c. Predict on fold k: m̂₋ₖ(Xₖ), ℓ̂₋ₖ(Xₖ)
3. Compute residuals: Ỹᵢ = Yᵢ - m̂₋ᵢ(Xᵢ), T̃ᵢ = Tᵢ - ℓ̂₋ᵢ(Xᵢ)
4. Estimate θ̂ = Σ(Ỹᵢ·T̃ᵢ) / Σ(T̃ᵢ²)
5. Compute SE via influence function: ψᵢ = (Ỹᵢ - θ̂·T̃ᵢ)·T̃ᵢ / E[T̃²]

For nuisance errors a = m̂ - E[Y|X] and b = ℓ̂ - E[T|X], the change in the
population moment at the true slope is E[a b - θ b²], conditional on the
training sample. The first derivatives vanish at a = b = 0. For root-n
inference the remaining term must be small relative to 1/√n, and residual
treatment variation must be positive.

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment
and structural parameters. The Econometrics Journal, 21(1), C1-C68.

Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
Econometrica, 56(4), 931-954.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from temporalcv import ci_ordered, coverage_in_unit, finite_se

from ._results import ResultBase
from ._utils import (
    compute_r2,
    cross_fit_predictions,
    get_nuisance_model,
    theta_via_fwl,
    validate_lengths,
)


@dataclass(frozen=True, slots=True, eq=False)
class DMLResult(ResultBase):
    """Result container for Double ML estimation.

    Attributes
    ----------
    theta : float
        Fitted partially linear slope.
    se : float
        Standard error from independent-observation influence scores.
    t_stat : float
        t-statistic = theta / se.
    p_value : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    Y_residual : NDArray[np.float64]
        Outcome residuals (Ỹ = Y - m̂(X)).
    T_residual : NDArray[np.float64]
        Treatment residuals (T̃ = T - ℓ̂(X)).
    influence_scores : NDArray[np.float64]
        Influence function values for each observation.
    outcome_r2_cv : float
        Cross-validated R² of outcome model.
    treatment_r2_cv : float
        Cross-validated R² of treatment model.
    n_folds : int
        Number of cross-fitting folds used.

    Examples
    --------
    >>> import numpy as np
    >>> from dml_ts.dml.double_ml import double_ml
    >>> np.random.seed(0)
    >>> n = 200
    >>> X = np.random.randn(n, 3)
    >>> T = 0.5 * X[:, 0] + np.random.randn(n)
    >>> Y = 2.0 * T + X[:, 1] + np.random.randn(n)
    >>> result = double_ml(Y, T, X, model="ridge", n_folds=3)
    >>> isinstance(result.theta, float)
    True
    >>> result.n_folds
    3
    """

    theta: float
    se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    Y_residual: NDArray[np.float64]
    T_residual: NDArray[np.float64]
    influence_scores: NDArray[np.float64]
    outcome_r2_cv: float
    treatment_r2_cv: float
    n_folds: int

    def _validate(self) -> None:
        """B3 numeric hard-fails at the result boundary."""
        if not np.isfinite(self.theta):
            raise ValueError(f"DMLResult.theta is not finite: {self.theta!r}")
        finite_se(self.se, name="DMLResult.se")
        ci_ordered(self.ci_lower, self.ci_upper, name="DMLResult.ci")
        coverage_in_unit(self.p_value, name="DMLResult.p_value")

    def __repr__(self) -> str:
        return (
            f"DMLResult(θ={self.theta:.4f}, SE={self.se:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4f})"
        )

    def summary(self) -> str:
        """Return formatted summary of DML results."""
        return f"""
Double Machine Learning Results
================================
Treatment Effect (θ):    {self.theta:.4f}
Standard Error:          {self.se:.4f}
t-statistic:             {self.t_stat:.2f}
p-value:                 {self.p_value:.4f}
95% Confidence Interval: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]

Nuisance Model Diagnostics:
  Outcome R² (CV):       {self.outcome_r2_cv:.3f}
  Treatment R² (CV):     {self.treatment_r2_cv:.3f}
  Number of folds:       {self.n_folds}

Interpretation:
  A one-unit increase in treatment is associated with a
  {self.theta:.4f} unit change in outcome (p={self.p_value:.4f}).
"""


def _influence_function_se(
    Y_tilde: NDArray[np.float64],
    T_tilde: NDArray[np.float64],
    theta: float,
) -> tuple[float, NDArray[np.float64]]:
    """Compute influence function-based standard error.

    The influence function for DML is:
        ψᵢ = (Ỹᵢ - θ·T̃ᵢ)·T̃ᵢ / E[T̃²]

    The variance is:
        Var(θ̂) = Var(ψ) / n

    This estimates sampling uncertainty under independent observations and
    the orthogonal-score regularity conditions described by ``double_ml``.

    Parameters
    ----------
    Y_tilde : array, shape (n,)
        Outcome residuals.
    T_tilde : array, shape (n,)
        Treatment residuals.
    theta : float
        Treatment effect estimate.

    Returns
    -------
    se : float
        Standard error of theta.
    psi : array, shape (n,)
        Influence function values.
    """
    n = len(Y_tilde)
    T_tilde_sq_mean = np.mean(T_tilde**2)

    # Influence function
    psi = (Y_tilde - theta * T_tilde) * T_tilde / T_tilde_sq_mean

    # Variance of theta is variance of influence function divided by n
    var_theta = np.var(psi, ddof=1) / n
    se = np.sqrt(var_theta)

    return se, psi


def double_ml(
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    X: NDArray[np.float64],
    n_folds: int = 5,
    model: BaseEstimator
    | Literal["ridge", "lasso", "random_forest", "gradient_boosting"]
    | None = None,
    outcome_model: BaseEstimator | None = None,
    treatment_model: BaseEstimator | None = None,
    alpha: float = 0.05,
    random_state: int = 42,
) -> DMLResult:
    """Double Machine Learning estimator with cross-fitting.

    Estimates the common slope θ in the partially linear model:
        Y = θ·T + g(X) + ε

    With an exogenous treatment and constant causal slope, θ gives the causal
    effect per unit of treatment. For conditionally unconfounded binary treatment
    with heterogeneous effects τ(X), the slope equals
    E[Var(T|X) τ(X)] / E[Var(T|X)]. The population ATE averages τ(X) with the
    population distribution of X, which can give a different value.

    Parameters
    ----------
    Y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable (continuous or binary).
    X : array, shape (n, p)
        Control variables / confounders.
    n_folds : int
        Number of cross-fitting folds. Default: 5.
    model : sklearn estimator or str, optional
        Model for both nuisance functions. Overridden by specific models.
        Options: "ridge", "lasso", "random_forest", "gradient_boosting"
        Default: "random_forest"
    outcome_model : sklearn estimator, optional
        Model for E[Y|X]. Overrides `model` for outcome.
    treatment_model : sklearn estimator, optional
        Model for E[T|X]. Overrides `model` for treatment.
        For either nuisance, a sklearn classifier must provide `predict_proba`.
        Its probability-weighted numeric class values estimate the conditional
        mean. For a 0/1 target this gives the probability of class 1.
    alpha : float
        Significance level for confidence intervals. Default: 0.05.
    random_state : int
        Random seed for cross-validation fold splitting.

    Returns
    -------
    DMLResult
        Contains theta, SE, confidence interval, residuals, and diagnostics.

    Notes
    -----
    The score uses held-out residuals for both conditional means. Its population
    derivative with respect to those means vanishes at their true values.
    Second-order nuisance errors can still bias the slope. The influence-function
    interval assumes independent observations, positive residual treatment
    variation, suitable moments, and nuisance convergence fast enough for the
    second-order errors to be negligible at the root-n scale.

    Examples
    --------
    >>> import numpy as np
    >>> from dml_ts.dml.double_ml import double_ml
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 3)
    >>> T = np.sin(X[:, 0]) + np.random.randn(n)
    >>> Y = 2.0 * T + np.exp(X[:, 1]) + np.random.randn(n)
    >>> result = double_ml(Y, T, X, model="ridge", n_folds=3)
    >>> isinstance(result.theta, float)
    True
    >>> bool(result.ci_lower < result.theta < result.ci_upper)
    True
    """
    # Validate inputs
    validate_lengths(Y, T, X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Set up models
    if model is None:
        model = "random_forest"

    if isinstance(model, str):
        model_type = cast(Literal["ridge", "lasso", "random_forest", "gradient_boosting"], model)
        # random_state=42 preserves this estimator's historical hardcoded
        # nuisance seed (independent of the fold-shuffling random_state).
        base_model = get_nuisance_model(model_type, random_state=42)
    else:
        base_model = model

    if outcome_model is None:
        outcome_model = clone(base_model)
    if treatment_model is None:
        treatment_model = clone(base_model)

    # Step 1-2: Cross-fit nuisance models
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    Y_hat, T_hat = cross_fit_predictions(X, Y, T, kf.split(X), outcome_model, treatment_model)

    # Compute cross-validated R²
    outcome_r2_cv = compute_r2(Y, Y_hat)
    treatment_r2_cv = compute_r2(T, T_hat)

    # Step 3: Compute residuals
    Y_tilde = Y - Y_hat
    T_tilde = T - T_hat

    # Step 4: Estimate theta
    theta, _ = theta_via_fwl(
        Y_tilde,
        T_tilde,
        treatment=T,
        no_variation_msg=(
            "Treatment has no variation after controlling for X. "
            "This may indicate perfect prediction of T by X."
        ),
    )

    # Step 5: Compute influence function-based SE
    se, influence_scores = _influence_function_se(Y_tilde, T_tilde, theta)

    # Compute inference statistics
    t_stat = theta / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = theta - z_crit * se
    ci_upper = theta + z_crit * se

    return DMLResult(
        theta=theta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        Y_residual=Y_tilde,
        T_residual=T_tilde,
        influence_scores=influence_scores,
        outcome_r2_cv=outcome_r2_cv,
        treatment_r2_cv=treatment_r2_cv,
        n_folds=n_folds,
    )


def compare_robinson_vs_dml(
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    X: NDArray[np.float64],
    true_theta: float | None = None,
    n_folds: int = 5,
) -> dict:
    """Compare Robinson (no cross-fitting) vs DML (with cross-fitting).

    Compare estimates obtained with in-sample and held-out nuisance predictions.

    Parameters
    ----------
    Y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable.
    X : array, shape (n, p)
        Control variables.
    true_theta : float, optional
        True treatment effect (if known).
    n_folds : int
        Number of folds for DML.

    Returns
    -------
    dict
        Comparison results.
    """
    from .robinson import robinson_estimator

    robinson_result = robinson_estimator(Y, T, X, model="random_forest")
    dml_result = double_ml(Y, T, X, model="random_forest", n_folds=n_folds)

    result = {
        "robinson_theta": robinson_result.theta,
        "robinson_se": robinson_result.se,
        "dml_theta": dml_result.theta,
        "dml_se": dml_result.se,
        "dml_ci_lower": dml_result.ci_lower,
        "dml_ci_upper": dml_result.ci_upper,
        "robinson_result": robinson_result,
        "dml_result": dml_result,
    }

    if true_theta is not None:
        result["true_theta"] = true_theta
        result["robinson_bias"] = robinson_result.theta - true_theta
        result["dml_bias"] = dml_result.theta - true_theta
        result["dml_covers"] = dml_result.ci_lower <= true_theta <= dml_result.ci_upper

    return result


def demonstrate_cross_fitting_benefit(seed: int = 42, n_sims: int = 100) -> None:
    """Demonstrate that cross-fitting reduces overfitting bias.

    Runs Monte Carlo simulations comparing Robinson (in-sample) vs
    DML (cross-fitted) to show the bias reduction from cross-fitting.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("Robinson vs DML: The Cross-Fitting Advantage")
    print("=" * 70)
    print()
    print("Monte Carlo simulation comparing:")
    print("  - Robinson (in-sample nuisance predictions)")
    print("  - DML (cross-fitted nuisance predictions)")
    print()
    print("DGP: Y = 2.0·T + sin(X₁) + X₂² + ε")
    print("     T = 0.5·X₁ + X₂ + η")
    print()
    print("True θ = 2.0")
    print()

    from .robinson import robinson_estimator

    true_theta = 2.0
    n_obs = 500

    robinson_thetas_list: list[float] = []
    dml_thetas_list: list[float] = []
    dml_coverages: list[bool] = []

    for _i in range(n_sims):
        X = np.random.randn(n_obs, 3)
        T = 0.5 * X[:, 0] + X[:, 1] + np.random.randn(n_obs)
        Y = true_theta * T + np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.randn(n_obs)

        robinson_result = robinson_estimator(Y, T, X, model="random_forest")
        dml_result = double_ml(Y, T, X, model="random_forest", n_folds=5)

        robinson_thetas_list.append(robinson_result.theta)
        dml_thetas_list.append(dml_result.theta)
        dml_coverages.append(dml_result.ci_lower <= true_theta <= dml_result.ci_upper)

    robinson_thetas = np.array(robinson_thetas_list)
    dml_thetas = np.array(dml_thetas_list)

    print(f"Results ({n_sims} simulations):")
    print()
    print("                     Robinson       DML (5-fold)")
    print("                     --------       ------------")
    print(f"  Mean θ̂:           {np.mean(robinson_thetas):.4f}         {np.mean(dml_thetas):.4f}")
    print(
        f"  Bias:             {np.mean(robinson_thetas) - true_theta:.4f}         {np.mean(dml_thetas) - true_theta:.4f}"
    )
    print(f"  Std Dev:          {np.std(robinson_thetas):.4f}         {np.std(dml_thetas):.4f}")
    print(
        f"  RMSE:             {np.sqrt(np.mean((robinson_thetas - true_theta) ** 2)):.4f}         {np.sqrt(np.mean((dml_thetas - true_theta) ** 2)):.4f}"
    )
    print(f"  95% CI Coverage:  N/A*           {np.mean(dml_coverages):.1%}")
    print()
    print("  *The coverage column summarizes DML's independent-observation")
    print("   influence-score intervals across these simulated datasets.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_cross_fitting_benefit(n_sims=50)
