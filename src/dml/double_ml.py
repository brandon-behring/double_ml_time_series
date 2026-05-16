"""Double Machine Learning (DML) with Cross-Fitting.

This module implements the Chernozhukov et al. (2018) Double ML estimator,
which achieves √n-consistency for treatment effects even with slow-rate
ML nuisance estimators.

Mathematical Foundation
-----------------------
The problem with naive ML in causal inference:

    1. Regularization bias: ML models shrink coefficients, biasing treatment effects
    2. Overfitting bias: In-sample predictions overfit, creating spurious correlations
    3. Standard errors: Don't account for nuisance estimation error

DML solves all three via:

    1. Neyman orthogonality: Moment condition is insensitive to small nuisance errors
    2. Cross-fitting: Out-of-sample predictions eliminate overfitting bias
    3. Influence functions: Valid inference without nuisance rate requirements

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

The critical insight: By using out-of-fold predictions, E[nuisance_error × score] = 0
even when nuisance models are imperfect. This yields √n-consistency without
requiring nuisance models to converge at √n rate.

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment
and structural parameters. The Econometrics Journal, 21(1), C1-C68.

Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
Econometrica, 56(4), 931-954.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Configurable parallelism: default uses all cores; tests use 1 (sequential).
# Set DML_N_JOBS=1 in test/conftest.py to avoid multiprocessing hangs under Python 3.13.
_DEFAULT_N_JOBS = int(os.environ.get("DML_N_JOBS", "-1"))


@dataclass
class DMLResult:
    """Result container for Double ML estimation.

    Attributes
    ----------
    theta : float
        Treatment effect estimate (ATE).
    se : float
        Standard error via influence function (valid inference).
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


def _get_nuisance_model(
    model_type: Literal["ridge", "random_forest", "gradient_boosting"],
) -> BaseEstimator:
    """Get a nuisance model for E[Y|X] or E[T|X] estimation."""
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=_DEFAULT_N_JOBS,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _compute_r2(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0


def _cross_fit_nuisance(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    n_folds: int,
    outcome_model: BaseEstimator,
    treatment_model: BaseEstimator,
    random_state: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate cross-fitted predictions for nuisance models.

    For each fold, trains models on out-of-fold data and predicts
    on in-fold data. This eliminates in-sample bias.

    Parameters
    ----------
    X : array, shape (n, p)
        Covariates.
    Y : array, shape (n,)
        Outcome.
    T : array, shape (n,)
        Treatment.
    n_folds : int
        Number of cross-validation folds.
    outcome_model : sklearn estimator
        Model for E[Y|X].
    treatment_model : sklearn estimator
        Model for E[T|X].
    random_state : int
        Random seed for fold splitting.

    Returns
    -------
    Y_hat : array, shape (n,)
        Cross-fitted predictions of E[Y|X].
    T_hat : array, shape (n,)
        Cross-fitted predictions of E[T|X].
    """
    n = len(Y)
    Y_hat = np.zeros(n)
    T_hat = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        T_train = T[train_idx]

        # Train on training fold, predict on test fold (out-of-sample)
        outcome_mod = clone(outcome_model)
        outcome_mod.fit(X_train, Y_train)
        Y_hat[test_idx] = outcome_mod.predict(X_test)

        treatment_mod = clone(treatment_model)
        treatment_mod.fit(X_train, T_train)
        T_hat[test_idx] = treatment_mod.predict(X_test)

    return Y_hat, T_hat


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

    This gives valid inference without requiring the nuisance models
    to converge at √n rate.

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
    model: Optional[
        Union[BaseEstimator, Literal["ridge", "random_forest", "gradient_boosting"]]
    ] = None,
    outcome_model: Optional[BaseEstimator] = None,
    treatment_model: Optional[BaseEstimator] = None,
    alpha: float = 0.05,
    random_state: int = 42,
) -> DMLResult:
    """Double Machine Learning estimator with cross-fitting.

    Estimates the Average Treatment Effect (ATE) in the partially linear model:
        Y = θ·T + g(X) + ε

    using Neyman-orthogonal moment conditions and cross-fitting for
    √n-consistent, asymptotically normal inference.

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
        Options: "ridge", "random_forest", "gradient_boosting"
        Default: "random_forest"
    outcome_model : sklearn estimator, optional
        Model for E[Y|X]. Overrides `model` for outcome.
    treatment_model : sklearn estimator, optional
        Model for E[T|X]. Overrides `model` for treatment.
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
    The key innovation of DML over Robinson (1988) is cross-fitting:
    - Robinson: In-sample nuisance predictions → overfitting bias
    - DML: Out-of-sample nuisance predictions → no overfitting bias

    Cross-fitting ensures E[nuisance_error × score] = 0 even when
    nuisance models are imperfect, yielding valid inference.

    The influence function-based SE accounts for nuisance estimation error,
    unlike naive SEs which assume known nuisance functions.

    Examples
    --------
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.randn(n, 5)
    >>> T = np.sin(X[:, 0]) + np.random.randn(n)
    >>> Y = 2.0 * T + np.exp(X[:, 1]) + np.random.randn(n)
    >>> result = double_ml(Y, T, X, model="random_forest")
    >>> print(result.summary())
    """
    n = len(Y)

    # Validate inputs
    if len(T) != n:
        raise ValueError(f"T length ({len(T)}) must match Y length ({n})")
    if X.shape[0] != n:
        raise ValueError(f"X rows ({X.shape[0]}) must match Y length ({n})")

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Set up models
    if model is None:
        model = "random_forest"

    if isinstance(model, str):
        model_type = cast(Literal["ridge", "random_forest", "gradient_boosting"], model)
        base_model = _get_nuisance_model(model_type)
    else:
        base_model = model

    if outcome_model is None:
        outcome_model = clone(base_model)
    if treatment_model is None:
        treatment_model = clone(base_model)

    # Step 1-2: Cross-fit nuisance models
    Y_hat, T_hat = _cross_fit_nuisance(
        X=X,
        Y=Y,
        T=T,
        n_folds=n_folds,
        outcome_model=outcome_model,
        treatment_model=treatment_model,
        random_state=random_state,
    )

    # Compute cross-validated R²
    outcome_r2_cv = _compute_r2(Y, Y_hat)
    treatment_r2_cv = _compute_r2(T, T_hat)

    # Step 3: Compute residuals
    Y_tilde = Y - Y_hat
    T_tilde = T - T_hat

    # Step 4: Estimate theta
    T_tilde_sq_sum = np.sum(T_tilde**2)

    if T_tilde_sq_sum < 1e-10:
        raise ValueError(
            "Treatment has no variation after controlling for X. "
            "This may indicate perfect prediction of T by X."
        )

    theta = np.sum(Y_tilde * T_tilde) / T_tilde_sq_sum

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
    true_theta: Optional[float] = None,
    n_folds: int = 5,
) -> dict:
    """Compare Robinson (no cross-fitting) vs DML (with cross-fitting).

    This demonstrates the overfitting bias that cross-fitting eliminates.

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

    for i in range(n_sims):
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
        f"  RMSE:             {np.sqrt(np.mean((robinson_thetas - true_theta)**2)):.4f}         {np.sqrt(np.mean((dml_thetas - true_theta)**2)):.4f}"
    )
    print(f"  95% CI Coverage:  N/A*           {np.mean(dml_coverages):.1%}")
    print()
    print("  *Robinson's naive SE doesn't account for nuisance estimation,")
    print("   so its coverage is typically too low (not computed here).")
    print()
    print("★ Key Insight:")
    print("  Cross-fitting eliminates the overfitting bias from in-sample")
    print("  nuisance predictions. DML achieves correct coverage because")
    print("  the influence function SE accounts for nuisance estimation error.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_cross_fitting_benefit(n_sims=50)
