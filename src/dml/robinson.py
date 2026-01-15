"""Robinson (1988) Semiparametric Estimator.

This module implements Robinson's semiparametric partial linear model estimator,
which extends FWL by using flexible ML models for nuisance estimation.

Mathematical Foundation
-----------------------
The partially linear model:

    Y = θ·T + g(X) + ε

where:
- θ is the treatment effect of interest
- g(X) is an unknown function (not assumed linear)

Robinson's insight: We can estimate θ without assuming the form of g(X):

    1. Estimate m(X) = E[T|X] flexibly
    2. Estimate ℓ(X) = E[Y|X] flexibly
    3. Compute residuals: T̃ = T - m̂(X), Ỹ = Y - ℓ̂(X)
    4. Estimate: θ̂ = Σ(Ỹᵢ·T̃ᵢ) / Σ(T̃ᵢ²)

This is exactly FWL, but with ML models replacing linear projection!

Why Robinson → DML
------------------
Robinson's estimator has two problems:
1. In-sample overfitting of m̂(X) and ℓ̂(X) creates bias
2. Standard errors don't account for nuisance estimation error

Double ML solves both via:
1. Cross-fitting: Out-of-sample predictions eliminate overfitting bias
2. Neyman orthogonality: Moment condition is insensitive to nuisance error

This module provides Robinson as the bridge between FWL and DML.

References
----------
Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
Econometrica, 56(4), 931-954.

Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment
and structural parameters. The Econometrics Journal, 21(1), C1-C68.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


@dataclass
class RobinsonResult:
    """Result container for Robinson estimation.

    Attributes
    ----------
    theta : float
        Treatment effect estimate.
    se : float
        Standard error (WARNING: naive SE, doesn't account for nuisance estimation).
    Y_residual : NDArray[np.float64]
        Outcome residuals after partialling out X.
    T_residual : NDArray[np.float64]
        Treatment residuals after partialling out X.
    outcome_r2 : float
        R² of outcome model E[Y|X].
    treatment_r2 : float
        R² of treatment model E[T|X].
    """

    theta: float
    se: float
    Y_residual: NDArray[np.float64]
    T_residual: NDArray[np.float64]
    outcome_r2: float
    treatment_r2: float

    def __repr__(self) -> str:
        return (
            f"RobinsonResult(θ={self.theta:.4f}, naive_SE={self.se:.4f}, "
            f"outcome_R²={self.outcome_r2:.3f}, treatment_R²={self.treatment_r2:.3f})"
        )


def _get_nuisance_model(
    model_type: Literal["ridge", "random_forest", "gradient_boosting"],
) -> BaseEstimator:
    """Get a nuisance model for E[Y|X] or E[T|X] estimation.

    Parameters
    ----------
    model_type : str
        Type of model to use.

    Returns
    -------
    sklearn estimator
        Unfitted model with fit() and predict() methods.
    """
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: 'ridge', 'random_forest', 'gradient_boosting'"
        )


def _compute_r2(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


def robinson_estimator(
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    X: NDArray[np.float64],
    model: Optional[Union[BaseEstimator, Literal["ridge", "random_forest", "gradient_boosting"]]] = None,
) -> RobinsonResult:
    """Robinson (1988) semiparametric estimator.

    Estimates the partially linear model Y = θ·T + g(X) + ε using flexible
    ML models for the nuisance functions m(X) = E[T|X] and ℓ(X) = E[Y|X].

    Parameters
    ----------
    Y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable (continuous or binary).
    X : array, shape (n, p)
        Control variables / confounders.
    model : sklearn estimator or str, optional
        Model for nuisance estimation. Can be:
        - An sklearn estimator with fit() and predict()
        - A string: "ridge", "random_forest", "gradient_boosting"
        - None: defaults to "random_forest"

    Returns
    -------
    RobinsonResult
        Contains theta, SE, residuals, and R² values.

    Notes
    -----
    WARNING: The standard error returned is NAIVE. It does not account for
    estimation error in the nuisance functions. For valid inference, use
    Double ML with cross-fitting (see `double_ml` module).

    This estimator uses IN-SAMPLE predictions for nuisance models, which
    can cause overfitting bias. Cross-fitting (as in DML) solves this.

    The algorithm:
    1. Fit outcome_model on (X, Y), predict Ŷ = ℓ̂(X)
    2. Fit treatment_model on (X, T), predict T̂ = m̂(X)
    3. Compute residuals: Ỹ = Y - Ŷ, T̃ = T - T̂
    4. Estimate: θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)

    Examples
    --------
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.randn(n, 3)
    >>> T = np.sin(X[:, 0]) + np.random.randn(n)  # Nonlinear confounding
    >>> Y = 2.0 * T + np.exp(X[:, 1]) + np.random.randn(n)  # Nonlinear outcome
    >>> result = robinson_estimator(Y, T, X, model="random_forest")
    >>> abs(result.theta - 2.0) < 0.2  # Should recover ~2.0
    True
    """
    n = len(Y)

    # Validate inputs
    if len(T) != n:
        raise ValueError(f"T length ({len(T)}) must match Y length ({n})")
    if X.shape[0] != n:
        raise ValueError(f"X rows ({X.shape[0]}) must match Y length ({n})")

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Get nuisance model
    if model is None:
        model = "random_forest"

    if isinstance(model, str):
        outcome_model = _get_nuisance_model(model)
        treatment_model = _get_nuisance_model(model)
    else:
        outcome_model = clone(model)
        treatment_model = clone(model)

    # Step 1: Estimate E[Y|X]
    outcome_model.fit(X, Y)
    Y_hat = outcome_model.predict(X)
    outcome_r2 = _compute_r2(Y, Y_hat)

    # Step 2: Estimate E[T|X]
    treatment_model.fit(X, T)
    T_hat = treatment_model.predict(X)
    treatment_r2 = _compute_r2(T, T_hat)

    # Step 3: Compute residuals
    Y_tilde = Y - Y_hat
    T_tilde = T - T_hat

    # Step 4: Robinson formula
    T_tilde_sq_sum = np.sum(T_tilde ** 2)

    if T_tilde_sq_sum < 1e-10:
        raise ValueError(
            "Treatment has no variation after controlling for X. "
            "The nuisance model may have overfit or T is deterministic given X."
        )

    theta = np.sum(Y_tilde * T_tilde) / T_tilde_sq_sum

    # Naive SE (doesn't account for nuisance estimation)
    final_residuals = Y_tilde - theta * T_tilde
    sigma_sq = np.var(final_residuals, ddof=1)
    se = np.sqrt(sigma_sq / T_tilde_sq_sum)

    return RobinsonResult(
        theta=theta,
        se=se,
        Y_residual=Y_tilde,
        T_residual=T_tilde,
        outcome_r2=outcome_r2,
        treatment_r2=treatment_r2,
    )


def compare_fwl_vs_robinson(
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    X: NDArray[np.float64],
    true_theta: Optional[float] = None,
) -> dict:
    """Compare FWL (linear) vs Robinson (ML) on the same data.

    This demonstrates when linear FWL works vs when Robinson is needed.

    Parameters
    ----------
    Y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable.
    X : array, shape (n, p)
        Control variables.
    true_theta : float, optional
        True treatment effect (if known, e.g., from simulation).

    Returns
    -------
    dict
        Comparison results including both estimates and biases.
    """
    from .fwl import fwl_estimate

    fwl_result = fwl_estimate(Y, T, X, add_intercept=True)
    robinson_result = robinson_estimator(Y, T, X, model="random_forest")

    result = {
        "fwl_theta": fwl_result.theta,
        "fwl_se": fwl_result.se,
        "robinson_theta": robinson_result.theta,
        "robinson_se": robinson_result.se,
        "fwl_result": fwl_result,
        "robinson_result": robinson_result,
    }

    if true_theta is not None:
        result["true_theta"] = true_theta
        result["fwl_bias"] = fwl_result.theta - true_theta
        result["robinson_bias"] = robinson_result.theta - true_theta
        result["fwl_better"] = abs(result["fwl_bias"]) < abs(result["robinson_bias"])

    return result


def demonstrate_when_fwl_fails(seed: int = 42) -> None:
    """Demonstrate a case where FWL fails but Robinson succeeds.

    This creates data with nonlinear confounding, showing why ML
    nuisance models are necessary.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("FWL vs Robinson: When Nonlinear Confounding Breaks FWL")
    print("=" * 70)
    print()
    print("Data Generating Process:")
    print("  Y = 2.0·T + sin(2·X₁) + X₂² + ε")
    print("  T = exp(X₁/2) - 1 + X₂ + η")
    print()
    print("Note: g(X) is NONLINEAR. FWL assumes linearity and will be biased.")
    print()

    # Generate data with nonlinear confounding
    n = 2000
    X = np.random.randn(n, 2)

    # Treatment depends nonlinearly on X
    T = np.exp(X[:, 0] / 2) - 1 + X[:, 1] + 0.5 * np.random.randn(n)

    # Outcome has nonlinear dependence on X
    Y = 2.0 * T + np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + np.random.randn(n)

    true_theta = 2.0

    result = compare_fwl_vs_robinson(Y, T, X, true_theta=true_theta)

    print("Results:")
    print(f"  True θ:          {true_theta:.4f}")
    print()
    print(f"  FWL θ̂:           {result['fwl_theta']:.4f}")
    print(f"  FWL Bias:        {result['fwl_bias']:.4f}")
    print(f"  FWL R²(T~X):     {result['fwl_result'].r2_T:.3f}")
    print()
    print(f"  Robinson θ̂:      {result['robinson_theta']:.4f}")
    print(f"  Robinson Bias:   {result['robinson_bias']:.4f}")
    print(f"  Robinson R²(T~X): {result['robinson_result'].treatment_r2:.3f}")
    print()
    print("★ Key Insight:")
    print("  FWL assumes E[Y|X] and E[T|X] are linear in X.")
    print("  When confounding is nonlinear, FWL is biased.")
    print("  Robinson uses ML models that can capture nonlinearity.")
    print()
    print("  BUT: Robinson still has overfitting bias from in-sample prediction.")
    print("  Double ML solves this via cross-fitting (next module).")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_when_fwl_fails()
