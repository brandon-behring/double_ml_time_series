"""Frisch-Waugh-Lovell Theorem Implementation.

This module implements the FWL theorem from scratch as a pedagogical foundation
for understanding Double Machine Learning. FWL shows that partial regression
(residualization) gives identical results to multiple regression.

Mathematical Foundation
-----------------------
The Frisch-Waugh-Lovell theorem states that in the regression:

    Y = β₀ + β_T·T + X·β_X + ε

The coefficient β_T can be equivalently computed by:

    1. Residualize Y on X: Ỹ = Y - X(X'X)⁻¹X'Y = M_X·Y
    2. Residualize T on X: T̃ = T - X(X'X)⁻¹X'T = M_X·T
    3. Regress residuals: β_T = (T̃'T̃)⁻¹T̃'Ỹ

where M_X = I - X(X'X)⁻¹X' is the annihilator/residual maker matrix.

Why This Matters for DML
------------------------
Double ML is essentially "FWL with ML nuisance models":
- FWL uses linear projection for E[Y|X] and E[T|X]
- DML uses flexible ML models (random forest, boosting, etc.)

When confounding is linear, FWL works perfectly.
When confounding is nonlinear, FWL fails but DML succeeds.

This module demonstrates both cases to build intuition.

References
----------
Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with
individual trends. Econometrica, 1(4), 387-401.

Lovell, M. C. (1963). Seasonal adjustment of economic time series and multiple
regression analysis. Journal of the American Statistical Association, 58(304),
993-1010.

Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment
and structural parameters. The Econometrics Journal, 21(1), C1-C68.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from temporalcv import coverage_in_unit, finite_se

from ._results import ResultBase


@dataclass(frozen=True, slots=True, eq=False)
class FWLResult(ResultBase):
    """Result container for FWL estimation.

    Attributes
    ----------
    theta : float
        Treatment effect estimate (β_T).
    se : float
        Standard error of theta.
    t_stat : float
        t-statistic = theta / se.
    p_value : float
        Two-sided p-value.
    Y_residual : NDArray[np.float64]
        Outcome residuals (Ỹ = Y - E[Y|X]).
    T_residual : NDArray[np.float64]
        Treatment residuals (T̃ = T - E[T|X]).
    r2_Y : float
        R² from regressing Y on X (outcome model fit).
    r2_T : float
        R² from regressing T on X (propensity model fit).
    """

    theta: float
    se: float
    t_stat: float
    p_value: float
    Y_residual: NDArray[np.float64]
    T_residual: NDArray[np.float64]
    r2_Y: float
    r2_T: float

    def _validate(self) -> None:
        """B3 numeric hard-fails at the result boundary."""
        if not np.isfinite(self.theta):
            raise ValueError(f"FWLResult.theta is not finite: {self.theta!r}")
        finite_se(self.se, name="FWLResult.se")
        coverage_in_unit(self.p_value, name="FWLResult.p_value")

    def __repr__(self) -> str:
        return (
            f"FWLResult(θ={self.theta:.4f}, SE={self.se:.4f}, "
            f"t={self.t_stat:.2f}, p={self.p_value:.4f})"
        )


def fwl_residualize(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    method: Literal["ols", "qr"] = "qr",
) -> tuple[NDArray[np.float64], float]:
    """Compute residuals from projecting Y onto X.

    This is the M_X·Y operation where M_X = I - X(X'X)⁻¹X'.

    Parameters
    ----------
    Y : array, shape (n,)
        Variable to residualize.
    X : array, shape (n, p)
        Variables to partial out. Should include intercept if desired.
    method : {"ols", "qr"}
        Computation method. "qr" is more numerically stable.

    Returns
    -------
    residuals : array, shape (n,)
        Residualized Y: Ỹ = Y - X·β̂ where β̂ = (X'X)⁻¹X'Y.
    r_squared : float
        R² of the projection (variance explained by X).

    Notes
    -----
    The QR method computes residuals via
    Ỹ = Y - Q(Q'Y) where X = QR.
    This avoids explicitly forming (X'X)⁻¹.

    Examples
    --------
    >>> Y = np.array([1, 2, 3, 4, 5.0])
    >>> X = np.column_stack([np.ones(5), np.arange(5)])
    >>> residuals, r2 = fwl_residualize(Y, X)
    >>> np.allclose(residuals.sum(), 0)  # Residuals sum to ~0
    True
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if method == "qr":
        # Numerically stable: Y_resid = Y - Q @ (Q.T @ Y)
        Q, R = np.linalg.qr(X)
        Y_hat = Q @ (Q.T @ Y)
        residuals = Y - Y_hat
    else:
        # Direct OLS: β̂ = (X'X)⁻¹X'Y, Ỹ = Y - Xβ̂
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ beta
        residuals = Y - Y_hat

    # Compute R²
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    ss_res = np.sum(residuals**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    return residuals, r_squared


def fwl_estimate(
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    X: NDArray[np.float64],
    add_intercept: bool = True,
    method: Literal["ols", "qr"] = "qr",
) -> FWLResult:
    """Estimate treatment effect using Frisch-Waugh-Lovell theorem.

    This computes β_T from Y = β₀ + β_T·T + X·β_X + ε by:
    1. Residualizing Y on X
    2. Residualizing T on X
    3. Regressing the residuals

    Parameters
    ----------
    Y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable (continuous or binary).
    X : array, shape (n, p)
        Control variables / confounders.
    add_intercept : bool
        Whether to add intercept to X for residualization.
    method : {"ols", "qr"}
        Method for computing projections.

    Returns
    -------
    FWLResult
        Contains theta (treatment effect), SE, t-stat, p-value, and residuals.

    Notes
    -----
    The FWL theorem guarantees that this gives the EXACT same coefficient
    as running the full regression Y ~ T + X. This is not an approximation.

    The standard error is computed assuming homoskedasticity:
    SE(θ̂) = σ̂ / √(Σ T̃ᵢ²) where σ̂² = Σ(Ỹᵢ - θ̂T̃ᵢ)² / (n - k - 1).

    For heteroskedasticity-robust SEs, use HC0/HC1/HC3 variants (not implemented
    here; see DML module for HAC inference).

    Examples
    --------
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.randn(n, 3)  # 3 confounders
    >>> T = 0.5 * X[:, 0] + np.random.randn(n)  # Treatment depends on X₀
    >>> Y = 2.0 * T + X @ [1, 0.5, -0.3] + np.random.randn(n)  # True effect = 2.0
    >>> result = fwl_estimate(Y, T, X)
    >>> abs(result.theta - 2.0) < 0.1  # Should recover ~2.0
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

    # Add intercept if requested
    if add_intercept:
        X = np.column_stack([np.ones(n), X])

    p = X.shape[1]  # Number of control variables (including intercept)

    # Step 1: Residualize Y on X
    Y_tilde, r2_Y = fwl_residualize(Y, X, method=method)

    # Step 2: Residualize T on X
    T_tilde, r2_T = fwl_residualize(T, X, method=method)

    # Step 3: Regress Ỹ on T̃
    # θ̂ = (T̃'T̃)⁻¹T̃'Ỹ = Cov(T̃,Ỹ) / Var(T̃) since both have mean 0
    T_tilde_sq_sum = np.sum(T_tilde**2)

    if T_tilde_sq_sum < 1e-10:
        raise ValueError(
            "Treatment has no variation after controlling for X. "
            "This indicates perfect collinearity between T and X."
        )

    theta = np.sum(T_tilde * Y_tilde) / T_tilde_sq_sum

    # Step 4: Compute standard error (homoskedastic)
    # Residuals from final regression
    final_residuals = Y_tilde - theta * T_tilde

    # Degrees of freedom: n - (p controls) - 1 (treatment)
    df = n - p - 1

    if df <= 0:
        raise ValueError(f"Not enough degrees of freedom: n={n}, p={p}. Need n > p + 1.")

    # σ̂² = RSS / df
    sigma_sq = np.sum(final_residuals**2) / df

    # SE(θ̂) = σ̂ / √(Σ T̃ᵢ²)
    se = np.sqrt(sigma_sq / T_tilde_sq_sum)

    # t-statistic and p-value
    t_stat = theta / se

    # Two-sided p-value from t-distribution
    from scipy import stats

    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return FWLResult(
        theta=theta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        Y_residual=Y_tilde,
        T_residual=T_tilde,
        r2_Y=r2_Y,
        r2_T=r2_T,
    )


def fwl_vs_ols_comparison(
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    X: NDArray[np.float64],
    rtol: float = 1e-8,
) -> dict:
    """Demonstrate FWL theorem by comparing with direct OLS.

    This function runs:
    1. Full OLS regression: Y ~ T + X
    2. FWL two-step procedure

    And verifies they give identical treatment effect estimates.

    Parameters
    ----------
    Y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable.
    X : array, shape (n, p)
        Control variables.
    rtol : float
        Relative tolerance for coefficient comparison.

    Returns
    -------
    dict
        Contains both estimates, their difference, and whether they match.

    Notes
    -----
    This is primarily for pedagogical demonstration. The FWL theorem
    guarantees algebraic equality (up to numerical precision).

    Examples
    --------
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.randn(n)
    >>> Y = 1.5 * T + X @ [0.5, -0.3] + np.random.randn(n)
    >>> result = fwl_vs_ols_comparison(Y, T, X)
    >>> result["match"]
    True
    """
    n = len(Y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Method 1: Full OLS with intercept
    # Design matrix: [1, T, X]
    design = np.column_stack([np.ones(n), T, X])
    beta_ols = np.linalg.lstsq(design, Y, rcond=None)[0]
    theta_ols = beta_ols[1]  # Coefficient on T

    # Method 2: FWL procedure
    fwl_result = fwl_estimate(Y, T, X, add_intercept=True)
    theta_fwl = fwl_result.theta

    # Compare
    diff = abs(theta_ols - theta_fwl)
    match = np.isclose(theta_ols, theta_fwl, rtol=rtol)

    return {
        "theta_ols": theta_ols,
        "theta_fwl": theta_fwl,
        "difference": diff,
        "relative_diff": diff / abs(theta_ols) if abs(theta_ols) > 1e-10 else diff,
        "match": match,
        "fwl_result": fwl_result,
    }


def demonstrate_fwl_theorem(seed: int = 42) -> None:
    """Print a demonstration of the FWL theorem.

    This function generates synthetic data and shows that FWL and OLS
    give identical results, building intuition for why residualization works.
    """
    np.random.seed(seed)

    print("=" * 60)
    print("Frisch-Waugh-Lovell Theorem Demonstration")
    print("=" * 60)
    print()
    print("Setup:")
    print("  Y = 2.0·T + 1.0·X₁ + 0.5·X₂ - 0.3·X₃ + ε")
    print("  T = 0.5·X₁ + η  (treatment depends on confounder X₁)")
    print("  True treatment effect: θ = 2.0")
    print()

    # Generate data
    n = 1000
    X = np.random.randn(n, 3)
    T = 0.5 * X[:, 0] + np.random.randn(n)
    Y = 2.0 * T + X @ np.array([1.0, 0.5, -0.3]) + np.random.randn(n)

    # Compare methods
    result = fwl_vs_ols_comparison(Y, T, X)

    print("Results:")
    print(f"  OLS (Y ~ T + X):      θ̂ = {result['theta_ols']:.6f}")
    print(f"  FWL (residualize):    θ̂ = {result['theta_fwl']:.6f}")
    print(f"  Difference:           |Δ| = {result['difference']:.2e}")
    print(f"  Match (rtol=1e-8):    {result['match']}")
    print()
    print("FWL Details:")
    fwl = result["fwl_result"]
    print(f"  Standard Error:       {fwl.se:.4f}")
    print(f"  t-statistic:          {fwl.t_stat:.2f}")
    print(f"  p-value:              {fwl.p_value:.4f}")
    print(f"  R²(Y~X):              {fwl.r2_Y:.3f}")
    print(f"  R²(T~X):              {fwl.r2_T:.3f}")
    print()
    print("★ Key Insight:")
    print("  FWL and OLS give EXACTLY the same coefficient (up to numerical precision).")
    print("  This is not an approximation—it's algebraic identity.")
    print("  Double ML extends this by replacing linear projection with ML models.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_fwl_theorem()
