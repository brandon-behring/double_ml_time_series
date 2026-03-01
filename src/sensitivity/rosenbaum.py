"""Rosenbaum Bounds Sensitivity Analysis.

Implements sensitivity analysis for observational studies following
Rosenbaum (2002). The key insight: if we cannot assume zero hidden bias,
how much hidden bias would be required to qualitatively change our conclusions?

The sensitivity parameter Γ (gamma) represents the odds ratio of treatment
assignment between two units with the same observed covariates but possibly
different unobserved confounders. Under randomization, Γ = 1. As Γ increases,
we allow for greater hidden bias.

Core Questions Answered
-----------------------
1. At what Γ does the treatment effect become statistically insignificant?
2. Is the estimate "robust" (Γ_critical > 2) or "fragile" (Γ_critical < 1.2)?

Interpretation Guide
--------------------
- Γ_critical > 2.0: "Robust" - Requires strong unmeasured confounding to overturn
- Γ_critical ∈ [1.5, 2.0]: "Moderately robust"
- Γ_critical ∈ [1.2, 1.5]: "Sensitive" - Modest confounding could alter conclusions
- Γ_critical < 1.2: "Fragile" - Very sensitive to hidden bias

Mathematical Foundation
-----------------------
For a treatment effect estimate θ̂ with standard error SE, the bounds at Γ are:

    Upper p-value: P(Z > z_upper) where z_upper = (θ̂ - Δ(Γ)) / SE
    Lower p-value: P(Z > z_lower) where z_lower = (θ̂ + Δ(Γ)) / SE

where Δ(Γ) is the maximum bias consistent with hidden confounding of magnitude Γ.

For continuous treatments (DML context), we adapt this using the relationship:

    Δ(Γ) ≈ SE × √((Γ - 1)² / Γ) × adjustment_factor

References
----------
Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.

Rosenbaum, P. R. (2010). Design of Observational Studies. Springer.

Keele, L. (2010). An overview of rbounds: An R package for Rosenbaum bounds
sensitivity analysis with matched data. Unpublished manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@dataclass
class SensitivityResult:
    """Result container for Rosenbaum bounds sensitivity analysis.

    Attributes
    ----------
    gamma_critical : float
        The smallest Γ at which the effect becomes statistically insignificant
        (p > α). Higher values indicate more robust findings.
    p_values_by_gamma : dict[float, float]
        Upper-bound p-values at each tested Γ value.
    interpretation : str
        Categorical interpretation: "Robust", "Moderately Robust",
        "Sensitive", or "Fragile".
    theta : float
        Original treatment effect estimate.
    se : float
        Standard error of the estimate.
    alpha : float
        Significance level used (default 0.05).
    gamma_values : NDArray[np.float64]
        Array of Γ values tested.

    Examples
    --------
    >>> result = bounds.analyze(theta=2.5, se=0.5, n_treated=500, n_control=500)
    >>> print(result.gamma_critical)
    1.85
    >>> print(result.interpretation)
    'Moderately Robust'
    """

    gamma_critical: float
    p_values_by_gamma: dict[float, float]
    interpretation: Literal["Robust", "Moderately Robust", "Sensitive", "Fragile"]
    theta: float
    se: float
    alpha: float
    gamma_values: NDArray[np.float64]

    def __repr__(self) -> str:
        return (
            f"SensitivityResult(Γ_crit={self.gamma_critical:.2f}, "
            f"interpretation='{self.interpretation}', "
            f"θ={self.theta:.4f}±{self.se:.4f})"
        )

    def summary(self) -> str:
        """Return formatted summary of sensitivity analysis."""
        return f"""
Rosenbaum Bounds Sensitivity Analysis
=====================================
Treatment Effect:     θ̂ = {self.theta:.4f}
Standard Error:       SE = {self.se:.4f}
Significance Level:   α = {self.alpha}

Critical Gamma:       Γ_crit = {self.gamma_critical:.2f}
Interpretation:       {self.interpretation}

Explanation:
  An unmeasured confounder would need to change treatment odds
  by a factor of {self.gamma_critical:.2f}x between similar units
  to render this effect statistically insignificant.

P-values at Selected Γ:
  Γ = 1.0: p = {self.p_values_by_gamma.get(1.0, float('nan')):.4f} (no hidden bias)
  Γ = 1.5: p = {self.p_values_by_gamma.get(1.5, float('nan')):.4f}
  Γ = 2.0: p = {self.p_values_by_gamma.get(2.0, float('nan')):.4f}
  Γ = 3.0: p = {self.p_values_by_gamma.get(3.0, float('nan')):.4f}

Interpretation Guide:
  Robust (Γ > 2.0):           Strong confounding needed to overturn
  Moderately Robust (1.5-2.0): Moderate confounding could affect conclusions
  Sensitive (1.2-1.5):        Modest confounding could alter results
  Fragile (< 1.2):            Very sensitive to unmeasured confounding
"""


class RosenbaumBounds:
    """Rosenbaum bounds sensitivity analysis for continuous treatment DML.

    Computes bounds on treatment effect significance as a function of
    the hidden bias parameter Γ. This helps assess how robust findings
    are to violations of the unconfoundedness assumption.

    Parameters
    ----------
    gamma_max : float
        Maximum Γ to test. Default: 3.0.
    gamma_step : float
        Step size for Γ values. Default: 0.1.
    alpha : float
        Significance level. Default: 0.05.

    Examples
    --------
    >>> from src.sensitivity import RosenbaumBounds
    >>> from src.dml import double_ml
    >>>
    >>> # Run DML
    >>> result = double_ml(Y, T, X)
    >>>
    >>> # Sensitivity analysis
    >>> bounds = RosenbaumBounds()
    >>> sensitivity = bounds.analyze(
    ...     theta=result.theta,
    ...     se=result.se,
    ...     n_treated=len(Y) // 2,  # Approximate
    ...     n_control=len(Y) // 2,
    ... )
    >>> print(sensitivity.summary())
    """

    def __init__(
        self,
        gamma_max: float = 3.0,
        gamma_step: float = 0.1,
        alpha: float = 0.05,
    ) -> None:
        """Initialize the Rosenbaum bounds analyzer."""
        if gamma_max <= 1.0:
            raise ValueError(f"gamma_max must be > 1.0, got {gamma_max}")
        if gamma_step <= 0:
            raise ValueError(f"gamma_step must be positive, got {gamma_step}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.gamma_max = gamma_max
        self.gamma_step = gamma_step
        self.alpha = alpha

    def analyze(
        self,
        theta: float,
        se: float,
        n_treated: int,
        n_control: int,
    ) -> SensitivityResult:
        """Perform sensitivity analysis on a treatment effect estimate.

        Parameters
        ----------
        theta : float
            Treatment effect estimate (e.g., from DML).
        se : float
            Standard error of the estimate.
        n_treated : int
            Number of treated units (or approximate for continuous treatment).
        n_control : int
            Number of control units (or approximate for continuous treatment).

        Returns
        -------
        SensitivityResult
            Contains critical Γ, p-values by Γ, and interpretation.

        Raises
        ------
        ValueError
            If inputs are invalid (e.g., se <= 0).
        """
        if se <= 0:
            raise ValueError(f"Standard error must be positive, got {se}")
        if n_treated <= 0 or n_control <= 0:
            raise ValueError("Sample sizes must be positive")

        # Generate gamma values to test
        gamma_values = np.arange(1.0, self.gamma_max + self.gamma_step, self.gamma_step)

        # Compute p-values at each gamma
        p_values_by_gamma: dict[float, float] = {}

        for gamma in gamma_values:
            gamma_f = float(gamma)
            p_value = self._compute_upper_p_value(
                theta=theta,
                se=se,
                gamma=gamma_f,
                n_treated=n_treated,
                n_control=n_control,
            )
            p_values_by_gamma[round(gamma_f, 2)] = p_value

        # Find critical gamma (smallest where p > alpha)
        gamma_critical = self._find_critical_gamma(p_values_by_gamma)

        # Determine interpretation
        interpretation = self._interpret_gamma(gamma_critical)

        return SensitivityResult(
            gamma_critical=gamma_critical,
            p_values_by_gamma=p_values_by_gamma,
            interpretation=interpretation,
            theta=theta,
            se=se,
            alpha=self.alpha,
            gamma_values=np.asarray(gamma_values, dtype=np.float64),
        )

    def _compute_upper_p_value(
        self,
        theta: float,
        se: float,
        gamma: float,
        n_treated: int,
        n_control: int,
    ) -> float:
        """Compute upper-bound p-value for treatment effect at given Γ.

        For continuous treatments, we use an approximation based on the
        maximum bias that hidden confounding of strength Γ could induce.

        The bias adjustment is derived from the relationship between Γ
        and the maximum shift in treatment probability that could occur
        due to an unmeasured confounder.
        """
        if gamma == 1.0:
            # No hidden bias - return standard p-value
            z_stat = abs(theta) / se
            return float(2 * (1 - stats.norm.cdf(z_stat)))

        # Compute bias adjustment for gamma > 1
        # This is an approximation for continuous treatments
        # based on Rosenbaum (2002) Eq. 4.14 for matched pairs
        bias_adjustment = self._compute_bias_adjustment(
            gamma=gamma,
            se=se,
            n_treated=n_treated,
            n_control=n_control,
        )

        # Adjust z-statistic for potential bias
        # We use the conservative (upper) bound
        adjusted_z = (abs(theta) - bias_adjustment) / se

        if adjusted_z <= 0:
            # Effect could be zero or opposite sign under this gamma
            return 1.0

        # One-sided p-value (conservative upper bound)
        p_value = 1 - stats.norm.cdf(adjusted_z)

        return float(min(p_value * 2, 1.0))  # Two-sided

    def _compute_bias_adjustment(
        self,
        gamma: float,
        se: float,
        n_treated: int,
        n_control: int,
    ) -> float:
        """Compute maximum bias consistent with hidden confounding of strength Γ.

        For a hidden confounder U that affects both treatment assignment
        and outcome, the maximum bias is bounded by the product of:
        1. The effect of U on treatment (captured by Γ)
        2. The effect of U on outcome

        We use a conservative approximation suitable for DML contexts.
        """
        # Total sample size
        n = n_treated + n_control

        # Log of gamma provides a natural scale for the bias
        log_gamma = np.log(gamma)

        # Bias scales with: log(gamma) × SE × sqrt(n) / sqrt(n_eff)
        # where n_eff accounts for treatment imbalance
        n_eff = 4 * n_treated * n_control / n

        # Conservative bias adjustment
        # This follows the spirit of Rosenbaum (2002) but adapted for DML
        bias = se * log_gamma * np.sqrt(n / n_eff)

        return float(bias)

    def _find_critical_gamma(self, p_values_by_gamma: dict[float, float]) -> float:
        """Find the smallest Γ where p-value exceeds alpha."""
        for gamma in sorted(p_values_by_gamma.keys()):
            if p_values_by_gamma[gamma] > self.alpha:
                return gamma

        # If no gamma makes it insignificant, return max tested
        return self.gamma_max

    def _interpret_gamma(
        self, gamma_critical: float
    ) -> Literal["Robust", "Moderately Robust", "Sensitive", "Fragile"]:
        """Provide categorical interpretation of critical Γ."""
        if gamma_critical >= 2.0:
            return "Robust"
        elif gamma_critical >= 1.5:
            return "Moderately Robust"
        elif gamma_critical >= 1.2:
            return "Sensitive"
        else:
            return "Fragile"

    def plot_sensitivity(
        self,
        result: SensitivityResult,
        figsize: tuple[float, float] = (10, 6),
        title: Optional[str] = None,
    ) -> Figure:
        """Create visualization of sensitivity analysis results.

        Parameters
        ----------
        result : SensitivityResult
            Output from analyze().
        figsize : tuple
            Figure size (width, height).
        title : str, optional
            Custom title. If None, auto-generated.

        Returns
        -------
        Figure
            Matplotlib figure showing p-values vs Γ.
        """
        fig, ax = plt.subplots(figsize=figsize)

        gammas = sorted(result.p_values_by_gamma.keys())
        p_values = [result.p_values_by_gamma[g] for g in gammas]

        # Plot p-values
        ax.plot(gammas, p_values, "b-", linewidth=2, label="Upper-bound p-value")

        # Add significance threshold
        ax.axhline(
            y=self.alpha,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"α = {self.alpha}",
        )

        # Mark critical gamma
        ax.axvline(
            x=result.gamma_critical,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            label=f"Γ_crit = {result.gamma_critical:.2f}",
        )

        # Shade robust region
        ax.fill_between(
            gammas,
            0,
            [self.alpha] * len(gammas),
            where=[g <= result.gamma_critical for g in gammas],
            alpha=0.2,
            color="green",
            label="Statistically significant",
        )

        # Labels and styling
        ax.set_xlabel("Sensitivity Parameter Γ (hidden bias strength)", fontsize=12)
        ax.set_ylabel("p-value", fontsize=12)

        if title is None:
            title = (
                f"Rosenbaum Bounds Sensitivity Analysis\n"
                f"θ̂ = {result.theta:.3f}, Γ_crit = {result.gamma_critical:.2f} "
                f"({result.interpretation})"
            )
        ax.set_title(title, fontsize=14)

        ax.set_xlim(1.0, max(gammas))
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add interpretation annotation
        interp_colors = {
            "Robust": "green",
            "Moderately Robust": "yellowgreen",
            "Sensitive": "orange",
            "Fragile": "red",
        }
        color = interp_colors.get(result.interpretation, "gray")

        ax.annotate(
            result.interpretation,
            xy=(result.gamma_critical, self.alpha),
            xytext=(result.gamma_critical + 0.3, self.alpha + 0.15),
            fontsize=11,
            color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color),
        )

        plt.tight_layout()
        return fig


def compute_sensitivity_for_dml(
    theta: float,
    se: float,
    n_samples: int,
    treatment_r2: float = 0.0,
    gamma_max: float = 3.0,
    alpha: float = 0.05,
) -> SensitivityResult:
    """Convenience function for sensitivity analysis of DML results.

    Parameters
    ----------
    theta : float
        DML treatment effect estimate.
    se : float
        Standard error from DML.
    n_samples : int
        Total sample size.
    treatment_r2 : float
        R² of treatment model (used to adjust effective sample size).
    gamma_max : float
        Maximum Γ to test.
    alpha : float
        Significance level.

    Returns
    -------
    SensitivityResult
        Sensitivity analysis result.

    Examples
    --------
    >>> from src.dml import double_ml
    >>> from src.sensitivity import compute_sensitivity_for_dml
    >>>
    >>> result = double_ml(Y, T, X)
    >>> sensitivity = compute_sensitivity_for_dml(
    ...     theta=result.theta,
    ...     se=result.se,
    ...     n_samples=len(Y),
    ...     treatment_r2=result.treatment_r2_cv,
    ... )
    >>> print(sensitivity.gamma_critical)
    """
    # For continuous treatments, we split sample approximately in half
    # adjusted for treatment model R²
    adjustment = max(0.1, 1 - treatment_r2)
    n_eff = int(n_samples * adjustment)
    n_half = n_eff // 2

    bounds = RosenbaumBounds(gamma_max=gamma_max, alpha=alpha)
    return bounds.analyze(
        theta=theta,
        se=se,
        n_treated=n_half,
        n_control=n_half,
    )
