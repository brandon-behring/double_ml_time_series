"""Illustrative sensitivity curve for a scalar estimate and standard error.

The assumed bias equals SE × log(Γ) × (n_treated + n_control) /
(2 × sqrt(n_treated × n_control)). Subtracting that bias from the absolute
estimate gives an adjusted two-sided normal test. Formal Rosenbaum bounds
require a specified assignment model and matched data.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats


@dataclass
class SensitivityResult:
    """Result container for Rosenbaum bounds sensitivity analysis.

    Attributes
    ----------
    gamma_critical : float
        The smallest Γ at which the effect becomes statistically insignificant
        (p > α), or the upper grid endpoint if the crossing is right-censored.
        Use crossing_observed to distinguish these cases.
    p_values_by_gamma : dict[float, float]
        Adjusted normal-test p-values under the assumed bias curve.
    interpretation : str
        Descriptive category of the reported grid value. For a censored result
        this categorizes the endpoint, which is a lower bound on the threshold.
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
    >>> from dml_ts.sensitivity import RosenbaumBounds
    >>> bounds = RosenbaumBounds()
    >>> result = bounds.analyze(theta=2.5, se=0.5, n_treated=500, n_control=500)
    >>> # 3.0 == gamma_max: significance survives the whole tested Gamma range
    >>> # (no tipping point found), so the finding is reported as "Robust".
    >>> float(result.gamma_critical)
    3.0
    >>> result.interpretation
    'Robust'
    """

    gamma_critical: float
    p_values_by_gamma: dict[float, float]
    interpretation: Literal["Robust", "Moderately Robust", "Sensitive", "Fragile"]
    theta: float
    se: float
    alpha: float
    gamma_values: NDArray[np.float64]

    @property
    def crossing_observed(self) -> bool:
        """Whether a tested p-value exceeded alpha, including at the endpoint."""
        # Historical grids could omit a non-divisible upper endpoint.
        p = self.p_values_by_gamma.get(self.gamma_critical)
        return p is not None and p > self.alpha

    @property
    def gamma_label(self) -> str:
        """Threshold or lower bound for display."""
        relation = "=" if self.crossing_observed else ">"
        return f"Γ_crit {relation} {self.gamma_critical:g}"

    def __repr__(self) -> str:
        return (
            f"SensitivityResult({self.gamma_label}, "
            f"crossing_observed={self.crossing_observed}, "
            f"θ={self.theta:.4f}±{self.se:.4f})"
        )

    def summary(self) -> str:
        """Return formatted summary of sensitivity analysis."""
        grid = "\n".join(f"  Γ = {g:g}    p = {p:.4g}" for g, p in self.p_values_by_gamma.items())
        description = (
            f"First tested crossing ({self.interpretation})"
            if self.crossing_observed
            else "No crossing within the tested range"
        )
        return (
            "Illustrative Rosenbaum Bounds Sensitivity Analysis\n"
            f"Estimate {self.theta:.4f}, SE {self.se:.4f}, α {self.alpha:g}\n"
            f"Critical Gamma    {self.gamma_label}    {description}\n"
            "The curve assumes a bias proportional to SE × log(Γ).\n"
            f"{grid}"
        )


class RosenbaumBounds:
    """Illustrative sensitivity curve for a scalar DML coefficient.

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
    >>> from dml_ts.sensitivity import RosenbaumBounds
    >>> # Given a DML estimate (theta, se) and approximate group sizes:
    >>> bounds = RosenbaumBounds()
    >>> sensitivity = bounds.analyze(
    ...     theta=2.5,
    ...     se=0.5,
    ...     n_treated=500,  # Approximate
    ...     n_control=500,
    ... )
    >>> sensitivity.interpretation
    'Robust'
    >>> isinstance(sensitivity.summary(), str)
    True
    """

    def __init__(
        self,
        gamma_max: float = 3.0,
        gamma_step: float = 0.1,
        alpha: float = 0.05,
    ) -> None:
        """Initialize the Rosenbaum bounds analyzer."""
        if not np.isfinite(gamma_max) or gamma_max <= 1.0:
            raise ValueError(f"gamma_max must be > 1.0, got {gamma_max}")
        if not np.isfinite(gamma_step) or gamma_step <= 0:
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
        if not np.isfinite(theta):
            raise ValueError("Treatment effect must be finite")
        if not np.isfinite(se) or se <= 0:
            raise ValueError(f"Standard error must be positive, got {se}")
        if not np.isfinite([n_treated, n_control]).all() or min(n_treated, n_control) <= 0:
            raise ValueError("Sample sizes must be positive")

        # Generate gamma values to test
        limit, step = Decimal(str(self.gamma_max)), Decimal(str(self.gamma_step))
        points = [float(1 + i * step) for i in range(int((limit - 1) // step) + 1)]
        if points[-1] < self.gamma_max:
            points.append(self.gamma_max)
        gamma_values = np.array(points)

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
            p_values_by_gamma[gamma_f] = p_value

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
        """Apply the assumed bias function to a two-sided normal test."""
        imbalance = (n_treated + n_control) / (2 * np.sqrt(float(n_treated) * n_control))
        adjusted_z = max(0.0, abs(theta) / se - np.log(gamma) * imbalance)
        return float(2 * stats.norm.sf(adjusted_z))

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
        title: str | None = None,
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
        ax.plot(gammas, p_values, "b-", linewidth=2, label="Adjusted p-value")

        # Add significance threshold
        ax.axhline(
            y=result.alpha,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"α = {result.alpha}",
        )

        # Mark critical gamma
        ax.axvline(
            x=result.gamma_critical,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            label=result.gamma_label,
        )

        # Shade robust region
        ax.fill_between(
            gammas,
            0,
            [result.alpha] * len(gammas),
            where=[p <= result.alpha for p in p_values],
            alpha=0.2,
            color="green",
            label="Statistically significant",
        )

        # Labels and styling
        ax.set_xlabel("Sensitivity Parameter Γ (hidden bias strength)", fontsize=12)
        ax.set_ylabel("p-value", fontsize=12)

        if title is None:
            title = (
                f"Illustrative sensitivity curve\n"
                f"θ̂ = {result.theta:.3f}, {result.gamma_label} "
                f"({result.interpretation if result.crossing_observed else 'no crossing'})"
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
            result.interpretation if result.crossing_observed else "No crossing",
            xy=(result.gamma_critical, result.alpha),
            xytext=(result.gamma_critical + 0.3, result.alpha + 0.15),
            fontsize=11,
            color=color,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": color},
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
        Retained for compatibility. Equal proxy group sizes cancel in the bias
        function, so treatment R² cannot calibrate this curve.
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
    >>> from dml_ts.sensitivity import compute_sensitivity_for_dml
    >>> # Given a DML estimate (theta, se, treatment R^2) on n_samples rows:
    >>> sensitivity = compute_sensitivity_for_dml(
    ...     theta=1.2,
    ...     se=0.3,
    ...     n_samples=1000,
    ...     treatment_r2=0.2,
    ... )
    >>> # 3.0 == gamma_max: no tipping point within the tested Gamma range.
    >>> float(sensitivity.gamma_critical)
    3.0
    """
    if n_samples < 2:
        raise ValueError("At least two observations are required")
    if not np.isfinite(treatment_r2):
        raise ValueError("treatment_r2 must be finite")
    n_half = n_samples // 2
    bounds = RosenbaumBounds(gamma_max=gamma_max, alpha=alpha)
    return bounds.analyze(
        theta=theta,
        se=se,
        n_treated=n_half,
        n_control=n_half,
    )
