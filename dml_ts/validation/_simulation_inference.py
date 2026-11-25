"""Inference across independent Monte Carlo simulations."""

from typing import Literal

import numpy as np
from scipy import stats


def bias_p_value(simulation_errors: np.ndarray) -> float:
    """Test zero mean error using one estimate minus truth per independent draw."""
    errors = np.asarray(simulation_errors, dtype=float)
    if errors.ndim != 1 or errors.size < 2:
        raise ValueError("Degenerate bias test requires at least two independent simulation errors")
    if not np.all(np.isfinite(errors)) or np.all(errors == errors[0]):
        raise ValueError("Degenerate bias test: simulation errors are non-finite or constant")
    se = stats.sem(errors, ddof=1)
    if not np.isfinite(se) or se <= 0:
        raise ValueError(f"Degenerate bias test: Monte Carlo standard error is {se!r}")
    return float(2 * stats.t.sf(abs(errors.mean() / se), df=errors.size - 1))


def baseline_bias_status(
    simulation_errors: np.ndarray, alpha_test: float = 0.05
) -> tuple[Literal["PASS", "FAIL", "WARNING"], float]:
    """Apply the baseline estimators' two significance thresholds to the bias test."""
    p_value = bias_p_value(simulation_errors)
    if p_value < 0.01:
        return "FAIL", p_value
    if p_value < alpha_test:
        return "WARNING", p_value
    return "PASS", p_value


def percentile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return the equal-tailed bootstrap interval at confidence level 1 - alpha."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")
    lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)
