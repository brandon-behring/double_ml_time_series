"""Causal inference helpers built on HAC standard errors.

The HAC machinery itself (kernels, bandwidth selection, Newey-West
long-run variance) lives upstream in ``temporalcv`` — `from temporalcv
import newey_west_se, optimal_bandwidth`. This module keeps only the
causal-layer convenience that interprets a point estimate together with
its HAC standard error.
"""

import numpy as np
from scipy import stats


def hac_inference(
    theta: float,
    se_hac: float,
    alpha: float = 0.05,
) -> dict:
    """Perform inference with HAC standard errors.

    Computes confidence intervals and p-values using HAC-adjusted
    standard errors.

    Args:
        theta: Point estimate
        se_hac: HAC standard error
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Dict with: theta, se, ci_lower, ci_upper, t_stat, p_value

    Raises:
        ValueError: If se_hac is not a positive finite number.

    Example:
        >>> result = hac_inference(theta=2.5, se_hac=0.5, alpha=0.05)
        >>> result['ci_lower'], result['ci_upper']
        (1.52..., 3.47...)
    """
    if not np.isfinite(se_hac) or se_hac <= 0:
        raise ValueError(
            f"se_hac must be a positive finite number, got {se_hac!r} — "
            "degenerate standard errors must not silently produce p-values."
        )

    z_crit = stats.norm.ppf(1 - alpha / 2)
    t_stat = theta / se_hac
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "theta": theta,
        "se": se_hac,
        "ci_lower": theta - z_crit * se_hac,
        "ci_upper": theta + z_crit * se_hac,
        "t_stat": t_stat,
        "p_value": p_value,
    }
