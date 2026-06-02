"""
Standardized bootstrap configuration for all validation estimators.

Provides consistent bootstrap parameters across bias validation, baseline
estimators, and future validation methods.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap resampling.

    Attributes:
        n_bootstrap_bias: Number of bootstrap samples for bias estimation
                         (used in hypothesis testing for E[bias]=0)
        n_bootstrap_ci: Number of bootstrap samples for confidence intervals
                       (used for CI construction in baseline methods)
        seed: Random seed for reproducibility (None = use parent estimator's seed)
        method: Bootstrap method ('standard' or 'percentile')

    Examples:
        >>> # Default configuration
        >>> config = BootstrapConfig()
        >>> config.n_bootstrap_bias
        1000

        >>> # Fast configuration for testing
        >>> config = BootstrapConfig.fast()
        >>> config.n_bootstrap_bias
        100

        >>> # High-precision configuration
        >>> config = BootstrapConfig.precise()
        >>> config.n_bootstrap_bias
        5000
    """

    n_bootstrap_bias: int = 1000
    n_bootstrap_ci: int = 500
    seed: Optional[int] = None
    method: str = "percentile"

    @classmethod
    def fast(cls) -> "BootstrapConfig":
        """Fast configuration for testing and development.

        Uses fewer bootstrap iterations for speed.

        Returns:
            BootstrapConfig with n_bootstrap_bias=100, n_bootstrap_ci=100
        """
        return cls(n_bootstrap_bias=100, n_bootstrap_ci=100)

    @classmethod
    def tier2(cls) -> "BootstrapConfig":
        """Light config for integration tests (tier2).

        Minimal bootstrap iterations for quick multi-component testing.

        Returns:
            BootstrapConfig with n_bootstrap_bias=5, n_bootstrap_ci=5
        """
        return cls(n_bootstrap_bias=5, n_bootstrap_ci=5)

    @classmethod
    def tier3(cls) -> "BootstrapConfig":
        """Moderate config for validation tests (tier3).

        Enough iterations for statistical property verification.

        Returns:
            BootstrapConfig with n_bootstrap_bias=100, n_bootstrap_ci=50
        """
        return cls(n_bootstrap_bias=100, n_bootstrap_ci=50)

    @classmethod
    def precise(cls) -> "BootstrapConfig":
        """High-precision configuration for final validation.

        Uses more bootstrap iterations for stable estimates.

        Returns:
            BootstrapConfig with n_bootstrap_bias=5000, n_bootstrap_ci=2000
        """
        return cls(n_bootstrap_bias=5000, n_bootstrap_ci=2000)

    @classmethod
    def default(cls) -> "BootstrapConfig":
        """Default balanced configuration.

        Returns:
            BootstrapConfig with standard defaults (1000, 500)
        """
        return cls()


# Module-level default
DEFAULT_BOOTSTRAP_CONFIG = BootstrapConfig.default()
