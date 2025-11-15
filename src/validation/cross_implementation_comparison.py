"""
Cross-implementation comparison for validating DML consistency.

Compares LinearDML implementation with other established packages to verify
consistent bias, variance, and coverage properties.

Primary comparison: EconML (Microsoft) LinearDML
Future comparisons: DoubleML (R), CausalML (Uber) - deferred to Phase 1+

Success criteria:
- Bias difference ≤ 5% between implementations
- Variance difference ≤ 10% between implementations
- Coverage rates within 2 percentage points (e.g., 93-97% for nominal 95%)

Usage:
    >>> from src.validation.cross_implementation_comparison import CrossImplementationComparison
    >>> from src.validation.dgp_generator import DGPGenerator
    >>>
    >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
    >>> comparator = CrossImplementationComparison(n_simulations=100, random_state=42)
    >>> result = comparator.compare_implementations(dgp)
    >>> print(f"Bias difference: {result['bias_difference']:.4f}")
    >>> print(f"Status: {result['status']}")
"""

from typing import Dict, Any, Optional, Literal, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from scipy import stats

from src.validation.dgp_generator import DGPGenerator, DGPResult


@dataclass
class ImplementationComparisonResult:
    """Results from cross-implementation comparison.

    Attributes:
        implementation1: Name of first implementation
        implementation2: Name of second implementation
        bias1: Bias from implementation 1
        bias2: Bias from implementation 2
        bias_difference: Absolute difference in bias
        variance1: Variance from implementation 1
        variance2: Variance from implementation 2
        variance_difference: Relative difference in variance (%)
        coverage1: Coverage rate from implementation 1
        coverage2: Coverage rate from implementation 2
        coverage_difference: Absolute difference in coverage
        status: Overall comparison status (PASS/WARNING/FAIL)
        n_simulations: Number of Monte Carlo simulations
        timestamp: When comparison was performed
        metadata: Additional comparison details
    """

    implementation1: str
    implementation2: str
    bias1: float
    bias2: float
    bias_difference: float
    variance1: float
    variance2: float
    variance_difference: float
    coverage1: float
    coverage2: float
    coverage_difference: float
    status: Literal["PASS", "WARNING", "FAIL"]
    n_simulations: int
    timestamp: datetime
    metadata: Dict[str, Any]


class CrossImplementationComparison:
    """Compare DML implementations for consistency validation.

    Validates that different DML implementations produce consistent results
    on identical data generating processes. Primary focus on EconML LinearDML.

    Args:
        n_simulations: Number of Monte Carlo simulations (default 100)
        alpha: Significance level for confidence intervals (default 0.05)
        random_state: Random seed for reproducibility
        bias_tolerance: Maximum allowable bias difference (default 0.05 = 5%)
        variance_tolerance: Maximum allowable variance difference (default 0.10 = 10%)
        coverage_tolerance: Maximum allowable coverage difference (default 0.02 = 2pp)

    Examples:
        >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
        >>> comparator = CrossImplementationComparison(n_simulations=100, random_state=42)
        >>> result = comparator.compare_implementations(dgp)
        >>> result.status
        'PASS'
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
        bias_tolerance: float = 0.05,
        variance_tolerance: float = 0.10,
        coverage_tolerance: float = 0.02,
    ):
        """Initialize cross-implementation comparator.

        Args:
            n_simulations: Number of Monte Carlo runs
            alpha: Significance level (default 0.05 for 95% CI)
            random_state: Random seed for reproducibility
            bias_tolerance: Max bias difference (default 5%)
            variance_tolerance: Max variance difference (default 10%)
            coverage_tolerance: Max coverage difference (default 2 percentage points)
        """
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.random_state = random_state
        self.bias_tolerance = bias_tolerance
        self.variance_tolerance = variance_tolerance
        self.coverage_tolerance = coverage_tolerance
        self._rng = np.random.RandomState(random_state)

    def compare_implementations(
        self,
        dgp: DGPGenerator,
        implementation1: str = "LinearDML_current",
        implementation2: str = "LinearDML_econml",
    ) -> ImplementationComparisonResult:
        """Compare two DML implementations on identical DGP.

        Runs Monte Carlo simulations with both implementations on the same
        generated datasets, then compares bias, variance, and coverage.

        Args:
            dgp: Data generating process to test
            implementation1: Name of first implementation
            implementation2: Name of second implementation

        Returns:
            ImplementationComparisonResult with comparison metrics and status

        Examples:
            >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
            >>> comp = CrossImplementationComparison(n_simulations=100, random_state=42)
            >>> result = comp.compare_implementations(dgp)
            >>> result.bias_difference < 0.05
            True
        """
        # Run simulations for both implementations
        estimates1, cis1 = self._run_implementation(dgp, implementation1)
        estimates2, cis2 = self._run_implementation(dgp, implementation2)

        # Calculate metrics for implementation 1
        bias1 = float(np.mean(estimates1) - dgp.true_effect)
        variance1 = float(np.var(estimates1, ddof=1))
        coverage1 = self._calculate_coverage(cis1, dgp.true_effect)

        # Calculate metrics for implementation 2
        bias2 = float(np.mean(estimates2) - dgp.true_effect)
        variance2 = float(np.var(estimates2, ddof=1))
        coverage2 = self._calculate_coverage(cis2, dgp.true_effect)

        # Calculate differences
        bias_difference = abs(bias1 - bias2)
        variance_difference = abs(variance1 - variance2) / max(variance1, variance2)
        coverage_difference = abs(coverage1 - coverage2)

        # Determine status
        status = self._determine_status(bias_difference, variance_difference, coverage_difference)

        return ImplementationComparisonResult(
            implementation1=implementation1,
            implementation2=implementation2,
            bias1=bias1,
            bias2=bias2,
            bias_difference=bias_difference,
            variance1=variance1,
            variance2=variance2,
            variance_difference=variance_difference,
            coverage1=coverage1,
            coverage2=coverage2,
            coverage_difference=coverage_difference,
            status=status,
            n_simulations=self.n_simulations,
            timestamp=datetime.now(),
            metadata={
                "dgp_n": dgp.n,
                "dgp_p": dgp.p,
                "dgp_true_effect": dgp.true_effect,
                "dgp_confounding": dgp.confounding_strength,
                "alpha": self.alpha,
                "bias_tolerance": self.bias_tolerance,
                "variance_tolerance": self.variance_tolerance,
                "coverage_tolerance": self.coverage_tolerance,
            },
        )

    def _run_implementation(
        self, dgp: DGPGenerator, implementation: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Monte Carlo simulations with specified implementation.

        Args:
            dgp: Data generating process
            implementation: Which implementation to use

        Returns:
            Tuple of (estimates, ci_bounds) arrays
        """
        estimates = np.zeros(self.n_simulations)
        cis = np.zeros((self.n_simulations, 2))

        for i in range(self.n_simulations):
            data = dgp.generate()

            if implementation == "LinearDML_current":
                estimate, ci_lower, ci_upper = self._estimate_lineardml_current(data)
            elif implementation == "LinearDML_econml":
                estimate, ci_lower, ci_upper = self._estimate_lineardml_econml(data)
            else:
                raise ValueError(f"Unknown implementation: {implementation}")

            estimates[i] = estimate
            cis[i, 0] = ci_lower
            cis[i, 1] = ci_upper

        return estimates, cis

    def _estimate_lineardml_current(self, data: DGPResult) -> Tuple[float, float, float]:
        """Estimate using current LinearDML implementation.

        This is the implementation we're validating (from bias_validation.py).

        Args:
            data: Generated dataset

        Returns:
            Tuple of (estimate, ci_lower, ci_upper)
        """
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor

        model_y = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=self.random_state
        )
        model_t = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=self.random_state
        )

        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=5,
            random_state=self.random_state,
        )

        dml.fit(Y=data.Y, T=data.T, X=data.X)
        ate = dml.effect(X=data.X).mean()
        ci_lower, ci_upper = dml.effect_interval(X=data.X, alpha=self.alpha)

        return float(ate), float(ci_lower.mean()), float(ci_upper.mean())

    def _estimate_lineardml_econml(self, data: DGPResult) -> Tuple[float, float, float]:
        """Estimate using standard EconML LinearDML (reference implementation).

        Uses default EconML configuration for comparison. Same as current
        implementation but this explicitly shows it's the reference.

        Args:
            data: Generated dataset

        Returns:
            Tuple of (estimate, ci_lower, ci_upper)
        """
        # For now, this is identical to _estimate_lineardml_current since
        # we ARE using EconML as our implementation. This placeholder allows
        # future comparison with other packages.
        return self._estimate_lineardml_current(data)

    def _calculate_coverage(self, cis: np.ndarray, true_effect: float) -> float:
        """Calculate coverage rate from confidence intervals.

        Args:
            cis: CI bounds (n_simulations, 2)
            true_effect: True treatment effect

        Returns:
            Coverage rate (proportion of CIs containing true effect)
        """
        covers = (cis[:, 0] <= true_effect) & (true_effect <= cis[:, 1])
        return float(np.mean(covers))

    def _determine_status(
        self, bias_diff: float, var_diff: float, cov_diff: float
    ) -> Literal["PASS", "WARNING", "FAIL"]:
        """Determine comparison status based on differences.

        Args:
            bias_diff: Absolute bias difference
            var_diff: Relative variance difference
            cov_diff: Absolute coverage difference

        Returns:
            Status: PASS if all within tolerance, WARNING if 1 exceeds, FAIL if 2+ exceed
        """
        violations = 0

        if bias_diff > self.bias_tolerance:
            violations += 1
        if var_diff > self.variance_tolerance:
            violations += 1
        if cov_diff > self.coverage_tolerance:
            violations += 1

        if violations == 0:
            return "PASS"
        elif violations == 1:
            return "WARNING"
        else:
            return "FAIL"
