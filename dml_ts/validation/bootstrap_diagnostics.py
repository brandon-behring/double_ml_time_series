"""
Bootstrap diagnostics for assessing resampling quality and convergence.

Provides tools to validate bootstrap procedures used in DML validation:
- Convergence diagnostics (stability as n_bootstrap increases)
- Distribution diagnostics (normality, symmetry)
- Variance stability (Monte Carlo error in estimates)
- Recommendations for appropriate n_bootstrap values

Usage:
    >>> from dml_ts.validation.bootstrap_diagnostics import BootstrapDiagnostics
    >>> from dml_ts.validation.dgp_generator import DGPGenerator
    >>>
    >>> dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
    >>> data = dgp.generate()
    >>>
    >>> # Diagnose bootstrap convergence for bias estimation.
    >>> # "OLS" keeps the example fast; "LinearDML" uses econml nuisance models.
    >>> diagnostics = BootstrapDiagnostics(
    ...     data=data, estimator_type="OLS", random_state=42
    ... )
    >>> result = diagnostics.diagnose_convergence(
    ...     target="bias",
    ...     n_bootstrap_range=[20, 40],
    ...     true_value=2.0,
    ...     n_replications=2,
    ... )
    >>> bool(result.recommended_n in {20, 40})
    True
    >>> isinstance(result.converged, bool)
    True
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from dml_ts.validation.dgp_generator import DGPResult

from ._bootstrap_folds import original_fold_ids, resampled_folds


@dataclass
class ConvergenceDiagnostic:
    """Results from bootstrap convergence analysis.

    Attributes:
        n_bootstrap_tested: List of bootstrap sample sizes tested
        estimates: Bootstrap estimates at each sample size
        std_errors: Standard errors at each sample size
        convergence_score: Score indicating convergence quality (0-1, higher is better)
        converged: Whether bootstrap has converged
        recommended_n: Recommended n_bootstrap for stable results
        diagnostics: Additional diagnostic information
    """

    n_bootstrap_tested: list[int]
    estimates: list[float]
    std_errors: list[float]
    convergence_score: float
    converged: bool
    recommended_n: int
    diagnostics: dict[str, Any]


@dataclass
class DistributionDiagnostic:
    """Results from bootstrap distribution analysis.

    Attributes:
        n_bootstrap: Number of bootstrap samples used
        normality_pvalue: P-value from Shapiro-Wilk normality test
        is_normal: Whether distribution appears normal (p > 0.05)
        skewness: Sample skewness
        kurtosis: Sample excess kurtosis
        symmetry_score: Score indicating distributional symmetry (0-1)
        diagnostics: Additional diagnostic information
    """

    n_bootstrap: int
    normality_pvalue: float
    is_normal: bool
    skewness: float
    kurtosis: float
    symmetry_score: float
    diagnostics: dict[str, Any]


class BootstrapDiagnostics:
    """Bootstrap quality diagnostics for DML validation.

    LinearDML case refits use fixed original-unit folds. Repeated copies of a
    source row share its fold; the underlying forest nuisance fits use fixed
    hyperparameters. Each original row is assumed to represent one sampling unit.

    Provides comprehensive diagnostics for bootstrap procedures including:
    - Convergence analysis (optimal n_bootstrap)
    - Distribution diagnostics (normality, skewness, kurtosis)
    - Variance stability assessment
    - Recommendations for bootstrap configuration

    Args:
        data: DGP result containing (Y, T, X)
        estimator_type: Type of estimator ("LinearDML", "IPW", "AIPW", "OLS")
        random_state: Random seed for reproducibility

    Examples:
        >>> from dml_ts.validation.dgp_generator import DGPGenerator
        >>> dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        >>> data = dgp.generate()
        >>> diag = BootstrapDiagnostics(data, "OLS", random_state=42)
        >>>
        >>> # Check convergence (OLS keeps the example fast)
        >>> conv = diag.diagnose_convergence(
        ...     "bias", [20, 40], true_value=2.0, n_replications=2
        ... )
        >>> isinstance(conv.converged, bool)
        True
        >>>
        >>> # Check distribution quality
        >>> dist = diag.diagnose_distribution(n_bootstrap=40)
        >>> isinstance(dist.is_normal, bool)
        True
        >>> bool(np.isfinite(dist.skewness))
        True
    """

    def __init__(
        self,
        data: DGPResult,
        estimator_type: Literal["LinearDML", "IPW", "AIPW", "OLS"] = "LinearDML",
        random_state: int | None = None,
    ):
        """Initialize bootstrap diagnostics.

        Args:
            data: Generated dataset (Y, T, X)
            estimator_type: Which estimator to use for bootstrap
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.estimator_type = estimator_type
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self._original_fold_ids: np.ndarray | None = None

    def diagnose_convergence(
        self,
        target: Literal["bias", "variance", "coverage", "conditional_inclusion"],
        n_bootstrap_range: list[int],
        true_value: float | None = None,
        n_replications: int = 10,
        tolerance: float = 0.05,
        data_generator: Callable[[], DGPResult] | None = None,
    ) -> ConvergenceDiagnostic:
        """Diagnose bootstrap convergence across different sample sizes.

        Tests whether bootstrap estimates stabilize as n_bootstrap increases.
        Uses multiple replications to assess Monte Carlo variability.

        Args:
            target: Statistic to diagnose. "coverage" uses independent generated datasets.
                "conditional_inclusion" repeats the bootstrap on the supplied fixed data.
            n_bootstrap_range: List of n_bootstrap values to test
            true_value: Parameter value for bias, coverage or conditional inclusion
            n_replications: Number of replications per n_bootstrap
            tolerance: Relative change threshold for convergence (default 5%)
            data_generator: Callable producing a fresh independent dataset per replication,
                e.g. a persistent seeded DGP's generate method. Required for coverage.
                The same replication is used for every bootstrap sample size.

        Returns:
            ConvergenceDiagnostic with convergence assessment and recommendations

        Examples:
            Illustrative (see the class docstring for a runnable example):

            >>> diag.diagnose_convergence(  # doctest: +SKIP
            ...     "bias", [100, 500, 1000, 2000], true_value=2.0
            ... )
            ConvergenceDiagnostic(converged=True, recommended_n=1000, ...)
        """
        if target not in ("bias", "variance", "coverage", "conditional_inclusion"):
            raise ValueError(f"Unknown convergence target: {target}")
        if target != "variance" and true_value is None:
            raise ValueError(f"true_value required for {target} diagnostics")
        if target == "coverage" and data_generator is None:
            raise ValueError(
                "coverage requires a data_generator for independent datasets; "
                "use conditional_inclusion for one fixed dataset"
            )
        if n_replications < 2 or not n_bootstrap_range or min(n_bootstrap_range) < 2:
            raise ValueError("Use at least two replications and bootstrap sizes of at least two")

        measurements = np.empty((n_replications, len(n_bootstrap_range)))
        for replication in range(n_replications):
            trial = self
            if target == "coverage" and data_generator is not None:
                trial = BootstrapDiagnostics(
                    data_generator(), self.estimator_type, int(self._rng.randint(2**31))
                )
            for column, n_boot in enumerate(n_bootstrap_range):
                estimates = trial._bootstrap_estimates(n_boot)
                if target == "variance":
                    value = np.var(estimates, ddof=1)
                elif target == "bias":
                    value = np.mean(estimates) - true_value
                else:
                    lower, upper = np.quantile(estimates, [0.025, 0.975])
                    value = float(lower <= true_value <= upper)
                measurements[replication, column] = value

        estimates_by_n = measurements.mean(axis=0).tolist()
        replication_sd = measurements.std(axis=0, ddof=1)
        std_errors_by_n = (replication_sd / np.sqrt(n_replications)).tolist()

        # Assess convergence: relative change between consecutive n_bootstrap values
        converged = False
        recommended_n = n_bootstrap_range[-1]
        convergence_score = 0.0

        if len(n_bootstrap_range) >= 2:
            relative_changes = []
            for i in range(1, len(estimates_by_n)):
                prev_est = estimates_by_n[i - 1]
                curr_est = estimates_by_n[i]
                if abs(prev_est) > 1e-10:
                    rel_change = abs((curr_est - prev_est) / prev_est)
                else:
                    rel_change = abs(curr_est - prev_est)
                relative_changes.append(rel_change)

            # Converged if last 2 changes are below tolerance
            if len(relative_changes) >= 2:
                recent_changes = relative_changes[-2:]
                if all(rc < tolerance for rc in recent_changes):
                    converged = True
                    # Find first n where convergence maintained
                    for i, rc in enumerate(relative_changes):
                        if rc < tolerance and all(r < tolerance for r in relative_changes[i:]):
                            recommended_n = n_bootstrap_range[i + 1]
                            break

            # Convergence score: inverse of average relative change (capped at 1.0)
            avg_change = np.mean(relative_changes) if relative_changes else 1.0
            convergence_score = min(1.0, tolerance / (avg_change + 1e-10))

        return ConvergenceDiagnostic(
            n_bootstrap_tested=n_bootstrap_range,
            estimates=estimates_by_n,
            std_errors=std_errors_by_n,
            convergence_score=convergence_score,
            converged=converged,
            recommended_n=recommended_n,
            diagnostics={
                "target": target,
                "tolerance": tolerance,
                "n_replications": n_replications,
                "true_value": true_value,
                "monte_carlo_error": std_errors_by_n,
                "replication_sd": replication_sd.tolist(),
                "independent_datasets": n_replications if target == "coverage" else 1,
            },
        )

    def diagnose_distribution(
        self, n_bootstrap: int = 1000, alpha: float = 0.05
    ) -> DistributionDiagnostic:
        """Diagnose bootstrap distribution properties.

        Assesses whether bootstrap distribution is appropriate:
        - Normality (Shapiro-Wilk test)
        - Skewness and kurtosis
        - Symmetry around mean

        Args:
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level for normality test

        Returns:
            DistributionDiagnostic with distribution properties

        Examples:
            Illustrative (see the class docstring for a runnable example):

            >>> diag.diagnose_distribution(n_bootstrap=1000)  # doctest: +SKIP
            DistributionDiagnostic(is_normal=True, skewness=0.12, ...)
        """
        # Shapiro-Wilk needs >= 3 observations to test normality. Fabricating
        # normality_pvalue=0.0 (-> is_normal=False) from fewer would assert
        # non-normality out of insufficient data, so fail loudly up front.
        if n_bootstrap < 3:
            raise ValueError(
                "diagnose_distribution needs n_bootstrap >= 3 for the Shapiro-Wilk "
                f"normality test, got {n_bootstrap}."
            )

        # Generate bootstrap distribution
        boot_estimates = self._bootstrap_estimates(n_bootstrap)

        # A non-finite or point-mass (all-identical) bootstrap distribution cannot be
        # diagnosed: Shapiro-Wilk returns p=1.0 (is_normal=True) and skew/kurtosis become
        # NaN, fabricating a normality verdict from degenerate data. Fail loud instead.
        if not np.all(np.isfinite(boot_estimates)) or np.ptp(boot_estimates) == 0:
            raise ValueError(
                "Degenerate bootstrap distribution: estimates are non-finite or a single "
                f"point mass (range={float(np.ptp(boot_estimates)):.3g}); normality and "
                "skewness diagnostics would be fabricated. Check for a constant/degenerate "
                "sample."
            )

        # Normality test (Shapiro-Wilk)
        stat, normality_pvalue = stats.shapiro(boot_estimates)
        is_normal = bool(float(normality_pvalue) > alpha)

        # Skewness and kurtosis
        skewness = float(stats.skew(boot_estimates))
        kurtosis = float(stats.kurtosis(boot_estimates))  # Excess kurtosis

        # Symmetry score: based on skewness (0 = perfect symmetry)
        symmetry_score = max(0.0, 1.0 - abs(skewness) / 2.0)  # Cap at skewness=2

        return DistributionDiagnostic(
            n_bootstrap=n_bootstrap,
            normality_pvalue=normality_pvalue,
            is_normal=is_normal,
            skewness=skewness,
            kurtosis=kurtosis,
            symmetry_score=symmetry_score,
            diagnostics={
                "mean": float(np.mean(boot_estimates)),
                "median": float(np.median(boot_estimates)),
                "std": float(np.std(boot_estimates, ddof=1)),
                "percentiles": {
                    "p05": float(np.percentile(boot_estimates, 5)),
                    "p25": float(np.percentile(boot_estimates, 25)),
                    "p75": float(np.percentile(boot_estimates, 75)),
                    "p95": float(np.percentile(boot_estimates, 95)),
                },
            },
        )

    def recommend_n_bootstrap(
        self,
        target_tasks: list[Literal["bias", "ci", "both"]] | None = None,
        precision_level: Literal["fast", "default", "precise"] = "default",
    ) -> dict[str, int]:
        """Recommend n_bootstrap values based on convergence diagnostics.

        Runs convergence diagnostics to determine appropriate bootstrap sample
        sizes for different tasks (bias estimation vs confidence intervals).

        Args:
            target_tasks: Which tasks to optimize for ("bias", "ci", "both").
                Defaults to ["both"].
            precision_level: Desired precision ("fast", "default", "precise")

        Returns:
            Dict with recommended n_bootstrap for each task

        Examples:
            Illustrative; runs the configured estimator many times and is slow:

            >>> diag.recommend_n_bootstrap(["bias", "ci"], "default")  # doctest: +SKIP
            {'bias': 1000, 'ci': 500, 'both': 1000}
        """
        if target_tasks is None:
            target_tasks = ["both"]

        # Default ranges based on precision level
        ranges_bias = {
            "fast": [100, 200, 500],
            "default": [500, 1000, 2000],
            "precise": [2000, 5000, 10000],
        }
        ranges_ci = {
            "fast": [50, 100, 200],
            "default": [200, 500, 1000],
            "precise": [1000, 2000, 5000],
        }

        recommendations = {}

        if "bias" in target_tasks or "both" in target_tasks:
            # Estimate true effect for bias calculation
            point_est = self._estimate_point()
            conv_diag = self.diagnose_convergence(
                target="bias",
                n_bootstrap_range=ranges_bias[precision_level],
                true_value=point_est,
                tolerance=0.05,
            )
            recommendations["bias"] = conv_diag.recommended_n

        if "ci" in target_tasks or "both" in target_tasks:
            # CI diagnostics based on variance stability
            conv_diag = self.diagnose_convergence(
                target="variance",
                n_bootstrap_range=ranges_ci[precision_level],
                tolerance=0.10,  # CI can tolerate slightly more variability
            )
            recommendations["ci"] = conv_diag.recommended_n

        if "both" in target_tasks:
            # Conservative: use larger of the two
            recommendations["both"] = max(
                recommendations.get("bias", 0), recommendations.get("ci", 0)
            )

        return recommendations

    def _bootstrap_estimates(self, n_bootstrap: int) -> np.ndarray:
        """Generate bootstrap distribution of point estimates.

        Args:
            n_bootstrap: Number of bootstrap samples

        Returns:
            Array of bootstrap estimates
        """
        boot_estimates = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # Resample data
            indices = self._rng.choice(len(self.data.Y), size=len(self.data.Y), replace=True)
            Y_boot = self.data.Y[indices]
            T_boot = self.data.T[indices]
            X_boot = self.data.X[indices]

            # Estimate on bootstrap sample
            if self.estimator_type == "LinearDML":
                folds = self._dml_folds(indices)
                estimate = self._estimate_dml(Y_boot, T_boot, X_boot, cv=folds)
            elif self.estimator_type == "OLS":
                estimate = self._estimate_ols(Y_boot, T_boot, X_boot)
            elif self.estimator_type in ["IPW", "AIPW"]:
                raise NotImplementedError(f"{self.estimator_type} not yet implemented")
            else:
                raise ValueError(f"Unknown estimator: {self.estimator_type}")

            boot_estimates[i] = estimate

        return boot_estimates

    def _dml_folds(self, indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._original_fold_ids is None:
            self._original_fold_ids = original_fold_ids(
                self.data.T, self.random_state, stratified=False
            )
        return resampled_folds(self._original_fold_ids, indices)

    def _estimate_point(self) -> float:
        """Estimate point estimate on full data."""
        if self.estimator_type == "LinearDML":
            folds = self._dml_folds(np.arange(len(self.data.Y)))
            return self._estimate_dml(self.data.Y, self.data.T, self.data.X, cv=folds)
        elif self.estimator_type == "OLS":
            return self._estimate_ols(self.data.Y, self.data.T, self.data.X)
        else:
            raise NotImplementedError(f"{self.estimator_type} not yet implemented")

    def _estimate_dml(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        cv: int | list[tuple[np.ndarray, np.ndarray]] = 5,
    ) -> float:
        """Estimate treatment effect using LinearDML."""
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor

        model_y = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=self.random_state
        )
        model_t = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=self.random_state
        )

        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=cv,
            random_state=self.random_state,
        )

        dml.fit(Y=Y, T=T, X=X)
        return float(dml.effect(X=X).mean())

    def _estimate_ols(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> float:
        """Estimate treatment effect using OLS with controls."""
        from sklearn.linear_model import LinearRegression

        X_with_T = np.column_stack([T, X])
        model = LinearRegression()
        model.fit(X_with_T, Y)
        return float(model.coef_[0])
