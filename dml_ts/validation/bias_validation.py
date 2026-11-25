"""Bias validation via Monte Carlo simulation.

Validates DML estimator bias, MSE, and confidence interval coverage
through repeated Monte Carlo draws from a known DGP. Uses EconML's
LinearDML with a one-sample t-test for mean simulation error and a
bootstrap percentile interval for bias.
"""

from datetime import datetime
from typing import Literal, NamedTuple

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy import stats
from statsmodels.stats.multitest import multipletests

from dml_ts.validation._simulation_inference import bias_p_value, percentile_interval
from dml_ts.validation.dgp_generator import DGPGenerator, DGPResult
from dml_ts.validation.validation_result import ValidationResult


def _corrected_rejections(
    p_values: list[float],
    alpha: float,
    method: Literal["bonferroni", "holm", "none"],
) -> np.ndarray:
    """Per-hypothesis rejection decisions under a multiple-testing correction.

    Bonferroni and Holm delegate to statsmodels' canonical implementation
    rather than hand-rolling the thresholds (the mislabeled hand-rolled
    'holm' was plain Bonferroni, #14). Holm's step-down rejects the sorted
    p_(i) while p_(i) <= alpha / (k - i + 1), so it is less conservative
    than Bonferroni per hypothesis — though its FIRST step is exactly the
    Bonferroni threshold, so "any rejection" coincides for both methods.

    Boundary semantics: the statsmodels methods reject at p <= threshold
    (inclusive); "none" keeps the pre-#14 strict p < alpha. Pinned in tests.

    Args:
        p_values: Raw per-test p-values. Must be non-empty and finite —
            statsmodels would otherwise silently REJECT NaN under 'holm'
            but not under 'bonferroni' (divergent garbage), so degenerate
            input fails loudly here instead.
        alpha: Familywise significance level.
        method: "bonferroni", "holm", or "none" (uncorrected).

    Returns:
        Boolean array, True where the hypothesis is rejected.

    Raises:
        ValueError: On an unknown method name, empty input, or non-finite
            p-values.
    """
    p_arr = np.asarray(p_values, dtype=float)
    if p_arr.size == 0:
        raise ValueError("p_values is empty; cannot apply multiple-testing correction")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError(f"p_values must be finite, got {p_values}")
    if method in ("bonferroni", "holm"):
        reject, _, _, _ = multipletests(p_arr, alpha=alpha, method=method)
        return np.asarray(reject, dtype=bool)
    if method == "none":
        return np.asarray(p_arr < alpha, dtype=bool)
    raise ValueError(f"Unknown correction_method: {method}. Use 'bonferroni', 'holm', or 'none'")


Status = Literal["PASS", "WARNING", "FAIL"]


class SimulationVerdicts(NamedTuple):
    bias: Status
    coverage: Status

    @property
    def overall(self) -> Status:
        if "FAIL" in self:
            return "FAIL"
        return "WARNING" if "WARNING" in self else "PASS"


def classify_bias_and_coverage(
    bias: float,
    p_bias: float,
    coverage: float,
    nominal: float,
    p_coverage: float,
    alpha_test: float = 0.05,
    correction_method: Literal["bonferroni", "holm", "none"] = "bonferroni",
    practical_epsilon: float = 0.1,
) -> SimulationVerdicts:
    """Classify mean bias and interval calibration at familywise significance levels."""
    p_values = [p_bias, p_coverage]
    bias_rejected, coverage_rejected = _corrected_rejections(
        p_values, alpha_test, correction_method
    )
    strict_bias = _corrected_rejections(p_values, 0.01, correction_method)[0]
    bias_status: Status = "PASS"
    if abs(bias) > 0.15 or (abs(bias) >= practical_epsilon and strict_bias):
        bias_status = "FAIL"
    elif bias_rejected:
        bias_status = "WARNING"

    coverage_status: Status = "PASS"
    if coverage_rejected:
        coverage_status = "FAIL" if coverage < nominal else "WARNING"
    return SimulationVerdicts(bias_status, coverage_status)


class BiasValidation:
    """Validates DML estimator bias through Monte Carlo simulation.

    Runs ``n_simulations`` DML fits on fresh DGP draws, computing point
    estimates and confidence intervals per draw. Aggregates bias, MSE,
    and CI coverage, then applies statistical hypothesis tests (t-test
    for bias, binomial test for coverage) with multiple-testing
    correction (Bonferroni by default, or Holm; both via statsmodels).

    Args:
        n_simulations: Number of Monte Carlo runs
        alpha: Significance level for confidence intervals (default: 0.05)
        random_state: Random seed for reproducibility

    Examples:
        This numerical example requires the optional EconML dependency.
        The test runs ten fits and checks that the recovered bias is finite.

        >>> from importlib.util import find_spec
        >>> if find_spec("econml") is None:
        ...     import pytest
        ...     pytest.skip("Install the full extra to run the EconML validation example")
        >>> validator = BiasValidation(n_simulations=10, random_state=42)
        >>> dgp = DGPGenerator(n=150, p=5, true_effect=2.0, random_state=42)
        >>> result = validator.validate(dgp)
        >>> bool(np.isfinite(result.bias))
        True
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        alpha: float = 0.05,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        """Initialize validation method.

        Args:
            n_simulations: Number of Monte Carlo simulations
            alpha: Significance level (default: 0.05 for 95% CI)
            random_state: Random seed for reproducibility
            n_jobs: Number of independent DML fits to run concurrently
        """
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive, got {n_simulations}")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")
        if n_jobs < 1:
            raise ValueError("n_jobs must be positive")
        self.n_jobs = n_jobs
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """
        Run validation on DGP.

        Args:
            dgp: Data generating process to validate

        Returns:
            ValidationResult with status and metrics
        """
        if self.n_simulations < 2:
            raise ValueError(
                "Degenerate bias test requires at least two independent simulation errors"
            )
        # Step 1: Run Monte Carlo simulations (now returns estimates + CI bounds)
        estimates, ci_bounds = self._run_simulations(dgp)

        # Step 2: Calculate validation metrics
        bias = self._calculate_bias(estimates, dgp.true_effect)
        mse = self._calculate_mse(estimates, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)  # Use actual DML CIs

        # Step 3: Test mean error across the independent simulations.
        verdicts, p_bias, coverage_p_value = self._assess_draws(
            estimates - dgp.true_effect, coverage, self.n_simulations, ci_bounds, dgp.true_effect
        )

        # Step 4: Bootstrap the percentile interval for bias.
        bias_samples = self._calculate_bias_samples(estimates, dgp.true_effect)
        ci_lower, ci_upper = percentile_interval(bias_samples, self.alpha)

        # Create result (with CI bounds and p-values stored for analysis)
        result = ValidationResult(
            method="BiasValidation",
            status=verdicts.overall,
            bias=bias,
            mse=mse,
            coverage=coverage,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_simulations=self.n_simulations,
            timestamp=datetime.now(),
            metadata={
                "dgp_n": dgp.n,
                "dgp_p": dgp.p,
                "dgp_true_effect": dgp.true_effect,
                "dgp_confounding": dgp.confounding_strength,
                "alpha": self.alpha,
                "nominal_coverage": 1 - self.alpha,
                "bias_ci_level": 1 - self.alpha,
                "estimator": "LinearDML",
                "model_y": "RandomForestRegressor(n_estimators=100)",
                "model_t": "RandomForestClassifier(n_estimators=100)",  # FIXED: Classifier for binary treatment
                "discrete_treatment": True,  # FIXED: Added to metadata
                "cv_folds": 5,
            },
            # Store CI estimates for detailed analysis (optional fields)
            ci_estimates=ci_bounds,
            point_estimates=estimates,
            # Store statistical test results (Phase 0 Day 2)
            bias_p_value=p_bias,
            coverage_p_value=coverage_p_value,
            statistical_status=verdicts.overall,
            bias_status=verdicts.bias,
            coverage_status=verdicts.coverage,
        )

        return result

    def _run_simulations(self, dgp: DGPGenerator) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Monte Carlo simulations.

        Args:
            dgp: Data generating process

        Returns:
            Tuple of (estimates, ci_bounds):
                - estimates: Point estimates (n_simulations,)
                - ci_bounds: CI bounds (n_simulations, 2) where [:, 0] = lower, [:, 1] = upper
        """
        # One producer advances the DGP's RNG in the original order. Only
        # independent fits are dispatched. Results retain their input order.
        if self.n_jobs == 1:
            fits = [self._estimate_effect(dgp.generate()) for _ in range(self.n_simulations)]
        else:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                fits = Parallel(n_jobs=min(self.n_jobs, self.n_simulations))(
                    delayed(self._estimate_effect)(dgp.generate())
                    for _ in range(self.n_simulations)
                )
        values = np.asarray(fits, dtype=float)
        return values[:, 0], values[:, 1:]

    def _estimate_effect(self, data: DGPResult) -> tuple[float, float, float]:
        """
        Estimate treatment effect using Double Machine Learning (DML).

        Uses five-fold EconML LinearDML with a forest regressor for the
        outcome and a forest classifier for the binary treatment.

        Args:
            data: Generated dataset

        Returns:
            Tuple of (estimate, ci_lower, ci_upper) for coverage calculation
        """
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Configure models
        model_y = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=self.random_state
        )
        # Estimate assignment probabilities with a classifier for the binary-treatment API.
        model_t = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=self.random_state
        )

        # DML estimator with cross-fitting
        # Tell EconML that the binary assignment has two discrete levels.
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=5,  # 5-fold cross-fitting
            random_state=self.random_state,
        )

        # Fit and estimate ATE with proper marginal CI (delta method SE)
        dml.fit(Y=data.Y, T=data.T, X=data.X)
        ate = float(dml.ate(X=data.X))

        # Use ate_interval() for correct marginal ATE CI construction.
        # Note: effect_interval().mean() averages CATE intervals — methodologically
        # incorrect for marginal inference. ate_interval() uses delta method SE.
        ci_lower, ci_upper = dml.ate_interval(X=data.X, alpha=self.alpha)

        return ate, float(ci_lower), float(ci_upper)

    def _calculate_bias(self, estimates: np.ndarray, true_effect: float) -> float:
        """
        Calculate bias: E[θ̂] - θ₀

        Args:
            estimates: Treatment effect estimates
            true_effect: True treatment effect

        Returns:
            Bias
        """
        return float(np.mean(estimates - true_effect))

    def _calculate_mse(self, estimates: np.ndarray, true_effect: float) -> float:
        """
        Calculate mean squared error: E[(θ̂ - θ₀)²]

        Args:
            estimates: Treatment effect estimates
            true_effect: True treatment effect

        Returns:
            MSE
        """
        return float(np.mean((estimates - true_effect) ** 2))

    def _calculate_coverage(self, ci_bounds: np.ndarray, true_effect: float) -> float:
        """
        Calculate confidence interval coverage rate using DML's actual CIs.

        Each row contains the marginal ATE interval endpoints returned by
        ``ate_interval()`` for one independent DML fit.

        Args:
            ci_bounds: CI bounds from DML (n_simulations, 2) where [:, 0] = lower, [:, 1] = upper
            true_effect: True treatment effect

        Returns:
            Coverage rate (proportion of CIs containing true effect)
        """
        # Vectorized check: does each CI contain the true effect?
        covers = (ci_bounds[:, 0] <= true_effect) & (true_effect <= ci_bounds[:, 1])
        return float(np.mean(covers))

    def _calculate_bias_samples(
        self, estimates: np.ndarray, true_effect: float, n_bootstrap: int = 1000
    ) -> np.ndarray:
        """
        Calculate bootstrap samples of bias.

        Used for the percentile confidence interval for mean simulation bias.

        Args:
            estimates: Treatment effect estimates
            true_effect: True treatment effect
            n_bootstrap: Number of bootstrap samples (default 1000)

        Returns:
            Array of bias samples from bootstrap resampling
        """
        simulation_errors = estimates - true_effect
        bias_samples = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = self._rng.choice(simulation_errors, size=len(simulation_errors), replace=True)
            bias_samples[i] = np.mean(sample)

        return bias_samples

    def _determine_statistical_status(
        self,
        simulation_errors: np.ndarray,
        coverage: float,
        n_simulations: int,
        ci_bounds: np.ndarray,
        true_effect: float,
        alpha_test: float = 0.05,
        correction_method: Literal["bonferroni", "holm", "none"] = "bonferroni",
        practical_epsilon: float = 0.1,
    ) -> tuple[Literal["PASS", "FAIL", "WARNING"], float, float]:
        """Return the overall verdict and raw bias/coverage p-values."""
        verdicts, p_bias, p_coverage = self._assess_draws(
            simulation_errors,
            coverage,
            n_simulations,
            ci_bounds,
            true_effect,
            alpha_test,
            correction_method,
            practical_epsilon,
        )
        return verdicts.overall, p_bias, p_coverage

    def _assess_draws(
        self,
        simulation_errors: np.ndarray,
        coverage: float,
        n_simulations: int,
        ci_bounds: np.ndarray,
        true_effect: float,
        alpha_test: float = 0.05,
        correction_method: Literal["bonferroni", "holm", "none"] = "bonferroni",
        practical_epsilon: float = 0.1,
    ) -> tuple[SimulationVerdicts, float, float]:
        """Apply the bias test and exact binomial coverage test to independent draws."""
        if simulation_errors.shape != (n_simulations,) or ci_bounds.shape != (n_simulations, 2):
            raise ValueError(
                "Expected one simulation error and one confidence interval per independent draw"
            )
        p_bias = bias_p_value(simulation_errors)
        if np.isnan(ci_bounds).any() or np.any(ci_bounds[:, 0] > ci_bounds[:, 1]):
            raise ValueError("Per-draw intervals must have ordered endpoints and contain no NaN")
        covered = int(
            np.count_nonzero((ci_bounds[:, 0] <= true_effect) & (true_effect <= ci_bounds[:, 1]))
        )
        if not np.isclose(coverage, covered / n_simulations, rtol=0, atol=1e-12):
            raise ValueError("Reported coverage disagrees with the per-draw intervals")
        nominal = 1 - self.alpha
        p_coverage = float(stats.binomtest(covered, n_simulations, nominal).pvalue)
        verdicts = classify_bias_and_coverage(
            float(np.mean(simulation_errors)),
            p_bias,
            coverage,
            nominal,
            p_coverage,
            alpha_test,
            correction_method,
            practical_epsilon,
        )
        return verdicts, p_bias, p_coverage
