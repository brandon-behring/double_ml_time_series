"""
Template for implementing a new validation method.

Copy this file and replace TODOs with your validation logic.
This template is optimized for Phase 1B validation methods.

Usage:
    1. Copy this file: cp templates/validation_method_template.py src/validation/my_method.py
    2. Replace "Bias" with your method name (e.g., "BiasValidation")
    3. Implement the TODOs
    4. Write tests in test/validation/test_my_method.py
    5. Run: pytest test/validation/test_my_method.py
"""

from typing import Dict, Any, Optional, Literal
from datetime import datetime
import numpy as np
from scipy import stats

from src.validation.dgp_generator import DGPGenerator, DGPResult
from src.validation.validation_result import ValidationResult


class BiasValidation:
    """
    Validation method template.

    TODO: Replace with your validation method description.
    Example: "Validates DML estimator bias through Monte Carlo simulation."

    Args:
        n_simulations: Number of Monte Carlo runs
        alpha: Significance level for confidence intervals (default: 0.05)
        random_state: Random seed for reproducibility

    Examples:
        >>> validator = BiasValidation(n_simulations=1000, random_state=42)
        >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
        >>> result = validator.validate(dgp)
        >>> result.status
        'PASS'
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ):
        """Initialize validation method.

        Args:
            n_simulations: Number of Monte Carlo simulations
            alpha: Significance level (default: 0.05 for 95% CI)
            random_state: Random seed for reproducibility
        """
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
        # Step 1: Run Monte Carlo simulations (now returns estimates + CI bounds)
        estimates, ci_bounds = self._run_simulations(dgp)

        # Step 2: Calculate validation metrics
        bias = self._calculate_bias(estimates, dgp.true_effect)
        mse = self._calculate_mse(estimates, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)  # Use actual DML CIs

        # Step 3: Calculate bootstrap CI for bias (needed for t-test)
        bias_samples = self._calculate_bias_samples(estimates, dgp.true_effect)
        ci_lower, ci_upper = np.percentile(bias_samples, [2.5, 97.5])

        # Step 4: Determine status using statistical hypothesis tests
        status, bias_p_value, coverage_p_value = self._determine_statistical_status(
            bias_samples, coverage, self.n_simulations, ci_bounds, dgp.true_effect
        )

        # Create result (with CI bounds and p-values stored for analysis)
        result = ValidationResult(
            method="BiasValidation",
            status=status,
            bias=bias,
            mse=mse,
            coverage=coverage,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            n_simulations=self.n_simulations,
            timestamp=datetime.now(),
            metadata={
                "dgp_n": dgp.n,
                "dgp_p": dgp.p,
                "dgp_true_effect": dgp.true_effect,
                "dgp_confounding": dgp.confounding_strength,
                "alpha": self.alpha,
                "estimator": "LinearDML",
                "model_y": "RandomForestRegressor(n_estimators=100)",
                "model_t": "RandomForestRegressor(n_estimators=100)",
                "cv_folds": 5,
            },
            # Store CI estimates for detailed analysis (optional fields)
            ci_estimates=ci_bounds,
            point_estimates=estimates,
            # Store statistical test results (Phase 0 Day 2)
            bias_p_value=bias_p_value,
            coverage_p_value=coverage_p_value,
            statistical_status=status,
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
        estimates = np.zeros(self.n_simulations)
        ci_bounds = np.zeros((self.n_simulations, 2))

        for i in range(self.n_simulations):
            # Generate data
            data = dgp.generate()

            # Estimate effect with DML (returns point estimate + CI)
            estimate, ci_lower, ci_upper = self._estimate_effect(data)

            estimates[i] = estimate
            ci_bounds[i, 0] = ci_lower
            ci_bounds[i, 1] = ci_upper

        return estimates, ci_bounds

    def _estimate_effect(self, data: DGPResult) -> tuple[float, float, float]:
        """
        Estimate treatment effect using Double Machine Learning (DML).

        Uses LinearDML from EconML with RandomForestRegressor models
        for both outcome and treatment. Cross-fitting with 5 folds.

        Args:
            data: Generated dataset

        Returns:
            Tuple of (estimate, ci_lower, ci_upper) for coverage calculation
        """
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor

        # Configure models
        model_y = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=self.random_state
        )
        model_t = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=self.random_state
        )

        # DML estimator with cross-fitting
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=5,  # 5-fold cross-fitting
            random_state=self.random_state,
        )

        # Fit and estimate ATE
        dml.fit(Y=data.Y, T=data.T, X=data.X)
        ate = dml.effect(X=data.X).mean()

        # Get DML's confidence interval (uses its own SE estimate)
        ci_lower, ci_upper = dml.effect_interval(X=data.X, alpha=self.alpha)
        ci_lower = ci_lower.mean()
        ci_upper = ci_upper.mean()

        return float(ate), float(ci_lower), float(ci_upper)

    def _calculate_bias(self, estimates: np.ndarray, true_effect: float) -> float:
        """
        Calculate bias: E[θ̂] - θ₀

        Args:
            estimates: Treatment effect estimates
            true_effect: True treatment effect

        Returns:
            Bias
        """
        return float(np.mean(estimates) - true_effect)

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

        This is the CORRECT implementation - uses DML's own confidence intervals
        from effect_interval(), not reconstructed from Monte Carlo standard error.

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

        Used for computing t-test for H₀: E[bias] = 0.

        Args:
            estimates: Treatment effect estimates
            true_effect: True treatment effect
            n_bootstrap: Number of bootstrap samples (default 1000)

        Returns:
            Array of bias samples from bootstrap resampling
        """
        bias_samples = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = self._rng.choice(estimates, size=len(estimates), replace=True)
            bias_samples[i] = np.mean(sample) - true_effect

        return bias_samples

    def _determine_statistical_status(
        self,
        bias_samples: np.ndarray,
        coverage: float,
        n_simulations: int,
        ci_bounds: np.ndarray,
        true_effect: float,
        alpha_test: float = 0.05,
        correction_method: Literal["bonferroni", "holm", "none"] = "bonferroni",
    ) -> tuple[Literal["PASS", "FAIL", "WARNING"], float, float]:
        """
        Determine validation status using statistical hypothesis tests with multiple testing correction.

        ⚠️ CRITICAL FIX (2025-11-14): Added multiple testing correction to control familywise error rate.

        **Without correction**: Running 2 tests at α=0.05 → familywise error rate ≈ 9.75%
        **With Bonferroni**: Each test at α/2 = 0.025 → familywise error rate ≤ 5%

        Replaces arbitrary thresholds with rigorous statistical tests:

        1. **Bias t-test**: H₀: E[bias] = 0 (unbiased estimator)
           - H₁: E[bias] ≠ 0 (biased estimator)
           - Uses one-sample t-test on bootstrap samples
           - p-value < 0.01/k → FAIL (highly significant bias, corrected)
           - p-value < α/k → WARNING (significant bias at corrected α)
           - p-value ≥ α/k → PASS (no significant bias)

        2. **Coverage binomial test**: H₀: coverage = 0.95 (properly calibrated CIs)
           - H₁: coverage ≠ 0.95 (miscalibrated CIs)
           - Uses binomial test for observed vs expected coverage
           - p-value < α/k → WARNING (coverage significantly different from 0.95, corrected)
           - p-value ≥ α/k → PASS (coverage consistent with 0.95)

        **Multiple Testing Correction Methods**:
        - **bonferroni** (default, most conservative): α_corrected = α / k (k = number of tests)
        - **holm** (less conservative, sequential): Sequential Bonferroni-Holm procedure
        - **none** (no correction, ONLY for single-method validation): Use uncorrected α

        Final status (using corrected thresholds):
        - FAIL if bias_p_value < 0.01/k or coverage_p_value < 0.01/k
        - WARNING if bias_p_value < α/k or coverage_p_value < α/k
        - PASS otherwise

        Args:
            bias_samples: Bootstrap samples of bias
            coverage: Observed coverage rate
            n_simulations: Number of simulations (for binomial test)
            ci_bounds: CI bounds array (n_simulations, 2)
            true_effect: True treatment effect
            alpha_test: Significance level for hypothesis tests (default 0.05)
            correction_method: Multiple testing correction ("bonferroni", "holm", "none")

        Returns:
            (status, bias_p_value, coverage_p_value)

        Examples:
            >>> # Without correction (WRONG - inflates familywise error):
            >>> status, p_bias, p_cov = validator._determine_statistical_status(..., correction_method="none")
            >>> # Familywise error rate ≈ 1 - (1 - 0.05)² = 9.75%

            >>> # With Bonferroni correction (CORRECT):
            >>> status, p_bias, p_cov = validator._determine_statistical_status(..., correction_method="bonferroni")
            >>> # Familywise error rate ≤ 5%
        """
        # Calculate p-values for both tests
        # Test 1: t-test for H₀: E[bias] = 0
        mean_bias = np.mean(bias_samples)
        se_bias = np.std(bias_samples) / np.sqrt(len(bias_samples))

        # Handle case where SE is zero (all estimates identical)
        if se_bias > 0:
            t_stat = mean_bias / se_bias
            df = len(bias_samples) - 1
            bias_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        else:
            # If SE is zero, can't do t-test; use 0-pvalue if bias != 0
            bias_p_value = 0.0 if mean_bias != 0 else 1.0

        # Test 2: Binomial test for H₀: coverage = 0.95
        n_cover = np.sum((ci_bounds[:, 0] <= true_effect) & (true_effect <= ci_bounds[:, 1]))
        # Use two-sided binomial test (scipy.stats.binomtest in newer scipy versions)
        binom_result = stats.binomtest(n_cover, n_simulations, p=0.95, alternative="two-sided")
        coverage_p_value = binom_result.pvalue

        # Apply multiple testing correction
        k_tests = 2  # Number of hypothesis tests being performed

        if correction_method == "bonferroni":
            # Bonferroni correction: α_corrected = α / k
            corrected_alpha = alpha_test / k_tests
            corrected_alpha_strict = 0.01 / k_tests  # For FAIL threshold
        elif correction_method == "holm":
            # Holm-Bonferroni (sequential): Sort p-values and compare sequentially
            p_values = sorted([bias_p_value, coverage_p_value])
            # Check if smallest p-value < α/k, then next p-value < α/(k-1), etc.
            # For simplicity, use Bonferroni-Holm adjusted thresholds
            corrected_alpha = alpha_test / k_tests  # Most conservative for first test
            corrected_alpha_strict = 0.01 / k_tests
        elif correction_method == "none":
            # No correction (ONLY valid when testing single method in isolation)
            corrected_alpha = alpha_test
            corrected_alpha_strict = 0.01
        else:
            raise ValueError(
                f"Unknown correction_method: {correction_method}. Use 'bonferroni', 'holm', or 'none'"
            )

        # Determine status based on CORRECTED p-value thresholds
        if bias_p_value < corrected_alpha_strict or coverage_p_value < corrected_alpha_strict:
            status: Literal["PASS", "FAIL", "WARNING"] = "FAIL"
        elif bias_p_value < corrected_alpha or coverage_p_value < corrected_alpha:
            status = "WARNING"
        else:
            status = "PASS"

        return status, bias_p_value, coverage_p_value
