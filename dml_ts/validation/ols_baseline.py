"""
OLS baseline estimators for causal inference.

Two variants:
1. NaiveOLS: Y ~ T only (no confounders) - shows confounding bias
2. OLSWithControls: Y ~ T + X - standard econometrics approach
"""

from datetime import datetime

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from dml_ts.validation._simulation_inference import percentile_interval
from dml_ts.validation.bias_validation import assess_simulation_draws
from dml_ts.validation.bootstrap_config import DEFAULT_BOOTSTRAP_CONFIG, BootstrapConfig
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.validation_result import ValidationResult


class NaiveOLS:
    """
    Naive OLS estimator: Y ~ T (no controls).

    Provides a baseline for the cost of ignoring confounding.
    Expected to be severely biased when confounding_strength > 0.

    Args:
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level for confidence intervals (default: 0.05)
        random_state: Random seed for reproducibility

    Examples:
        >>> from dml_ts.validation.bootstrap_config import BootstrapConfig
        >>> naive = NaiveOLS(
        ...     n_simulations=5,
        ...     bootstrap_config=BootstrapConfig.tier2(),
        ...     random_state=42,
        ... )
        >>> dgp = DGPGenerator(n=200, p=5, true_effect=2.0,
        ...                    confounding_strength=1.0, random_state=42)
        >>> result = naive.validate(dgp)
        >>> result.method
        'NaiveOLS'
        >>> bool(abs(result.bias) > 0.1)  # naive OLS is biased by the confounding
        True
        >>> bool(result.metadata["controls_used"] is False)
        True
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        bootstrap_config: BootstrapConfig | None = None,
        random_state: int | None = None,
    ):
        """Initialize naive OLS estimator."""
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.bootstrap_config = bootstrap_config or DEFAULT_BOOTSTRAP_CONFIG
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """
        Run validation on DGP using naive OLS (Y ~ T only).

        Args:
            dgp: Data generating process to validate

        Returns:
            ValidationResult with status and metrics
        """
        # Run Monte Carlo simulations
        estimates = []
        ci_bounds = []

        z_critical = stats.norm.isf(self.alpha / 2)
        for _ in range(self.n_simulations):
            data = dgp.generate()

            # Naive OLS: Y ~ T (no X controls)
            model = LinearRegression()
            T_reshaped = data.T.reshape(-1, 1)
            model.fit(T_reshaped, data.Y)

            # Point estimate
            ate = float(model.coef_[0])

            # Calculate residuals and standard error
            y_pred = model.predict(T_reshaped)
            residuals = data.Y - y_pred
            n = len(data.Y)
            mse_residuals = np.sum(residuals**2) / (n - 2)

            # SE for slope = sqrt(mse / sum((T - mean(T))^2))
            T_var = np.sum((data.T - np.mean(data.T)) ** 2)
            se = np.sqrt(mse_residuals / T_var) if T_var > 0 else np.inf

            # Confidence interval
            ci_lower = ate - z_critical * se
            ci_upper = ate + z_critical * se

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        estimates_arr = np.array(estimates)
        bias = self._calculate_bias(estimates_arr, dgp.true_effect)
        mse = self._calculate_mse(estimates_arr, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)

        verdicts, bias_p_value, coverage_p_value = assess_simulation_draws(
            estimates_arr - dgp.true_effect,
            np.asarray(ci_bounds, dtype=float),
            dgp.true_effect,
            nominal=1 - self.alpha,
        )
        bias_samples = self._calculate_bias_samples(estimates_arr, dgp.true_effect)
        bias_lower, bias_upper = percentile_interval(bias_samples, self.alpha)

        return ValidationResult(
            method="NaiveOLS",
            status=verdicts.overall,
            bias_status=verdicts.bias,
            coverage_status=verdicts.coverage,
            bias=bias,
            mse=mse,
            coverage=coverage,
            ci_lower=bias_lower,
            ci_upper=bias_upper,
            n_simulations=self.n_simulations,
            timestamp=datetime.now(),
            metadata={
                "dgp_n": dgp.n,
                "dgp_p": dgp.p,
                "dgp_true_effect": dgp.true_effect,
                "dgp_confounding": dgp.confounding_strength,
                "alpha": self.alpha,
                "bias_ci_level": 1 - self.alpha,
                "estimator": "NaiveOLS",
                "controls_used": False,
            },
            bias_p_value=bias_p_value,
            coverage_p_value=coverage_p_value,
        )

    def _calculate_bias(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate bias: E[θ̂] - θ₀"""
        return float(np.mean(estimates - true_effect))

    def _calculate_mse(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate mean squared error: E[(θ̂ - θ₀)²]"""
        return float(np.mean((estimates - true_effect) ** 2))

    def _calculate_coverage(self, ci_bounds: list, true_effect: float) -> float:
        """Calculate coverage: proportion of CIs containing true effect"""
        covers = [ci[0] <= true_effect <= ci[1] for ci in ci_bounds]
        return float(np.mean(covers))

    def _calculate_bias_samples(self, estimates: np.ndarray, true_effect: float) -> np.ndarray:
        """Resample independent estimates for the percentile interval on mean bias."""
        n_bootstrap = self.bootstrap_config.n_bootstrap_bias
        simulation_errors = estimates - true_effect
        bias_samples = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = self._rng.choice(simulation_errors, size=len(simulation_errors), replace=True)
            bias_samples[i] = np.mean(sample)
        return bias_samples


class OLSWithControls:
    """
    OLS estimator with controls: Y ~ T + X.

    Standard econometrics approach that controls for observed confounders.
    Should have less bias than NaiveOLS but more than DML (less efficient).

    Known defect (#61): the standard error is computed from a design matrix
    without the fitted intercept, so the intervals are too short. ``validate``
    reports this as a coverage verdict of FAIL even where the model is correct:
    in Chapter 3's Experiment 3.1 its 95% intervals cover in 69 of 100 datasets,
    and a reference interval that includes the intercept covers in 91.

    Args:
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level for confidence intervals (default: 0.05)
        bootstrap_config: Bootstrap configuration (default: DEFAULT_BOOTSTRAP_CONFIG)
        random_state: Random seed for reproducibility

    Examples:
        >>> from dml_ts.validation.bootstrap_config import BootstrapConfig
        >>> ols_controls = OLSWithControls(
        ...     n_simulations=5,
        ...     bootstrap_config=BootstrapConfig.tier2(),
        ...     random_state=42,
        ... )
        >>> dgp = DGPGenerator(n=200, p=5, true_effect=2.0,
        ...                    confounding_strength=1.0, random_state=42)
        >>> result = ols_controls.validate(dgp)
        >>> result.method
        'OLSWithControls'
        >>> bool(result.metadata["controls_used"] is True)
        True
        >>> bool(np.isfinite(result.bias))
        True
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        bootstrap_config: BootstrapConfig | None = None,
        random_state: int | None = None,
    ):
        """Initialize OLS with controls estimator."""
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.bootstrap_config = bootstrap_config or DEFAULT_BOOTSTRAP_CONFIG
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """
        Run validation on DGP using OLS with controls (Y ~ T + X).

        Args:
            dgp: Data generating process to validate

        Returns:
            ValidationResult with status and metrics
        """
        # Run Monte Carlo simulations
        estimates = []
        ci_bounds = []

        z_critical = stats.norm.isf(self.alpha / 2)
        for _ in range(self.n_simulations):
            data = dgp.generate()

            # OLS with controls: Y ~ T + X
            X_with_T = np.column_stack([data.T, data.X])
            model = LinearRegression()
            model.fit(X_with_T, data.Y)

            # ATE = coefficient on T (first column)
            ate = float(model.coef_[0])

            # Calculate residuals and standard error
            y_pred = model.predict(X_with_T)
            residuals = data.Y - y_pred
            n = len(data.Y)
            k = X_with_T.shape[1]
            mse_residuals = np.sum(residuals**2) / (n - k)

            # SE for slope: sqrt(mse * (X'X)^(-1)[0,0])
            try:
                XtX_inv = np.linalg.inv(X_with_T.T @ X_with_T)
                se = np.sqrt(mse_residuals * XtX_inv[0, 0])
            except np.linalg.LinAlgError:
                se = np.inf

            # Confidence interval
            ci_lower = ate - z_critical * se
            ci_upper = ate + z_critical * se

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        estimates_arr = np.array(estimates)
        bias = self._calculate_bias(estimates_arr, dgp.true_effect)
        mse = self._calculate_mse(estimates_arr, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)

        verdicts, bias_p_value, coverage_p_value = assess_simulation_draws(
            estimates_arr - dgp.true_effect,
            np.asarray(ci_bounds, dtype=float),
            dgp.true_effect,
            nominal=1 - self.alpha,
        )
        bias_samples = self._calculate_bias_samples(estimates_arr, dgp.true_effect)
        bias_lower, bias_upper = percentile_interval(bias_samples, self.alpha)

        return ValidationResult(
            method="OLSWithControls",
            status=verdicts.overall,
            bias_status=verdicts.bias,
            coverage_status=verdicts.coverage,
            bias=bias,
            mse=mse,
            coverage=coverage,
            ci_lower=bias_lower,
            ci_upper=bias_upper,
            n_simulations=self.n_simulations,
            timestamp=datetime.now(),
            metadata={
                "dgp_n": dgp.n,
                "dgp_p": dgp.p,
                "dgp_true_effect": dgp.true_effect,
                "dgp_confounding": dgp.confounding_strength,
                "alpha": self.alpha,
                "bias_ci_level": 1 - self.alpha,
                "estimator": "OLSWithControls",
                "controls_used": True,
            },
            bias_p_value=bias_p_value,
            coverage_p_value=coverage_p_value,
        )

    def _calculate_bias(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate bias: E[θ̂] - θ₀"""
        return float(np.mean(estimates - true_effect))

    def _calculate_mse(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate mean squared error: E[(θ̂ - θ₀)²]"""
        return float(np.mean((estimates - true_effect) ** 2))

    def _calculate_coverage(self, ci_bounds: list, true_effect: float) -> float:
        """Calculate coverage: proportion of CIs containing true effect"""
        covers = [ci[0] <= true_effect <= ci[1] for ci in ci_bounds]
        return float(np.mean(covers))

    def _calculate_bias_samples(self, estimates: np.ndarray, true_effect: float) -> np.ndarray:
        """Resample independent estimates for the percentile interval on mean bias."""
        n_bootstrap = self.bootstrap_config.n_bootstrap_bias
        simulation_errors = estimates - true_effect
        bias_samples = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = self._rng.choice(simulation_errors, size=len(simulation_errors), replace=True)
            bias_samples[i] = np.mean(sample)
        return bias_samples
