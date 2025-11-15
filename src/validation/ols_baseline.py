"""
OLS baseline estimators for causal inference.

Two variants:
1. NaiveOLS: Y ~ T only (no confounders) - shows confounding bias
2. OLSWithControls: Y ~ T + X - standard econometrics approach
"""

from typing import Tuple, Optional, Literal
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.validation.dgp_generator import DGPGenerator, DGPResult
from src.validation.validation_result import ValidationResult
from src.validation.bootstrap_config import BootstrapConfig, DEFAULT_BOOTSTRAP_CONFIG


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
        >>> naive = NaiveOLS(n_simulations=100, random_state=42)
        >>> dgp = DGPGenerator(n=1000, p=5, confounding_strength=1.0)
        >>> result = naive.validate(dgp)
        >>> result.status  # Likely 'FAIL' due to confounding bias
        'FAIL'
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        bootstrap_config: Optional[BootstrapConfig] = None,
        random_state: Optional[int] = None,
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
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        estimates_arr = np.array(estimates)
        bias = self._calculate_bias(estimates_arr, dgp.true_effect)
        mse = self._calculate_mse(estimates_arr, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)

        # Statistical hypothesis test for bias
        bias_samples = self._calculate_bias_samples(estimates_arr, dgp.true_effect)
        status, bias_p_value = self._determine_status(bias_samples)

        return ValidationResult(
            method="NaiveOLS",
            status=status,
            bias=bias,
            mse=mse,
            coverage=coverage,
            ci_lower=float(np.percentile(bias_samples, 2.5)),
            ci_upper=float(np.percentile(bias_samples, 97.5)),
            n_simulations=self.n_simulations,
            timestamp=datetime.now(),
            metadata={
                "dgp_n": dgp.n,
                "dgp_p": dgp.p,
                "dgp_true_effect": dgp.true_effect,
                "dgp_confounding": dgp.confounding_strength,
                "alpha": self.alpha,
                "estimator": "NaiveOLS",
                "controls_used": False,
            },
            bias_p_value=bias_p_value,
        )

    def _calculate_bias(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate bias: E[θ̂] - θ₀"""
        return float(np.mean(estimates) - true_effect)

    def _calculate_mse(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate mean squared error: E[(θ̂ - θ₀)²]"""
        return float(np.mean((estimates - true_effect) ** 2))

    def _calculate_coverage(self, ci_bounds: list, true_effect: float) -> float:
        """Calculate coverage: proportion of CIs containing true effect"""
        covers = [ci[0] <= true_effect <= ci[1] for ci in ci_bounds]
        return float(np.mean(covers))

    def _calculate_bias_samples(self, estimates: np.ndarray, true_effect: float) -> np.ndarray:
        """Bootstrap samples of bias for hypothesis testing"""
        n_bootstrap = self.bootstrap_config.n_bootstrap_bias
        bias_samples = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = self._rng.choice(estimates, size=len(estimates), replace=True)
            bias_samples[i] = np.mean(sample) - true_effect
        return bias_samples

    def _determine_status(
        self, bias_samples: np.ndarray, alpha_test: float = 0.05
    ) -> Tuple[Literal["PASS", "FAIL", "WARNING"], float]:
        """Determine validation status using t-test for bias"""
        mean_bias = np.mean(bias_samples)
        se_bias = np.std(bias_samples) / np.sqrt(len(bias_samples))

        if se_bias > 0:
            t_stat = mean_bias / se_bias
            df = len(bias_samples) - 1
            bias_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        else:
            bias_p_value = 0.0 if mean_bias != 0 else 1.0

        if bias_p_value < 0.01:
            status = "FAIL"
        elif bias_p_value < alpha_test:
            status = "WARNING"
        else:
            status = "PASS"

        return status, bias_p_value


class OLSWithControls:
    """
    OLS estimator with controls: Y ~ T + X.

    Standard econometrics approach that controls for observed confounders.
    Should have less bias than NaiveOLS but more than DML (less efficient).

    Args:
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level for confidence intervals (default: 0.05)
        bootstrap_config: Bootstrap configuration (default: DEFAULT_BOOTSTRAP_CONFIG)
        random_state: Random seed for reproducibility

    Examples:
        >>> ols_controls = OLSWithControls(n_simulations=100, random_state=42)
        >>> dgp = DGPGenerator(n=1000, p=5, confounding_strength=1.0)
        >>> result = ols_controls.validate(dgp)
        >>> result.bias  # Much smaller than NaiveOLS
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        bootstrap_config: Optional[BootstrapConfig] = None,
        random_state: Optional[int] = None,
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
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        estimates_arr = np.array(estimates)
        bias = self._calculate_bias(estimates_arr, dgp.true_effect)
        mse = self._calculate_mse(estimates_arr, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)

        # Statistical hypothesis test for bias
        bias_samples = self._calculate_bias_samples(estimates_arr, dgp.true_effect)
        status, bias_p_value = self._determine_status(bias_samples)

        return ValidationResult(
            method="OLSWithControls",
            status=status,
            bias=bias,
            mse=mse,
            coverage=coverage,
            ci_lower=float(np.percentile(bias_samples, 2.5)),
            ci_upper=float(np.percentile(bias_samples, 97.5)),
            n_simulations=self.n_simulations,
            timestamp=datetime.now(),
            metadata={
                "dgp_n": dgp.n,
                "dgp_p": dgp.p,
                "dgp_true_effect": dgp.true_effect,
                "dgp_confounding": dgp.confounding_strength,
                "alpha": self.alpha,
                "estimator": "OLSWithControls",
                "controls_used": True,
            },
            bias_p_value=bias_p_value,
        )

    def _calculate_bias(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate bias: E[θ̂] - θ₀"""
        return float(np.mean(estimates) - true_effect)

    def _calculate_mse(self, estimates: np.ndarray, true_effect: float) -> float:
        """Calculate mean squared error: E[(θ̂ - θ₀)²]"""
        return float(np.mean((estimates - true_effect) ** 2))

    def _calculate_coverage(self, ci_bounds: list, true_effect: float) -> float:
        """Calculate coverage: proportion of CIs containing true effect"""
        covers = [ci[0] <= true_effect <= ci[1] for ci in ci_bounds]
        return float(np.mean(covers))

    def _calculate_bias_samples(self, estimates: np.ndarray, true_effect: float) -> np.ndarray:
        """Bootstrap samples of bias for hypothesis testing"""
        n_bootstrap = self.bootstrap_config.n_bootstrap_bias
        bias_samples = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = self._rng.choice(estimates, size=len(estimates), replace=True)
            bias_samples[i] = np.mean(sample) - true_effect
        return bias_samples

    def _determine_status(
        self, bias_samples: np.ndarray, alpha_test: float = 0.05
    ) -> Tuple[Literal["PASS", "FAIL", "WARNING"], float]:
        """Determine validation status using t-test for bias"""
        mean_bias = np.mean(bias_samples)
        se_bias = np.std(bias_samples) / np.sqrt(len(bias_samples))

        if se_bias > 0:
            t_stat = mean_bias / se_bias
            df = len(bias_samples) - 1
            bias_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        else:
            bias_p_value = 0.0 if mean_bias != 0 else 1.0

        if bias_p_value < 0.01:
            status = "FAIL"
        elif bias_p_value < alpha_test:
            status = "WARNING"
        else:
            status = "PASS"

        return status, bias_p_value
