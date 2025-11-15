"""
ML baseline estimators for causal inference.

Two variants:
1. RandomForestEstimator: Random forest regression for outcome prediction
2. XGBoostEstimator: Gradient boosting for outcome prediction

Both use simple outcome regression without propensity score weighting.
"""

from typing import Tuple, Optional, Literal
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.validation.dgp_generator import DGPGenerator, DGPResult
from src.validation.validation_result import ValidationResult
from src.validation.bootstrap_config import BootstrapConfig, DEFAULT_BOOTSTRAP_CONFIG


class RandomForestEstimator:
    """
    Random Forest outcome regression estimator.

    Estimates treatment effect by fitting separate forests for T=1 and T=0,
    then computing the difference in predicted outcomes.
    Provides non-parametric alternative to linear regression.

    Args:
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level for confidence intervals (default: 0.05)
        n_estimators: Number of trees in forest (default: 100)
        random_state: Random seed for reproducibility

    Examples:
        >>> rf = RandomForestEstimator(n_simulations=100, random_state=42)
        >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
        >>> result = rf.validate(dgp)
        >>> result.status
        'PASS'
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        n_estimators: int = 100,
        bootstrap_config: Optional[BootstrapConfig] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize Random Forest estimator."""
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.bootstrap_config = bootstrap_config or DEFAULT_BOOTSTRAP_CONFIG
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """
        Run validation on DGP using Random Forest estimator.

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

            # Fit separate forests for treated and control units
            X_treated = data.X[data.T == 1]
            Y_treated = data.Y[data.T == 1]
            X_control = data.X[data.T == 0]
            Y_control = data.Y[data.T == 0]

            # Only proceed if both groups have samples
            if len(X_treated) > 0 and len(X_control) > 0:
                rf_treated = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                )
                rf_control = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                )

                rf_treated.fit(X_treated, Y_treated)
                rf_control.fit(X_control, Y_control)

                # Predict outcomes for all units under both treatment conditions
                y1_pred = rf_treated.predict(data.X)
                y0_pred = rf_control.predict(data.X)

                # ATE = mean(Y1 - Y0)
                ate = np.mean(y1_pred - y0_pred)
            else:
                ate = np.nan

            # Bootstrap CI
            ci_lower, ci_upper = self._calculate_ci_bootstrap(data, ate)

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        estimates_arr = np.array([e for e in estimates if not np.isnan(e)])
        if len(estimates_arr) == 0:
            estimates_arr = np.array(estimates)

        bias = self._calculate_bias(estimates_arr, dgp.true_effect)
        mse = self._calculate_mse(estimates_arr, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)

        # Statistical hypothesis test for bias
        bias_samples = self._calculate_bias_samples(estimates_arr, dgp.true_effect)
        status, bias_p_value = self._determine_status(bias_samples)

        return ValidationResult(
            method="RandomForestEstimator",
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
                "estimator": "RandomForestEstimator",
                "n_estimators": self.n_estimators,
                "max_depth": 10,
            },
            bias_p_value=bias_p_value,
        )

    def _calculate_ci_bootstrap(self, data: DGPResult, ate: float) -> Tuple[float, float]:
        """Calculate confidence interval via bootstrap."""
        n_bootstrap = self.bootstrap_config.n_bootstrap_ci
        bootstrap_estimates = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            indices = self._rng.choice(len(data.X), size=len(data.X), replace=True)
            X_boot = data.X[indices]
            T_boot = data.T[indices]
            Y_boot = data.Y[indices]

            X_treated = X_boot[T_boot == 1]
            Y_treated = Y_boot[T_boot == 1]
            X_control = X_boot[T_boot == 0]
            Y_control = Y_boot[T_boot == 0]

            if len(X_treated) > 0 and len(X_control) > 0:
                rf_t = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=self.random_state
                )
                rf_c = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=self.random_state
                )
                rf_t.fit(X_treated, Y_treated)
                rf_c.fit(X_control, Y_control)

                y1 = rf_t.predict(X_boot)
                y0 = rf_c.predict(X_boot)
                bootstrap_estimates[i] = np.mean(y1 - y0)
            else:
                bootstrap_estimates[i] = ate

        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)
        return float(ci_lower), float(ci_upper)

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

        status: Literal["PASS", "FAIL", "WARNING"]
        if bias_p_value < 0.01:
            status = "FAIL"
        elif bias_p_value < alpha_test:
            status = "WARNING"
        else:
            status = "PASS"

        return status, bias_p_value


class XGBoostEstimator:
    """
    XGBoost (Gradient Boosting) outcome regression estimator.

    Estimates treatment effect using gradient boosted trees for outcome regression.
    Provides high-capacity non-parametric alternative to random forests.

    Args:
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level for confidence intervals (default: 0.05)
        n_estimators: Number of boosting stages (default: 100)
        random_state: Random seed for reproducibility

    Examples:
        >>> xgb = XGBoostEstimator(n_simulations=100, random_state=42)
        >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
        >>> result = xgb.validate(dgp)
        >>> result.status
        'PASS'
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        n_estimators: int = 100,
        bootstrap_config: Optional[BootstrapConfig] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize XGBoost estimator."""
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.bootstrap_config = bootstrap_config or DEFAULT_BOOTSTRAP_CONFIG
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """
        Run validation on DGP using XGBoost estimator.

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

            # Fit separate gradient boosting models for treated and control
            X_treated = data.X[data.T == 1]
            Y_treated = data.Y[data.T == 1]
            X_control = data.X[data.T == 0]
            Y_control = data.Y[data.T == 0]

            # Only proceed if both groups have samples
            if len(X_treated) > 0 and len(X_control) > 0:
                gb_treated = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                )
                gb_control = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                )

                gb_treated.fit(X_treated, Y_treated)
                gb_control.fit(X_control, Y_control)

                # Predict outcomes for all units under both treatment conditions
                y1_pred = gb_treated.predict(data.X)
                y0_pred = gb_control.predict(data.X)

                # ATE = mean(Y1 - Y0)
                ate = np.mean(y1_pred - y0_pred)
            else:
                ate = np.nan

            # Bootstrap CI
            ci_lower, ci_upper = self._calculate_ci_bootstrap(data, ate)

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        estimates_arr = np.array([e for e in estimates if not np.isnan(e)])
        if len(estimates_arr) == 0:
            estimates_arr = np.array(estimates)

        bias = self._calculate_bias(estimates_arr, dgp.true_effect)
        mse = self._calculate_mse(estimates_arr, dgp.true_effect)
        coverage = self._calculate_coverage(ci_bounds, dgp.true_effect)

        # Statistical hypothesis test for bias
        bias_samples = self._calculate_bias_samples(estimates_arr, dgp.true_effect)
        status, bias_p_value = self._determine_status(bias_samples)

        return ValidationResult(
            method="XGBoostEstimator",
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
                "estimator": "XGBoostEstimator",
                "n_estimators": self.n_estimators,
                "max_depth": 5,
                "learning_rate": 0.1,
            },
            bias_p_value=bias_p_value,
        )

    def _calculate_ci_bootstrap(self, data: DGPResult, ate: float) -> Tuple[float, float]:
        """Calculate confidence interval via bootstrap."""
        n_bootstrap = self.bootstrap_config.n_bootstrap_ci
        bootstrap_estimates = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            indices = self._rng.choice(len(data.X), size=len(data.X), replace=True)
            X_boot = data.X[indices]
            T_boot = data.T[indices]
            Y_boot = data.Y[indices]

            X_treated = X_boot[T_boot == 1]
            Y_treated = Y_boot[T_boot == 1]
            X_control = X_boot[T_boot == 0]
            Y_control = Y_boot[T_boot == 0]

            if len(X_treated) > 0 and len(X_control) > 0:
                gb_t = GradientBoostingRegressor(
                    n_estimators=50, max_depth=4, learning_rate=0.1, random_state=self.random_state
                )
                gb_c = GradientBoostingRegressor(
                    n_estimators=50, max_depth=4, learning_rate=0.1, random_state=self.random_state
                )
                gb_t.fit(X_treated, Y_treated)
                gb_c.fit(X_control, Y_control)

                y1 = gb_t.predict(X_boot)
                y0 = gb_c.predict(X_boot)
                bootstrap_estimates[i] = np.mean(y1 - y0)
            else:
                bootstrap_estimates[i] = ate

        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)
        return float(ci_lower), float(ci_upper)

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

        status: Literal["PASS", "FAIL", "WARNING"]
        if bias_p_value < 0.01:
            status = "FAIL"
        elif bias_p_value < alpha_test:
            status = "WARNING"
        else:
            status = "PASS"

        return status, bias_p_value
