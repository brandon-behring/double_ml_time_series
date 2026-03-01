"""
IPW baseline estimators for causal inference.

Two variants:
1. IPWEstimator: Inverse propensity weighting (IPW)
2. AugmentedIPW: Augmented IPW / Doubly Robust estimator
"""

from typing import Tuple, Optional, Literal
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.validation.dgp_generator import DGPGenerator, DGPResult
from src.validation.validation_result import ValidationResult
from src.validation.bootstrap_config import BootstrapConfig, DEFAULT_BOOTSTRAP_CONFIG


class IPWEstimator:
    """
    Inverse propensity weighting estimator.

    Estimates treatment effect by weighting observations by inverse propensity scores.
    Provides consistent estimation when propensity score model is correct.
    Sensitive to overlap violations (extreme propensity scores).

    Args:
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level for confidence intervals (default: 0.05)
        random_state: Random seed for reproducibility

    Examples:
        >>> ipw = IPWEstimator(n_simulations=100, random_state=42)
        >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
        >>> result = ipw.validate(dgp)
        >>> result.status  # Status depends on confounding and overlap
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        bootstrap_config: Optional[BootstrapConfig] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize IPW estimator."""
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.bootstrap_config = bootstrap_config or DEFAULT_BOOTSTRAP_CONFIG
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """
        Run validation on DGP using IPW estimator.

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

            # Step 1: Estimate propensity score P(T=1|X)
            ps_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            ps_model.fit(data.X, data.T)
            propensity_scores = ps_model.predict_proba(data.X)[:, 1]

            # Clip propensity scores to avoid division by zero
            propensity_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)

            # Step 2: IPW estimator
            # E[Y * T / P(T=1|X)] - E[Y * (1-T) / (1-P(T=1|X))]
            weights_t = data.T / propensity_scores
            weights_0 = (1 - data.T) / (1 - propensity_scores)

            ate = np.mean(data.Y * weights_t) - np.mean(data.Y * weights_0)

            # Step 3: Approximate CI via bootstrap
            ci_lower, ci_upper = self._calculate_ci_bootstrap(
                data.Y, data.T, propensity_scores, ate
            )

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
            method="IPWEstimator",
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
                "estimator": "IPWEstimator",
                "propensity_model": "LogisticRegression",
            },
            bias_p_value=bias_p_value,
        )

    def _calculate_ci_bootstrap(
        self, Y: np.ndarray, T: np.ndarray, ps: np.ndarray, ate: float
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval via bootstrap.

        Args:
            Y: Outcome variable
            T: Treatment indicator
            ps: Propensity scores
            ate: Point estimate of ATE
            n_bootstrap: Number of bootstrap samples

        Returns:
            (ci_lower, ci_upper) tuple
        """
        n_bootstrap = self.bootstrap_config.n_bootstrap_ci
        bootstrap_estimates = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # Resample with replacement
            indices = self._rng.choice(len(Y), size=len(Y), replace=True)
            Y_boot = Y[indices]
            T_boot = T[indices]
            ps_boot = ps[indices]

            # Calculate IPW estimator on bootstrap sample
            weights_t = T_boot / ps_boot
            weights_0 = (1 - T_boot) / (1 - ps_boot)
            bootstrap_estimates[i] = np.mean(Y_boot * weights_t) - np.mean(Y_boot * weights_0)

        # Calculate percentile CI
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


class AugmentedIPW:
    """
    Augmented inverse propensity weighting (doubly robust estimator).

    Combines outcome regression with IPW for double robustness.
    Consistent if EITHER outcome or propensity model is correct (not necessarily both).
    More robust to model misspecification than pure IPW or regression alone.

    Args:
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level for confidence intervals (default: 0.05)
        bootstrap_config: Bootstrap configuration (default: DEFAULT_BOOTSTRAP_CONFIG)
        random_state: Random seed for reproducibility

    Examples:
        >>> aipw = AugmentedIPW(n_simulations=100, random_state=42)
        >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
        >>> result = aipw.validate(dgp)
        >>> result.status  # Usually PASS due to double robustness
    """

    def __init__(
        self,
        n_simulations: int = 100,
        alpha: float = 0.05,
        bootstrap_config: Optional[BootstrapConfig] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize augmented IPW estimator."""
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.bootstrap_config = bootstrap_config or DEFAULT_BOOTSTRAP_CONFIG
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """
        Run validation on DGP using augmented IPW estimator.

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

            # Step 1: Estimate outcome models separately for T=1 and T=0
            y1_model = LinearRegression()
            y0_model = LinearRegression()

            # Fit on treated and control subgroups
            if np.sum(data.T == 1) > 0:
                y1_model.fit(data.X[data.T == 1], data.Y[data.T == 1])
            if np.sum(data.T == 0) > 0:
                y0_model.fit(data.X[data.T == 0], data.Y[data.T == 0])

            # Predict potential outcomes for all units
            y1_pred = y1_model.predict(data.X)
            y0_pred = y0_model.predict(data.X)

            # Step 2: Estimate propensity score P(T=1|X)
            ps_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            ps_model.fit(data.X, data.T)
            propensity_scores = ps_model.predict_proba(data.X)[:, 1]

            # Clip propensity scores to avoid division by zero
            propensity_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)

            # Step 3: Doubly robust combination
            # ATE = E[Y1 - Y0] + E[T * (Y - Y1) / P] - E[(1-T) * (Y - Y0) / (1-P)]
            ate = (
                np.mean(y1_pred - y0_pred)
                + np.mean(data.T * (data.Y - y1_pred) / propensity_scores)
                - np.mean((1 - data.T) * (data.Y - y0_pred) / (1 - propensity_scores))
            )

            # Step 4: Approximate CI via bootstrap
            ci_lower, ci_upper = self._calculate_ci_bootstrap(
                data.Y, data.T, data.X, y1_pred, y0_pred, propensity_scores, ate
            )

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
            method="AugmentedIPW",
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
                "estimator": "AugmentedIPW",
                "outcome_model": "LinearRegression",
                "propensity_model": "LogisticRegression",
                "doubly_robust": True,
            },
            bias_p_value=bias_p_value,
        )

    def _calculate_ci_bootstrap(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        y1_pred: np.ndarray,
        y0_pred: np.ndarray,
        ps: np.ndarray,
        ate: float,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval via bootstrap.

        Args:
            Y: Outcome variable
            T: Treatment indicator
            X: Covariates
            y1_pred: Predicted Y when T=1
            y0_pred: Predicted Y when T=0
            ps: Propensity scores
            ate: Point estimate of ATE
            n_bootstrap: Number of bootstrap samples

        Returns:
            (ci_lower, ci_upper) tuple
        """
        n_bootstrap = self.bootstrap_config.n_bootstrap_ci
        bootstrap_estimates = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # Resample with replacement
            indices = self._rng.choice(len(Y), size=len(Y), replace=True)
            Y_boot = Y[indices]
            T_boot = T[indices]
            X_boot = X[indices]
            y1_pred_boot = y1_pred[indices]
            y0_pred_boot = y0_pred[indices]
            ps_boot = ps[indices]

            # Calculate doubly robust estimator on bootstrap sample
            bootstrap_estimates[i] = (
                np.mean(y1_pred_boot - y0_pred_boot)
                + np.mean(T_boot * (Y_boot - y1_pred_boot) / ps_boot)
                - np.mean((1 - T_boot) * (Y_boot - y0_pred_boot) / (1 - ps_boot))
            )

        # Calculate percentile CI
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
