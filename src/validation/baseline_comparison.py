"""
Unified comparison framework for baseline estimation methods.

Provides side-by-side comparison of multiple causal inference estimators:
- NaiveOLS (baseline: shows confounding bias)
- OLSWithControls (standard econometrics)
- IPWEstimator (inverse propensity weighting)
- AugmentedIPW (doubly robust)
- RandomForestEstimator (non-parametric ML baseline)
- XGBoostEstimator (gradient boosting ML baseline)
- BiasValidation (Double Machine Learning - DML)

Enables systematic evaluation of method performance across different DGP configurations.
"""

from typing import Dict, Optional, Protocol
from datetime import datetime
import numpy as np
import pandas as pd

from src.validation.dgp_generator import DGPGenerator
from src.validation.bias_validation import BiasValidation
from src.validation.ols_baseline import NaiveOLS, OLSWithControls
from src.validation.ipw_baseline import IPWEstimator, AugmentedIPW
from src.validation.ml_baseline import RandomForestEstimator, XGBoostEstimator
from src.validation.validation_result import ValidationResult


class _Validatable(Protocol):
    """Protocol for objects that support validate(dgp) -> ValidationResult."""

    def validate(self, dgp: DGPGenerator) -> ValidationResult: ...


class BaselineComparison:
    """
    Run all baseline methods on same DGP and compare results.

    Provides unified framework for comparing:
    - Bias and MSE across methods
    - Statistical properties (coverage, CI validity)
    - Computational efficiency
    - Robustness to confounding and model misspecification

    Args:
        n_simulations: Number of Monte Carlo simulations per method
        random_state: Random seed for reproducibility
        include_dml: Whether to include DML method (default: True)
    """

    def __init__(
        self,
        n_simulations: int = 100,
        random_state: Optional[int] = None,
        include_dml: bool = True,
        include_ml: bool = False,
    ):
        """
        Initialize comparison framework.

        Args:
            n_simulations: Number of Monte Carlo simulations per method
            random_state: Random seed for reproducibility
            include_dml: Whether to include DML method (default: True)
            include_ml: Whether to include ML baseline methods (RF, XGBoost) (default: False)
        """
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.include_dml = include_dml
        self.include_ml = include_ml

        # Initialize all baseline methods
        self.methods: Dict[str, _Validatable] = {
            "NaiveOLS": NaiveOLS(n_simulations=n_simulations, random_state=random_state),
            "OLSWithControls": OLSWithControls(
                n_simulations=n_simulations, random_state=random_state
            ),
            "IPWEstimator": IPWEstimator(n_simulations=n_simulations, random_state=random_state),
            "AugmentedIPW": AugmentedIPW(n_simulations=n_simulations, random_state=random_state),
        }

        # Optionally include ML baselines
        if include_ml:
            self.methods["RandomForest"] = RandomForestEstimator(
                n_simulations=n_simulations, random_state=random_state
            )
            self.methods["XGBoost"] = XGBoostEstimator(
                n_simulations=n_simulations, random_state=random_state
            )

        # Optionally include DML
        if include_dml:
            self.methods["DML"] = BiasValidation(
                n_simulations=n_simulations, random_state=random_state
            )

    def compare(self, dgp: DGPGenerator) -> Dict[str, ValidationResult]:
        """
        Run all methods on same DGP with deterministic comparability.

        FIXED (Issue C3): Reset DGP random state before each method to ensure
        all methods see identical random draws. This prevents ranking instability
        where methods could vary by up to 4 positions due to different data.

        Args:
            dgp: Data generating process to validate against

        Returns:
            Dictionary mapping method names to ValidationResults
        """
        results = {}
        for name, method in self.methods.items():
            # FIXED (Issue C3): Reset DGP random state for deterministic comparisons
            # Each method now sees the SAME random draws for fair comparison
            dgp._rng = np.random.RandomState(dgp.random_state)
            results[name] = method.validate(dgp)
        return results

    def create_comparison_table(self, dgp: DGPGenerator) -> pd.DataFrame:
        """
        Create summary table comparing all methods.

        Args:
            dgp: Data generating process to validate against

        Returns:
            DataFrame with one row per method, columns for key metrics
        """
        results = self.compare(dgp)

        # Build comparison table
        data = []
        for name, result in results.items():
            data.append(
                {
                    "Method": name,
                    "Bias": f"{result.bias:.4f}",
                    "MSE": f"{result.mse:.4f}",
                    "RMSE": f"{np.sqrt(result.mse):.4f}",
                    "Coverage": f"{result.coverage:.1%}",
                    "Status": result.status,
                    "Bias 95% CI": f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
                    "P-value": f"{result.bias_p_value:.4e}",
                }
            )

        return pd.DataFrame(data)

    def create_detailed_comparison_table(self, dgp: DGPGenerator) -> pd.DataFrame:
        """
        Create detailed comparison table with additional statistics.

        Args:
            dgp: Data generating process to validate against

        Returns:
            DataFrame with extended metrics for in-depth analysis
        """
        results = self.compare(dgp)

        # Build detailed table
        data = []
        for name, result in results.items():
            data.append(
                {
                    "Method": name,
                    "Bias": result.bias,
                    "Abs_Bias": abs(result.bias),
                    "MSE": result.mse,
                    "RMSE": np.sqrt(result.mse),
                    "Coverage": result.coverage,
                    "CI_Lower": result.ci_lower,
                    "CI_Upper": result.ci_upper,
                    "CI_Width": result.ci_upper - result.ci_lower,
                    "Status": result.status,
                    "P_Value": result.bias_p_value,
                    "N_Simulations": result.n_simulations,
                    "Timestamp": result.timestamp,
                }
            )

        return pd.DataFrame(data)

    def compare_across_dgps(self, dgp_configs: list[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Compare methods across multiple DGP configurations.

        Args:
            dgp_configs: List of DGPGenerator configuration dicts
                        (passed as **kwargs to DGPGenerator)

        Returns:
            Dictionary mapping config string → comparison table
        """
        all_results = {}

        for config in dgp_configs:
            dgp = DGPGenerator(**config)
            config_str = self._config_to_string(config)
            all_results[config_str] = self.create_comparison_table(dgp)

        return all_results

    def _config_to_string(self, config: Dict) -> str:
        """Convert DGP config dict to readable string."""
        parts = []
        for key in ["n", "p", "true_effect", "confounding_strength"]:
            if key in config:
                parts.append(f"{key}={config[key]}")
        return ", ".join(parts)

    def generate_summary_statistics(self, dgp: DGPGenerator) -> Dict:
        """
        Generate summary statistics across all methods.

        Args:
            dgp: Data generating process to validate against

        Returns:
            Dictionary with summary metrics (best, worst, average performance)
        """
        results = self.compare(dgp)

        # Extract biases
        biases = {name: abs(result.bias) for name, result in results.items()}
        mses = {name: result.mse for name, result in results.items()}
        coverages = {name: result.coverage for name, result in results.items()}

        # Find best/worst
        best_bias_method = min(biases, key=biases.get)
        worst_bias_method = max(biases, key=biases.get)
        best_mse_method = min(mses, key=mses.get)
        best_coverage_method = max(coverages, key=coverages.get)

        return {
            "dgp_config": {
                "n": dgp.n,
                "p": dgp.p,
                "true_effect": dgp.true_effect,
                "confounding_strength": dgp.confounding_strength,
            },
            "best_bias_method": best_bias_method,
            "worst_bias_method": worst_bias_method,
            "best_mse_method": best_mse_method,
            "best_coverage_method": best_coverage_method,
            "all_biases": biases,
            "all_mses": mses,
            "all_coverages": coverages,
            "mean_absolute_bias": np.mean(list(biases.values())),
            "mean_mse": np.mean(list(mses.values())),
            "mean_coverage": np.mean(list(coverages.values())),
        }
