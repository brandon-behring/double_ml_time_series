"""Stability diagnostics for the companion 401(k) eligibility fit.

The companion lasso uses a single fivefold partition, cross-validated nuisance
penalties and linearly heterogeneous effects. Its default inputs include the
wealth measures nifa and tw. Chernozhukov et al. (2018), Table 2, Panel B,
report $8,187 for the fivefold homogeneous partially linear coefficient with
rigorous lasso, aggregated across 100 partitions. The diagnostics here measure
sensitivity to resampling, tuning and seeds within the companion specification.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._bootstrap_folds import original_fold_ids, resampled_folds


@dataclass
class BootstrapDiagnosticResult:
    """Results from bootstrap distribution analysis.

    Attributes:
        ate_estimates: Array of ATE estimates from bootstrap samples
        mean_ate: Mean of bootstrap distribution
        std_ate: Standard deviation of bootstrap distribution
        ci_lower: Lower CI bound (2.5th percentile)
        ci_upper: Upper CI bound (97.5th percentile)
        normality_pvalue: P-value from Shapiro-Wilk normality test
        is_normal: Whether distribution passes normality test (p > 0.05)
        has_outliers: Whether distribution has extreme outliers (>3 SD)
        n_outliers: Number of outliers detected
        converged: Whether bootstrap distribution appears converged
        n_bootstrap: Number of bootstrap samples used
        metadata: Additional diagnostic info
    """

    ate_estimates: np.ndarray
    mean_ate: float
    std_ate: float
    ci_lower: float
    ci_upper: float
    normality_pvalue: float
    is_normal: bool
    has_outliers: bool
    n_outliers: int
    converged: bool
    n_bootstrap: int
    metadata: dict[str, Any]


@dataclass
class HyperparameterSensitivityResult:
    """Results from hyperparameter sensitivity analysis.

    Attributes:
        parameter_name: Name of varied parameter (cv_folds, max_iter)
        parameter_values: Array of tested parameter values
        ate_estimates: Array of ATE estimates for each parameter value
        std_errors: Array of standard errors for each parameter value
        sensitivity_score: Coefficient of variation (std/mean) of ATE estimates
        is_sensitive: Whether ATE is sensitive to this parameter (CV > 0.1)
        recommended_value: Candidate with the smallest reported ATE standard error
        metadata: Additional analysis info
    """

    parameter_name: str
    parameter_values: np.ndarray
    ate_estimates: np.ndarray
    std_errors: np.ndarray
    sensitivity_score: float
    is_sensitive: bool
    recommended_value: Any
    metadata: dict[str, Any]


@dataclass
class SeedSensitivityResult:
    """Results from cross-fitting seed sensitivity analysis.

    Attributes:
        random_states: Array of tested random_state values
        ate_estimates: Array of ATE estimates for each seed
        mean_ate: Mean ATE across seeds
        std_ate: Standard deviation of ATE across seeds
        min_ate: Minimum ATE observed
        max_ate: Maximum ATE observed
        range_ate: Range of ATE estimates (max - min)
        cv_ate: Coefficient of variation (std/mean)
        is_stable: Whether estimates are stable across seeds (CV < 0.1)
        metadata: Additional analysis info
    """

    random_states: np.ndarray
    ate_estimates: np.ndarray
    mean_ate: float
    std_ate: float
    min_ate: float
    max_ate: float
    range_ate: float
    cv_ate: float
    is_stable: bool
    metadata: dict[str, Any]


@dataclass
class ComprehensiveDiagnosticResult:
    """Results of the 401(k) lasso stability checks.

    Attributes:
        bootstrap_diagnostic: Bootstrap distribution analysis
        hyperparameter_sensitivity: Dict of sensitivity analyses (cv_folds, max_iter)
        seed_sensitivity: Cross-fitting seed analysis
        root_cause_analysis: Summary of the observed diagnostic signals
        recommendations: List of recommendations for fixing or documenting issue
        timestamp: When diagnostic was performed
    """

    bootstrap_diagnostic: BootstrapDiagnosticResult
    hyperparameter_sensitivity: dict[str, HyperparameterSensitivityResult]
    seed_sensitivity: SeedSensitivityResult
    root_cause_analysis: str
    recommendations: list[str]
    timestamp: datetime


class LassoDiagnostic:
    """Diagnose sensitivity of the companion lasso eligibility estimate.

    Args:
        data_path: Path to 401(k) dataset (optional, loads from doubleml if None)
        random_state: Base random seed for reproducibility
        verbose: Whether to print diagnostic progress

    Examples:
        Requires the real 401(k) dataset and many econml DML fits:

        >>> diagnostic = LassoDiagnostic(random_state=42, verbose=True)  # doctest: +SKIP
        >>> results = diagnostic.run_comprehensive_diagnostic()  # doctest: +SKIP
        >>> if not results.bootstrap_diagnostic.converged:  # doctest: +SKIP
        ...     print("Bootstrap distribution not converged - increase n_bootstrap")
    """

    def __init__(
        self,
        data_path: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize diagnostic tool.

        Args:
            data_path: Path to 401(k) dataset
            random_state: Base random seed
            verbose: Print progress messages
        """
        self.data_path = data_path
        self.random_state = random_state
        self.verbose = verbose
        self._rng = np.random.RandomState(random_state)
        self._data: pd.DataFrame | None = None
        self._Y: np.ndarray | None = None
        self._T: np.ndarray | None = None
        self._X: np.ndarray | None = None

    def _log(self, message: str) -> None:
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)

    def load_data(self) -> pd.DataFrame:
        """Load 401(k) dataset.

        Returns:
            DataFrame with 401(k) data
        """
        if self._data is not None:
            return self._data

        if self.data_path is not None:
            self._data = pd.read_csv(self.data_path)
        else:
            try:
                from doubleml.datasets import fetch_401K

                self._data = fetch_401K(return_type="DataFrame")
            except ImportError as e:
                raise ImportError(
                    "doubleml package required. Install with: pip install doubleml"
                ) from e

        assert self._data is not None
        return self._data

    def preprocess_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess 401(k) data for DML estimation.

        Returns:
            Tuple of (Y, T, X) arrays
        """
        df = self.load_data()

        # Outcome
        self._Y = np.asarray(df["net_tfa"])

        # Treatment (e401 eligibility)
        self._T = np.asarray(df["e401"])

        # The companion adds nifa and tw to the published nine controls.
        control_vars = [col for col in df.columns if col not in ["net_tfa", "e401", "p401"]]
        self._X = np.asarray(df[control_vars])

        return self._Y, self._T, self._X

    def analyze_bootstrap_distribution(self, n_bootstrap: int = 1000) -> BootstrapDiagnosticResult:
        """Analyze bootstrap distribution of Lasso ATE estimates.

        This pairs bootstrap holds a fivefold, treatment-stratified partition of
        the original household records fixed. Every resampled copy inherits its
        household's fold. The output describes variability of the refitted
        effect, conditional on that partition. In EconML 0.16 the two penalties
        are selected using these folds across the whole bootstrap sample.

        Args:
            n_bootstrap: Number of bootstrap samples

        Returns:
            BootstrapDiagnosticResult with distribution analysis

        Examples:
            Requires the real 401(k) dataset and an econml DML fit per sample:

            >>> diagnostic = LassoDiagnostic(random_state=42)  # doctest: +SKIP
            >>> result = diagnostic.analyze_bootstrap_distribution(  # doctest: +SKIP
            ...     n_bootstrap=1000
            ... )
            >>> result.is_normal  # doctest: +SKIP
            False
            >>> result.n_outliers  # doctest: +SKIP
            23
        """
        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        self._log(f"Running bootstrap distribution analysis (B={n_bootstrap})...")

        Y, T, X = self.preprocess_data()
        n = len(Y)
        original_fold = original_fold_ids(T, self.random_state, stratified=True)
        n_arms = np.unique(T).size

        ate_estimates = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            if self.verbose and (b + 1) % 100 == 0:
                self._log(f"  Bootstrap sample {b + 1}/{n_bootstrap}")

            # Resample with replacement
            boot_idx = self._rng.choice(n, size=n, replace=True)
            Y_boot = Y[boot_idx]
            T_boot = T[boot_idx]
            X_boot = X[boot_idx, :]

            folds = resampled_folds(original_fold, boot_idx)
            if any(np.unique(T_boot[train]).size != n_arms for train, _ in folds):
                raise ValueError(
                    f"Bootstrap draw {b + 1} lacks a treatment arm in a household training fold"
                )

            model_y = LassoCV(random_state=self.random_state)
            model_t = LogisticRegressionCV(
                penalty="l1", solver="saga", random_state=self.random_state, max_iter=1000
            )

            dml = LinearDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                cv=folds,
                random_state=self.random_state,
            )

            dml.fit(Y=Y_boot, T=T_boot, X=X_boot)
            ate_estimates[b] = float(dml.effect(X=X_boot).mean())

        # Distribution statistics
        mean_ate = float(np.mean(ate_estimates))
        std_ate = float(np.std(ate_estimates, ddof=1))
        ci_lower = float(np.percentile(ate_estimates, 2.5))
        ci_upper = float(np.percentile(ate_estimates, 97.5))

        # Normality test
        normality_stat, normality_pvalue = stats.shapiro(ate_estimates)
        is_normal = normality_pvalue > 0.05

        # Outlier detection (>3 SD)
        z_scores = np.abs((ate_estimates - mean_ate) / std_ate)
        outlier_mask = z_scores > 3
        n_outliers = int(np.sum(outlier_mask))
        has_outliers = n_outliers > 0

        # Convergence check (CV of running mean < 1%)
        running_means = np.cumsum(ate_estimates) / np.arange(1, n_bootstrap + 1)
        cv_running = np.std(running_means[-100:]) / np.abs(np.mean(running_means[-100:]))
        converged = cv_running < 0.01

        self._log(f"  Mean ATE: ${mean_ate:,.2f}")
        self._log(f"  Std ATE: ${std_ate:,.2f}")
        self._log(f"  95% CI: (${ci_lower:,.2f}, ${ci_upper:,.2f})")
        self._log(f"  Shapiro-Wilk p: {normality_pvalue:.4f}")
        self._log(f"  Outliers: {n_outliers}/{n_bootstrap} ({100 * n_outliers / n_bootstrap:.1f}%)")
        self._log(f"  Last 100 cumulative means: CV={100 * cv_running:.2f}%")

        return BootstrapDiagnosticResult(
            ate_estimates=ate_estimates,
            mean_ate=mean_ate,
            std_ate=std_ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            normality_pvalue=float(normality_pvalue),
            is_normal=is_normal,
            has_outliers=has_outliers,
            n_outliers=n_outliers,
            converged=converged,
            n_bootstrap=n_bootstrap,
            metadata={
                "cv_running_mean": float(cv_running),
                "outlier_threshold": 3.0,
                "normality_test": "Shapiro-Wilk",
                "resampling_unit": "original household row",
                "folds": "five fixed stratified original-household folds",
            },
        )

    def analyze_hyperparameter_sensitivity(
        self, parameter: str, values: list[Any] | None = None
    ) -> HyperparameterSensitivityResult:
        """Analyze ATE sensitivity to hyperparameter changes.

        Tests how ATE estimate varies with different hyperparameter values.

        Args:
            parameter: Parameter to vary (cv_folds, max_iter)
            values: List of values to test (uses defaults if None)

        Returns:
            HyperparameterSensitivityResult with sensitivity analysis

        Examples:
            Requires the real 401(k) dataset and an econml DML fit per value.
            Note ``parameter`` must be one of "cv_folds" or "max_iter":

            >>> diagnostic = LassoDiagnostic(random_state=42)  # doctest: +SKIP
            >>> result = diagnostic.analyze_hyperparameter_sensitivity(  # doctest: +SKIP
            ...     parameter="max_iter", values=[500, 1000, 2000]
            ... )
            >>> result.is_sensitive  # doctest: +SKIP
            True
            >>> result.sensitivity_score  # doctest: +SKIP
            0.25
        """
        # Default values for each parameter
        default_values = {
            "cv_folds": [3, 5, 10],
            "max_iter": [500, 1000, 2000, 5000],
        }

        # Validate parameter before expensive imports (fail fast)
        if values is None:
            if parameter not in default_values:
                raise ValueError(
                    f"Unknown parameter: {parameter}. Must be one of {list(default_values.keys())}"
                )
            values = default_values[parameter]

        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV

        self._log(f"Analyzing {parameter} sensitivity ({len(values)} values)...")

        Y, T, X = self.preprocess_data()

        ate_estimates = np.zeros(len(values))
        std_errors = np.zeros(len(values))

        for i, value in enumerate(values):
            self._log(f"  Testing {parameter}={value}")

            # Configure models based on parameter
            # FIXED (Issue C2): Import LogisticRegressionCV
            from sklearn.linear_model import LogisticRegressionCV

            if parameter == "cv_folds":
                model_y = LassoCV(cv=value, random_state=self.random_state)
                # FIXED (Issue C2): Use logistic regression for binary treatment
                model_t = LogisticRegressionCV(
                    cv=value,
                    penalty="l1",
                    solver="saga",
                    random_state=self.random_state,
                    max_iter=1000,
                )
                cv = value
            elif parameter == "max_iter":
                model_y = LassoCV(cv=5, max_iter=value, random_state=self.random_state)
                # FIXED (Issue C2): Use logistic regression for binary treatment
                model_t = LogisticRegressionCV(
                    cv=5,
                    penalty="l1",
                    solver="saga",
                    max_iter=value,
                    random_state=self.random_state,
                )
                cv = 5
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            # Fit DML
            # FIXED (Issue C2): discrete_treatment=True for binary treatment
            dml = LinearDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                cv=cv,
                random_state=self.random_state,
            )

            dml.fit(Y=Y, T=T, X=X)

            # Extract ATE and SE
            inference = dml.ate_inference(X=X)
            ate = float(np.squeeze(inference.mean_point))
            se = float(np.squeeze(inference.stderr_mean))

            ate_estimates[i] = ate
            std_errors[i] = se

            self._log(f"    ATE: ${ate:,.2f} (SE: ${se:,.2f})")

        # Sensitivity analysis
        mean_ate = np.mean(ate_estimates)
        std_ate = np.std(ate_estimates, ddof=1)
        sensitivity_score = float(std_ate / np.abs(mean_ate)) if mean_ate != 0 else np.inf
        is_sensitive = sensitivity_score > 0.1

        # Descriptive minimum of the fitted standard errors across tested settings.
        recommended_idx = np.argmin(std_errors)
        recommended_value = values[recommended_idx]

        self._log(f"  Sensitivity: {sensitivity_score:.3f} ({'HIGH' if is_sensitive else 'LOW'})")
        self._log(f"  {parameter} with smallest fitted SE: {recommended_value}")

        return HyperparameterSensitivityResult(
            parameter_name=parameter,
            parameter_values=np.array(values),
            ate_estimates=ate_estimates,
            std_errors=std_errors,
            sensitivity_score=sensitivity_score,
            is_sensitive=is_sensitive,
            recommended_value=recommended_value,
            metadata={
                "mean_ate": float(mean_ate),
                "std_ate": float(std_ate),
                "min_ate": float(np.min(ate_estimates)),
                "max_ate": float(np.max(ate_estimates)),
            },
        )

    def analyze_seed_sensitivity(self, n_seeds: int = 20) -> SeedSensitivityResult:
        """Analyze ATE sensitivity to cross-fitting random seed.

        Tests how ATE varies across different random_state values for cross-fitting.

        Args:
            n_seeds: Number of random seeds to test

        Returns:
            SeedSensitivityResult with seed sensitivity analysis

        Examples:
            Requires the real 401(k) dataset and an econml DML fit per seed:

            >>> diagnostic = LassoDiagnostic()  # doctest: +SKIP
            >>> result = diagnostic.analyze_seed_sensitivity(n_seeds=20)  # doctest: +SKIP
            >>> result.is_stable  # doctest: +SKIP
            False
            >>> result.cv_ate  # doctest: +SKIP
            0.35
        """
        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold

        self._log(f"Analyzing seed sensitivity ({n_seeds} seeds)...")

        Y, T, X = self.preprocess_data()

        # Test range of seeds
        random_states = np.arange(n_seeds)
        ate_estimates = np.zeros(n_seeds)

        for i, seed in enumerate(random_states):
            self._log(f"  Testing random_state={seed}")

            model_y = LassoCV(cv=5, random_state=self.random_state)

            model_t = LogisticRegressionCV(
                cv=5, penalty="l1", solver="saga", random_state=self.random_state, max_iter=1000
            )

            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
            dml = LinearDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                cv=list(splitter.split(X, T)),
                random_state=self.random_state,
            )

            dml.fit(Y=Y, T=T, X=X)
            ate_estimates[i] = float(dml.effect(X=X).mean())

            self._log(f"    ATE: ${ate_estimates[i]:,.2f}")

        # Stability analysis
        mean_ate = float(np.mean(ate_estimates))
        std_ate = float(np.std(ate_estimates, ddof=1))
        min_ate = float(np.min(ate_estimates))
        max_ate = float(np.max(ate_estimates))
        range_ate = max_ate - min_ate
        cv_ate = float(std_ate / np.abs(mean_ate)) if mean_ate != 0 else np.inf
        is_stable = cv_ate < 0.1

        self._log(f"  Mean ATE: ${mean_ate:,.2f} ± ${std_ate:,.2f}")
        self._log(f"  Range: ${min_ate:,.2f} to ${max_ate:,.2f} (span: ${range_ate:,.2f})")
        self._log(f"  CV: {cv_ate:.3f} ({'STABLE' if is_stable else 'UNSTABLE'})")

        return SeedSensitivityResult(
            random_states=random_states,
            ate_estimates=ate_estimates,
            mean_ate=mean_ate,
            std_ate=std_ate,
            min_ate=min_ate,
            max_ate=max_ate,
            range_ate=range_ate,
            cv_ate=cv_ate,
            is_stable=is_stable,
            metadata={
                "n_seeds": n_seeds,
                "learner_seed": self.random_state,
                "fold_design": "5 shuffled treatment-stratified folds",
                "stability_threshold": 0.1,
            },
        )

    def run_comprehensive_diagnostic(
        self, n_bootstrap: int = 1000, n_seeds: int = 20
    ) -> ComprehensiveDiagnosticResult:
        """Run the bootstrap, tuning and split-sensitivity diagnostics.

        Performs:
        1. Bootstrap distribution analysis
        2. Hyperparameter sensitivity (cv_folds, max_iter)
        3. Cross-fitting seed analysis
        4. Summary of observed diagnostic signals
        5. Recommendations

        Args:
            n_bootstrap: Number of bootstrap samples
            n_seeds: Number of seeds to test

        Returns:
            ComprehensiveDiagnosticResult with all analyses

        Examples:
            Requires the real 401(k) dataset and many econml DML fits:

            >>> diagnostic = LassoDiagnostic(random_state=42, verbose=True)  # doctest: +SKIP
            >>> results = diagnostic.run_comprehensive_diagnostic()  # doctest: +SKIP
            >>> print(results.root_cause_analysis)  # doctest: +SKIP
            >>> for rec in results.recommendations:  # doctest: +SKIP
            ...     print(f"- {rec}")
        """
        self._log("=" * 80)
        self._log("COMPREHENSIVE LASSO DIAGNOSTIC ANALYSIS")
        self._log("=" * 80)
        self._log("")

        # 1. Bootstrap distribution
        self._log("PHASE 1: Bootstrap Distribution Analysis")
        self._log("-" * 80)
        bootstrap_result = self.analyze_bootstrap_distribution(n_bootstrap=n_bootstrap)
        self._log("")

        # 2. Hyperparameter sensitivity
        self._log("PHASE 2: Hyperparameter Sensitivity Analysis")
        self._log("-" * 80)
        hyperparameter_results = {}

        for param in ["cv_folds", "max_iter"]:
            hyperparameter_results[param] = self.analyze_hyperparameter_sensitivity(param)
        self._log("")

        # 3. Seed sensitivity
        self._log("PHASE 3: Cross-Fitting Seed Sensitivity")
        self._log("-" * 80)
        seed_result = self.analyze_seed_sensitivity(n_seeds=n_seeds)
        self._log("")

        # 4. Root cause analysis
        self._log("PHASE 4: Diagnostic Findings")
        self._log("-" * 80)
        root_cause, recommendations = self._analyze_root_cause(
            bootstrap_result, hyperparameter_results, seed_result
        )
        self._log(f"Diagnostic findings: {root_cause}")
        self._log("")
        self._log("Recommendations:")
        for rec in recommendations:
            self._log(f"  - {rec}")
        self._log("")

        return ComprehensiveDiagnosticResult(
            bootstrap_diagnostic=bootstrap_result,
            hyperparameter_sensitivity=hyperparameter_results,
            seed_sensitivity=seed_result,
            root_cause_analysis=root_cause,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

    def _analyze_root_cause(
        self,
        bootstrap: BootstrapDiagnosticResult,
        hyperparameters: dict[str, HyperparameterSensitivityResult],
        seed: SeedSensitivityResult,
    ) -> tuple[str, list[str]]:
        """Summarize the observed diagnostic signals and corresponding checks.

        Args:
            bootstrap: Bootstrap distribution analysis
            hyperparameters: Hyperparameter sensitivity results
            seed: Seed sensitivity results

        Returns:
            Tuple of (root_cause_description, recommendations_list)
        """
        causes = []
        recommendations = []

        # Check bootstrap convergence
        if not bootstrap.converged:
            causes.append("CV of the last 100 cumulative bootstrap means reaches 1%")
            recommendations.append(f"Increase bootstrap samples (current: {bootstrap.n_bootstrap})")

        # Check normality
        if not bootstrap.is_normal:
            causes.append("Shapiro-Wilk rejects normality of the bootstrap estimates at 5%")
            recommendations.append(
                "Inspect the shape of the reported percentile bootstrap distribution"
            )

        # Check outliers
        if bootstrap.has_outliers:
            causes.append(
                f"Bootstrap distribution has {bootstrap.n_outliers} outliers "
                f"({100 * bootstrap.n_outliers / bootstrap.n_bootstrap:.1f}%)"
            )
            recommendations.append(
                "Inspect input rows and convergence for the extreme resampled fits"
            )

        # Check hyperparameter sensitivity
        sensitive_params = [name for name, result in hyperparameters.items() if result.is_sensitive]
        if sensitive_params:
            causes.append(f"High sensitivity to hyperparameters: {', '.join(sensitive_params)}")
            for param in sensitive_params:
                rec_value = hyperparameters[param].recommended_value
                recommendations.append(
                    f"Report all tested {param} values; {rec_value} gave the smallest fitted SE"
                )

        # Check seed stability
        if not seed.is_stable:
            causes.append(
                f"High cross-fitting seed variability (CV={seed.cv_ate:.3f}, "
                f"range=${seed.range_ate:,.2f})"
            )
            recommendations.append("Average results over multiple random_state values (ensemble)")

        # Synthesize root cause
        if len(causes) == 0:
            root_cause = "All assessed stability diagnostics met their configured thresholds."
        elif len(causes) == 1:
            root_cause = causes[0]
        else:
            root_cause = "Multiple issues: " + "; ".join(causes)

        # Additional recommendations
        recommendations.append(
            "Compare Lasso solver and convergence settings with published implementation"
        )
        recommendations.append("Report the fitted effect model, covariates and split design.")

        return root_cause, recommendations
