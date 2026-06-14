"""
Causal-specific monitoring utilities for DML research/demo pipelines.

This module monitors phenomena UNIQUE to causal inference:

1. Treatment Distribution Shift
   - Standard ML: monitors feature drift
   - Causal: ALSO monitors if treatment assignment mechanism changed

2. Overlap Violations (Positivity)
   - Standard ML: N/A
   - Causal: Propensity scores near 0 or 1 indicate extrapolation

3. Nuisance Model Degradation
   - Standard ML: monitors prediction accuracy
   - Causal: monitors nuisance R² (NOT causal accuracy - that's unknowable!)

4. Effect Stability
   - Standard ML: monitors prediction stability
   - Causal: monitors if treatment effect itself is changing over time

Key insight: In causal inference, we CANNOT directly monitor treatment effect
accuracy because ground truth counterfactuals are unobservable. We can only
monitor the *conditions* under which our estimates are valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats


class AlertLevel(Enum):
    """Severity levels for monitoring alerts."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MonitoringResult:
    """
    Result from a single monitoring check.

    Attributes:
        check_name: Name of the check performed
        level: Alert severity level
        message: Human-readable description
        value: Numeric value of the metric
        threshold: Threshold that triggered the alert (if any)
        timestamp: When the check was performed
        details: Additional diagnostic information
    """

    check_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def is_ok(self) -> bool:
        """Return True if no issues detected."""
        return self.level == AlertLevel.OK

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_name": self.check_name,
            "level": self.level.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class CausalMonitorConfig:
    """
    Configuration for CausalMonitor thresholds.

    Attributes:
        propensity_clip_min: Minimum acceptable propensity score
        propensity_clip_max: Maximum acceptable propensity score
        overlap_violation_rate_warning: Warning threshold for overlap violations
        overlap_violation_rate_critical: Critical threshold for overlap violations
        treatment_shift_warning: Warning threshold for treatment distribution change
        treatment_shift_critical: Critical threshold for treatment distribution change
        nuisance_r2_warning: Warning threshold for nuisance R² degradation
        nuisance_r2_critical: Critical threshold for nuisance R² degradation
        effect_change_warning: Warning threshold for effect magnitude change
        effect_change_critical: Critical threshold for effect magnitude change
    """

    propensity_clip_min: float = 0.01
    propensity_clip_max: float = 0.99
    overlap_violation_rate_warning: float = 0.05
    overlap_violation_rate_critical: float = 0.10
    treatment_shift_warning: float = 0.05  # KS statistic
    treatment_shift_critical: float = 0.10
    nuisance_r2_warning: float = 0.50
    nuisance_r2_critical: float = 0.30
    effect_change_warning: float = 0.20  # 20% relative change
    effect_change_critical: float = 0.50  # 50% relative change
    covariate_shift_warning: float = 0.05  # KS statistic per feature
    covariate_shift_critical: float = 0.10


class CausalMonitor:
    """
    Companion monitoring utility for DML pipelines.

    Monitors four causal-specific phenomena:
    1. Overlap violations (positivity assumption)
    2. Treatment distribution shift
    3. Nuisance model degradation
    4. Effect stability over time

    Unlike standard ML monitoring, this class tracks conditions required
    for causal identification, not predictive accuracy.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e_hat = rng.uniform(0.1, 0.9, size=200)
        >>> T = rng.binomial(1, 0.5, size=200)
        >>> T_train = rng.binomial(1, 0.5, size=200)
        >>> monitor = CausalMonitor()
        >>> results = monitor.run_all_checks(
        ...     propensity_scores=e_hat,
        ...     treatment_current=T,
        ...     treatment_baseline=T_train,
        ...     r2_propensity=0.65,
        ...     r2_outcome=0.72,
        ...     current_effect=0.15,
        ...     baseline_effect=0.12,
        ... )
        >>> all(isinstance(r.is_ok(), bool) for r in results)
        True
    """

    def __init__(self, config: CausalMonitorConfig | None = None):
        """
        Initialize causal monitor.

        Args:
            config: Monitoring configuration (uses defaults if None)
        """
        self.config = config or CausalMonitorConfig()
        self._history: list[list[MonitoringResult]] = []

    def check_overlap_violations(
        self,
        propensity_scores: np.ndarray,
    ) -> MonitoringResult:
        """
        Check for overlap (positivity) violations.

        Overlap requires P(T=1|X) bounded away from 0 and 1. Violations
        indicate regions where treatment effect is extrapolated, not estimated.

        Args:
            propensity_scores: Estimated P(T=1|X) for inference sample

        Returns:
            MonitoringResult with violation rate and severity
        """
        propensity_scores = np.asarray(propensity_scores).flatten()
        n_samples = len(propensity_scores)

        # Count violations
        too_low = propensity_scores < self.config.propensity_clip_min
        too_high = propensity_scores > self.config.propensity_clip_max
        n_violations = np.sum(too_low) + np.sum(too_high)
        violation_rate = n_violations / n_samples

        # Determine severity
        if violation_rate >= self.config.overlap_violation_rate_critical:
            level = AlertLevel.CRITICAL
            message = (
                f"CRITICAL: {violation_rate:.1%} overlap violations "
                f"(>{self.config.overlap_violation_rate_critical:.1%}). "
                "Treatment effects may be unreliable for extreme propensity regions."
            )
        elif violation_rate >= self.config.overlap_violation_rate_warning:
            level = AlertLevel.WARNING
            message = (
                f"WARNING: {violation_rate:.1%} overlap violations. "
                "Consider trimming or IPW adjustment."
            )
        else:
            level = AlertLevel.OK
            message = f"Overlap OK: {violation_rate:.1%} violations (within tolerance)"

        return MonitoringResult(
            check_name="overlap_violations",
            level=level,
            message=message,
            value=violation_rate,
            threshold=self.config.overlap_violation_rate_warning,
            details={
                "n_samples": n_samples,
                "n_violations": int(n_violations),
                "n_too_low": int(np.sum(too_low)),
                "n_too_high": int(np.sum(too_high)),
                "propensity_min": float(np.min(propensity_scores)),
                "propensity_max": float(np.max(propensity_scores)),
                "propensity_mean": float(np.mean(propensity_scores)),
            },
        )

    def check_treatment_shift(
        self,
        treatment_current: np.ndarray,
        treatment_baseline: np.ndarray,
    ) -> MonitoringResult:
        """
        Check for treatment distribution shift.

        Unlike feature drift in standard ML, treatment shift affects causal
        identification. If treatment assignment mechanism changed, historical
        effects may not transfer.

        Uses KS statistic for continuous, chi-squared for binary treatment.

        Args:
            treatment_current: Current period treatment values
            treatment_baseline: Baseline period treatment values

        Returns:
            MonitoringResult with shift magnitude and severity
        """
        treatment_current = np.asarray(treatment_current).flatten()
        treatment_baseline = np.asarray(treatment_baseline).flatten()

        # Determine if binary or continuous
        unique_current = np.unique(treatment_current)
        unique_baseline = np.unique(treatment_baseline)

        if len(unique_current) <= 2 and len(unique_baseline) <= 2:
            # Binary treatment - compare proportions
            p_current = np.mean(treatment_current)
            p_baseline = np.mean(treatment_baseline)
            diff = abs(p_current - p_baseline)

            # Use proportion difference as metric
            metric_value = diff
            metric_name = "proportion_diff"

            details = {
                "p_current": float(p_current),
                "p_baseline": float(p_baseline),
                "absolute_diff": float(diff),
                "metric": metric_name,
            }
        else:
            # Continuous treatment - KS test
            ks_stat, ks_pval = stats.ks_2samp(treatment_current, treatment_baseline)
            metric_value = ks_stat
            metric_name = "ks_statistic"

            details = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "mean_current": float(np.mean(treatment_current)),
                "mean_baseline": float(np.mean(treatment_baseline)),
                "std_current": float(np.std(treatment_current)),
                "std_baseline": float(np.std(treatment_baseline)),
                "metric": metric_name,
            }

        # Determine severity
        if metric_value >= self.config.treatment_shift_critical:
            level = AlertLevel.CRITICAL
            message = (
                f"CRITICAL: Treatment distribution shift detected "
                f"({metric_name}={metric_value:.3f} > {self.config.treatment_shift_critical}). "
                "Treatment assignment mechanism may have changed."
            )
        elif metric_value >= self.config.treatment_shift_warning:
            level = AlertLevel.WARNING
            message = (
                f"WARNING: Moderate treatment shift "
                f"({metric_name}={metric_value:.3f}). Monitor closely."
            )
        else:
            level = AlertLevel.OK
            message = f"Treatment distribution stable ({metric_name}={metric_value:.3f})"

        return MonitoringResult(
            check_name="treatment_shift",
            level=level,
            message=message,
            value=metric_value,
            threshold=self.config.treatment_shift_warning,
            details=details,
        )

    def check_nuisance_degradation(
        self,
        r2_propensity: float,
        r2_outcome: float,
        r2_propensity_baseline: float | None = None,
        r2_outcome_baseline: float | None = None,
    ) -> MonitoringResult:
        """
        Check for nuisance model degradation.

        DML requires well-fitting nuisance models for valid inference.
        Poor nuisance fit leads to biased treatment effects.

        Note: This checks nuisance R², NOT treatment effect accuracy
        (which is fundamentally unknowable).

        Args:
            r2_propensity: Current propensity model R² (or pseudo-R² for classification)
            r2_outcome: Current outcome model R²
            r2_propensity_baseline: Optional baseline propensity R² for comparison
            r2_outcome_baseline: Optional baseline outcome R² for comparison

        Returns:
            MonitoringResult with degradation assessment
        """
        # Use minimum of the two as the overall quality metric
        min_r2 = min(r2_propensity, r2_outcome)

        details = {
            "r2_propensity": float(r2_propensity),
            "r2_outcome": float(r2_outcome),
        }

        # Check absolute thresholds
        if min_r2 < self.config.nuisance_r2_critical:
            level = AlertLevel.CRITICAL
            message = (
                f"CRITICAL: Nuisance model fit very poor "
                f"(min R²={min_r2:.2f} < {self.config.nuisance_r2_critical}). "
                "DML estimates may be severely biased."
            )
        elif min_r2 < self.config.nuisance_r2_warning:
            level = AlertLevel.WARNING
            message = (
                f"WARNING: Nuisance model fit degraded "
                f"(min R²={min_r2:.2f} < {self.config.nuisance_r2_warning}). "
                "Consider retraining."
            )
        else:
            level = AlertLevel.OK
            message = (
                f"Nuisance fit OK (R² propensity={r2_propensity:.2f}, outcome={r2_outcome:.2f})"
            )

        # Check relative degradation if baselines provided
        if r2_propensity_baseline is not None:
            prop_degradation = (r2_propensity_baseline - r2_propensity) / r2_propensity_baseline
            details["propensity_degradation"] = float(prop_degradation)
            if prop_degradation > 0.2:
                if level == AlertLevel.OK:
                    level = AlertLevel.WARNING
                message += f" Propensity degraded {prop_degradation:.1%} from baseline."

        if r2_outcome_baseline is not None:
            out_degradation = (r2_outcome_baseline - r2_outcome) / r2_outcome_baseline
            details["outcome_degradation"] = float(out_degradation)
            if out_degradation > 0.2:
                if level == AlertLevel.OK:
                    level = AlertLevel.WARNING
                message += f" Outcome degraded {out_degradation:.1%} from baseline."

        return MonitoringResult(
            check_name="nuisance_degradation",
            level=level,
            message=message,
            value=min_r2,
            threshold=self.config.nuisance_r2_warning,
            details=details,
        )

    def check_effect_stability(
        self,
        current_effect: float,
        baseline_effect: float,
        current_se: float | None = None,
        baseline_se: float | None = None,
    ) -> MonitoringResult:
        """
        Check for treatment effect stability over time.

        Large changes in estimated effects may indicate:
        - Genuine effect heterogeneity over time (valid)
        - Model degradation or data issues (problematic)
        - Selection/survivor bias in new data (problematic)

        Args:
            current_effect: Current period ATE estimate
            baseline_effect: Baseline period ATE estimate
            current_se: Optional standard error for current estimate
            baseline_se: Optional standard error for baseline estimate

        Returns:
            MonitoringResult with effect change assessment
        """
        # Compute relative change (handle zero baseline)
        if abs(baseline_effect) < 1e-10:
            if abs(current_effect) < 1e-10:
                relative_change = 0.0
            else:
                relative_change = float("inf")
        else:
            relative_change = abs(current_effect - baseline_effect) / abs(baseline_effect)

        details = {
            "current_effect": float(current_effect),
            "baseline_effect": float(baseline_effect),
            "absolute_change": float(current_effect - baseline_effect),
            "relative_change": float(relative_change),
        }

        # If standard errors provided, compute significance of change
        if current_se is not None and baseline_se is not None:
            combined_se = np.sqrt(current_se**2 + baseline_se**2)
            z_score = (current_effect - baseline_effect) / combined_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            details["z_score"] = float(z_score)
            details["p_value"] = float(p_value)
            details["current_se"] = float(current_se)
            details["baseline_se"] = float(baseline_se)
            details["statistically_significant"] = p_value < 0.05

        # Determine severity
        if relative_change >= self.config.effect_change_critical:
            level = AlertLevel.CRITICAL
            message = (
                f"CRITICAL: Treatment effect changed {relative_change:.0%} "
                f"({baseline_effect:.3f} -> {current_effect:.3f}). "
                "Investigate cause before deployment."
            )
        elif relative_change >= self.config.effect_change_warning:
            level = AlertLevel.WARNING
            message = (
                f"WARNING: Treatment effect changed {relative_change:.0%}. "
                "May indicate effect heterogeneity or model issues."
            )
        else:
            level = AlertLevel.OK
            message = (
                f"Effect stable: {relative_change:.1%} change "
                f"({baseline_effect:.3f} -> {current_effect:.3f})"
            )

        return MonitoringResult(
            check_name="effect_stability",
            level=level,
            message=message,
            value=relative_change,
            threshold=self.config.effect_change_warning,
            details=details,
        )

    def check_covariate_shift(
        self,
        X_current: np.ndarray,
        X_baseline: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> MonitoringResult:
        """
        Check for covariate distribution shift.

        Covariate shift affects:
        1. Nuisance model accuracy (may need retraining)
        2. Effect generalization (extrapolation risk)
        3. Overlap (new covariate regions may have extreme propensity)

        Args:
            X_current: Current period covariates (n_samples, n_features)
            X_baseline: Baseline period covariates (n_samples, n_features)
            feature_names: Optional names for features

        Returns:
            MonitoringResult with shift assessment for each feature
        """
        X_current = np.asarray(X_current)
        X_baseline = np.asarray(X_baseline)

        if X_current.ndim == 1:
            X_current = X_current.reshape(-1, 1)
            X_baseline = X_baseline.reshape(-1, 1)

        n_features = X_current.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Compute KS statistic for each feature
        ks_stats = []
        feature_shifts = {}
        for i, name in enumerate(feature_names):
            ks_stat, ks_pval = stats.ks_2samp(X_current[:, i], X_baseline[:, i])
            ks_stats.append(ks_stat)
            feature_shifts[name] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "mean_current": float(np.mean(X_current[:, i])),
                "mean_baseline": float(np.mean(X_baseline[:, i])),
            }

        max_ks = max(ks_stats)
        n_warning = sum(1 for ks in ks_stats if ks >= self.config.covariate_shift_warning)
        n_critical = sum(1 for ks in ks_stats if ks >= self.config.covariate_shift_critical)

        # Determine overall severity
        if n_critical > 0:
            level = AlertLevel.CRITICAL
            message = (
                f"CRITICAL: {n_critical} features show significant covariate shift "
                f"(max KS={max_ks:.3f}). Nuisance models may need retraining."
            )
        elif n_warning > 0:
            level = AlertLevel.WARNING
            message = (
                f"WARNING: {n_warning} features show moderate covariate shift. "
                "Monitor nuisance model performance."
            )
        else:
            level = AlertLevel.OK
            message = f"Covariate distribution stable (max KS={max_ks:.3f})"

        return MonitoringResult(
            check_name="covariate_shift",
            level=level,
            message=message,
            value=max_ks,
            threshold=self.config.covariate_shift_warning,
            details={
                "n_features": n_features,
                "n_warning": n_warning,
                "n_critical": n_critical,
                "feature_shifts": feature_shifts,
            },
        )

    def run_all_checks(
        self,
        propensity_scores: np.ndarray | None = None,
        treatment_current: np.ndarray | None = None,
        treatment_baseline: np.ndarray | None = None,
        r2_propensity: float | None = None,
        r2_outcome: float | None = None,
        current_effect: float | None = None,
        baseline_effect: float | None = None,
        X_current: np.ndarray | None = None,
        X_baseline: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        r2_propensity_baseline: float | None = None,
        r2_outcome_baseline: float | None = None,
        current_se: float | None = None,
        baseline_se: float | None = None,
    ) -> list[MonitoringResult]:
        """
        Run all applicable monitoring checks.

        Skips checks where required inputs are not provided.

        Args:
            propensity_scores: For overlap check
            treatment_current: For treatment shift check
            treatment_baseline: For treatment shift check
            r2_propensity: For nuisance degradation check
            r2_outcome: For nuisance degradation check
            current_effect: For effect stability check
            baseline_effect: For effect stability check
            X_current: For covariate shift check
            X_baseline: For covariate shift check
            feature_names: Feature names for covariate shift
            r2_propensity_baseline: Optional baseline propensity R² (nuisance check)
            r2_outcome_baseline: Optional baseline outcome R² (nuisance check)
            current_se: Optional SE for the current effect (stability check)
            baseline_se: Optional SE for the baseline effect (stability check)

        Returns:
            List of MonitoringResult for all executed checks
        """
        results = []

        # Overlap check
        if propensity_scores is not None:
            results.append(self.check_overlap_violations(propensity_scores))

        # Treatment shift check
        if treatment_current is not None and treatment_baseline is not None:
            results.append(self.check_treatment_shift(treatment_current, treatment_baseline))

        # Nuisance degradation check
        if r2_propensity is not None and r2_outcome is not None:
            results.append(
                self.check_nuisance_degradation(
                    r2_propensity,
                    r2_outcome,
                    r2_propensity_baseline,
                    r2_outcome_baseline,
                )
            )

        # Effect stability check
        if current_effect is not None and baseline_effect is not None:
            results.append(
                self.check_effect_stability(
                    current_effect,
                    baseline_effect,
                    current_se,
                    baseline_se,
                )
            )

        # Covariate shift check
        if X_current is not None and X_baseline is not None:
            results.append(self.check_covariate_shift(X_current, X_baseline, feature_names))

        # Store in history
        self._history.append(results)

        return results

    def get_summary(self, results: list[MonitoringResult] | None = None) -> dict[str, Any]:
        """
        Get summary of monitoring results.

        Args:
            results: Results to summarize (defaults to most recent)

        Returns:
            Dict with overall status and check summaries
        """
        if results is None:
            if not self._history:
                return {"status": "no_data", "checks": []}
            results = self._history[-1]

        # Compute overall status
        levels = [r.level for r in results]
        if AlertLevel.CRITICAL in levels:
            overall = AlertLevel.CRITICAL
        elif AlertLevel.WARNING in levels:
            overall = AlertLevel.WARNING
        else:
            overall = AlertLevel.OK

        return {
            "status": overall.value,
            "n_checks": len(results),
            "n_ok": sum(1 for r in results if r.level == AlertLevel.OK),
            "n_warning": sum(1 for r in results if r.level == AlertLevel.WARNING),
            "n_critical": sum(1 for r in results if r.level == AlertLevel.CRITICAL),
            "checks": [r.to_dict() for r in results],
        }

    def __repr__(self) -> str:
        return f"CausalMonitor(n_checks_run={len(self._history)})"
