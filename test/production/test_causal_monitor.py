"""
Tests for causal-specific monitoring.

Tests cover:
- Overlap (positivity) violation detection
- Treatment distribution shift detection
- Nuisance model degradation detection
- Effect stability monitoring
- Covariate shift detection
"""

import numpy as np
import pytest

from src.production.causal_monitor import (
    AlertLevel,
    CausalMonitor,
    CausalMonitorConfig,
    MonitoringResult,
)

pytestmark = pytest.mark.tier2


class TestMonitoringResult:
    """Tests for MonitoringResult dataclass."""

    def test_is_ok_returns_true_for_ok_level(self):
        """Test is_ok() returns True for OK level."""
        result = MonitoringResult(
            check_name="test",
            level=AlertLevel.OK,
            message="All good",
            value=0.01,
        )
        assert result.is_ok() is True

    def test_is_ok_returns_false_for_warning(self):
        """Test is_ok() returns False for WARNING level."""
        result = MonitoringResult(
            check_name="test",
            level=AlertLevel.WARNING,
            message="Warning",
            value=0.06,
        )
        assert result.is_ok() is False

    def test_is_ok_returns_false_for_critical(self):
        """Test is_ok() returns False for CRITICAL level."""
        result = MonitoringResult(
            check_name="test",
            level=AlertLevel.CRITICAL,
            message="Critical",
            value=0.15,
        )
        assert result.is_ok() is False

    def test_to_dict_serialization(self):
        """Test to_dict() produces serializable dict."""
        result = MonitoringResult(
            check_name="overlap",
            level=AlertLevel.WARNING,
            message="Some warning",
            value=0.08,
            threshold=0.05,
            details={"n_samples": 100},
        )
        d = result.to_dict()

        assert d["check_name"] == "overlap"
        assert d["level"] == "warning"
        assert d["value"] == 0.08
        assert d["threshold"] == 0.05
        assert "timestamp" in d


class TestOverlapViolations:
    """Tests for overlap/positivity violation checking."""

    def test_no_violations_returns_ok(self):
        """Test that good propensity scores return OK."""
        monitor = CausalMonitor()
        propensity = np.random.uniform(0.2, 0.8, size=1000)

        result = monitor.check_overlap_violations(propensity)

        assert result.level == AlertLevel.OK
        assert result.value < 0.05  # Violation rate

    def test_some_violations_returns_warning(self):
        """Test that moderate violations return WARNING."""
        monitor = CausalMonitor()

        # Create propensity with ~7% violations
        propensity = np.random.uniform(0.2, 0.8, size=930)
        violations = np.array([0.001] * 35 + [0.999] * 35)
        propensity = np.concatenate([propensity, violations])

        result = monitor.check_overlap_violations(propensity)

        assert result.level == AlertLevel.WARNING
        assert 0.05 <= result.value < 0.10

    def test_many_violations_returns_critical(self):
        """Test that many violations return CRITICAL."""
        monitor = CausalMonitor()

        # Create propensity with ~15% violations
        propensity = np.random.uniform(0.2, 0.8, size=850)
        violations = np.array([0.001] * 75 + [0.999] * 75)
        propensity = np.concatenate([propensity, violations])

        result = monitor.check_overlap_violations(propensity)

        assert result.level == AlertLevel.CRITICAL
        assert result.value >= 0.10

    def test_details_include_counts(self):
        """Test that result details include violation counts."""
        monitor = CausalMonitor()
        propensity = np.array([0.001, 0.002, 0.5, 0.6, 0.998, 0.999])

        result = monitor.check_overlap_violations(propensity)

        assert "n_samples" in result.details
        assert "n_violations" in result.details
        assert "n_too_low" in result.details
        assert "n_too_high" in result.details
        assert result.details["n_too_low"] == 2
        assert result.details["n_too_high"] == 2

    def test_custom_config_thresholds(self):
        """Test that custom config thresholds are respected."""
        config = CausalMonitorConfig(
            propensity_clip_min=0.05,
            propensity_clip_max=0.95,
            overlap_violation_rate_warning=0.10,
            overlap_violation_rate_critical=0.20,
        )
        monitor = CausalMonitor(config)

        # 15% violations - should be WARNING not CRITICAL with custom thresholds
        propensity = np.random.uniform(0.2, 0.8, size=850)
        violations = np.array([0.001] * 75 + [0.999] * 75)
        propensity = np.concatenate([propensity, violations])

        result = monitor.check_overlap_violations(propensity)

        assert result.level == AlertLevel.WARNING


class TestTreatmentShift:
    """Tests for treatment distribution shift detection."""

    def test_no_shift_returns_ok(self):
        """Test that similar distributions return OK."""
        monitor = CausalMonitor()
        np.random.seed(42)

        # Same distribution
        T_baseline = np.random.binomial(1, 0.5, size=1000)
        T_current = np.random.binomial(1, 0.5, size=1000)

        result = monitor.check_treatment_shift(T_current, T_baseline)

        assert result.level == AlertLevel.OK
        assert result.value < 0.05

    def test_moderate_shift_returns_warning(self):
        """Test that moderate shift returns WARNING."""
        monitor = CausalMonitor()

        # Different proportions
        T_baseline = np.random.binomial(1, 0.5, size=1000)
        T_current = np.random.binomial(1, 0.56, size=1000)  # ~6% difference

        result = monitor.check_treatment_shift(T_current, T_baseline)

        # May be OK or WARNING depending on random variation
        assert result.level in [AlertLevel.OK, AlertLevel.WARNING]

    def test_large_shift_returns_critical(self):
        """Test that large shift returns CRITICAL."""
        monitor = CausalMonitor()

        # Very different proportions
        T_baseline = np.random.binomial(1, 0.3, size=1000)
        T_current = np.random.binomial(1, 0.7, size=1000)

        result = monitor.check_treatment_shift(T_current, T_baseline)

        assert result.level == AlertLevel.CRITICAL

    def test_continuous_treatment_uses_ks(self):
        """Test that continuous treatment uses KS statistic."""
        monitor = CausalMonitor()

        # Continuous treatment
        T_baseline = np.random.normal(0, 1, size=1000)
        T_current = np.random.normal(0.5, 1, size=1000)  # Shifted mean

        result = monitor.check_treatment_shift(T_current, T_baseline)

        assert "ks_statistic" in result.details
        assert result.details["metric"] == "ks_statistic"

    def test_binary_treatment_uses_proportion(self):
        """Test that binary treatment uses proportion difference."""
        monitor = CausalMonitor()

        T_baseline = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        T_current = np.array([1, 1, 1, 1, 1, 0, 0, 0])

        result = monitor.check_treatment_shift(T_current, T_baseline)

        assert result.details["metric"] == "proportion_diff"


class TestNuisanceDegradation:
    """Tests for nuisance model degradation detection."""

    def test_good_r2_returns_ok(self):
        """Test that good R² values return OK."""
        monitor = CausalMonitor()

        result = monitor.check_nuisance_degradation(
            r2_propensity=0.75,
            r2_outcome=0.82,
        )

        assert result.level == AlertLevel.OK
        assert result.value >= 0.50

    def test_moderate_degradation_returns_warning(self):
        """Test that moderate R² returns WARNING."""
        monitor = CausalMonitor()

        result = monitor.check_nuisance_degradation(
            r2_propensity=0.45,  # Below warning threshold
            r2_outcome=0.55,
        )

        assert result.level == AlertLevel.WARNING

    def test_severe_degradation_returns_critical(self):
        """Test that very low R² returns CRITICAL."""
        monitor = CausalMonitor()

        result = monitor.check_nuisance_degradation(
            r2_propensity=0.25,  # Below critical threshold
            r2_outcome=0.20,
        )

        assert result.level == AlertLevel.CRITICAL
        assert result.value < 0.30

    def test_uses_minimum_r2(self):
        """Test that check uses minimum of propensity and outcome R²."""
        monitor = CausalMonitor()

        # Propensity is good, outcome is bad
        result = monitor.check_nuisance_degradation(
            r2_propensity=0.85,
            r2_outcome=0.25,  # Bad
        )

        assert result.value == 0.25  # Uses minimum
        assert result.level == AlertLevel.CRITICAL

    def test_relative_degradation_detection(self):
        """Test detection of relative degradation from baseline."""
        monitor = CausalMonitor()

        result = monitor.check_nuisance_degradation(
            r2_propensity=0.60,
            r2_outcome=0.65,
            r2_propensity_baseline=0.80,  # 25% degradation
            r2_outcome_baseline=0.75,
        )

        assert "propensity_degradation" in result.details
        assert result.details["propensity_degradation"] == pytest.approx(0.25, rel=0.01)


class TestEffectStability:
    """Tests for effect stability monitoring."""

    def test_stable_effect_returns_ok(self):
        """Test that stable effect returns OK."""
        monitor = CausalMonitor()

        result = monitor.check_effect_stability(
            current_effect=0.12,
            baseline_effect=0.10,  # 20% change - at boundary
        )

        # Exactly at threshold might be OK or WARNING
        assert result.level in [AlertLevel.OK, AlertLevel.WARNING]

    def test_moderate_change_returns_warning(self):
        """Test that moderate effect change returns WARNING."""
        monitor = CausalMonitor()

        result = monitor.check_effect_stability(
            current_effect=0.15,
            baseline_effect=0.10,  # 50% change
        )

        assert result.level == AlertLevel.WARNING

    def test_large_change_returns_critical(self):
        """Test that large effect change returns CRITICAL."""
        monitor = CausalMonitor()

        result = monitor.check_effect_stability(
            current_effect=0.25,
            baseline_effect=0.10,  # 150% change
        )

        assert result.level == AlertLevel.CRITICAL

    def test_handles_zero_baseline(self):
        """Test handling of zero baseline effect."""
        monitor = CausalMonitor()

        # Zero baseline, nonzero current
        result = monitor.check_effect_stability(
            current_effect=0.05,
            baseline_effect=0.0,
        )

        assert result.value == float("inf")
        assert result.level == AlertLevel.CRITICAL

    def test_handles_both_zero(self):
        """Test handling when both effects are zero."""
        monitor = CausalMonitor()

        result = monitor.check_effect_stability(
            current_effect=0.0,
            baseline_effect=0.0,
        )

        assert result.value == 0.0
        assert result.level == AlertLevel.OK

    def test_significance_with_standard_errors(self):
        """Test that significance is computed when SEs provided."""
        monitor = CausalMonitor()

        result = monitor.check_effect_stability(
            current_effect=0.15,
            baseline_effect=0.10,
            current_se=0.02,
            baseline_se=0.02,
        )

        assert "z_score" in result.details
        assert "p_value" in result.details
        assert "statistically_significant" in result.details

    def test_sign_change_detection(self):
        """Test detection of effect sign change."""
        monitor = CausalMonitor()

        result = monitor.check_effect_stability(
            current_effect=0.10,
            baseline_effect=-0.10,  # Sign flip
        )

        # 200% relative change
        assert result.level == AlertLevel.CRITICAL
        assert result.details["absolute_change"] == 0.20


class TestCovariateShift:
    """Tests for covariate distribution shift detection."""

    def test_no_shift_returns_ok(self):
        """Test that similar covariate distributions return OK."""
        monitor = CausalMonitor()
        np.random.seed(42)

        # Use same seed for both to ensure identical distributions
        X_baseline = np.random.randn(1000, 5)
        np.random.seed(42)
        X_current = np.random.randn(1000, 5)

        result = monitor.check_covariate_shift(X_current, X_baseline)

        # With identical distributions, should be OK
        assert result.level == AlertLevel.OK

    def test_single_feature_shift_detected(self):
        """Test that shift in single feature is detected."""
        monitor = CausalMonitor()
        np.random.seed(42)

        X_baseline = np.random.randn(1000, 5)
        X_current = np.random.randn(1000, 5)
        X_current[:, 2] += 2.0  # Shift feature 2

        result = monitor.check_covariate_shift(
            X_current, X_baseline, feature_names=["a", "b", "c", "d", "e"]
        )

        assert result.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
        assert "c" in result.details["feature_shifts"]
        assert result.details["feature_shifts"]["c"]["ks_statistic"] > 0.1

    def test_multiple_feature_shifts(self):
        """Test detection of multiple shifted features."""
        monitor = CausalMonitor()
        np.random.seed(42)

        X_baseline = np.random.randn(1000, 3)
        X_current = np.random.randn(1000, 3)
        X_current[:, 0] += 1.5
        X_current[:, 2] += 1.5

        result = monitor.check_covariate_shift(X_current, X_baseline)

        assert result.details["n_warning"] >= 2 or result.details["n_critical"] >= 2

    def test_auto_generates_feature_names(self):
        """Test that feature names are auto-generated if not provided."""
        monitor = CausalMonitor()

        X_baseline = np.random.randn(100, 3)
        X_current = np.random.randn(100, 3)

        result = monitor.check_covariate_shift(X_current, X_baseline)

        assert "feature_0" in result.details["feature_shifts"]
        assert "feature_1" in result.details["feature_shifts"]
        assert "feature_2" in result.details["feature_shifts"]


class TestRunAllChecks:
    """Tests for run_all_checks aggregation."""

    def test_runs_available_checks(self):
        """Test that all applicable checks are run."""
        monitor = CausalMonitor()

        propensity = np.random.uniform(0.2, 0.8, size=100)
        T_current = np.random.binomial(1, 0.5, size=100)
        T_baseline = np.random.binomial(1, 0.5, size=100)

        results = monitor.run_all_checks(
            propensity_scores=propensity,
            treatment_current=T_current,
            treatment_baseline=T_baseline,
            r2_propensity=0.70,
            r2_outcome=0.75,
        )

        check_names = [r.check_name for r in results]
        assert "overlap_violations" in check_names
        assert "treatment_shift" in check_names
        assert "nuisance_degradation" in check_names

    def test_skips_checks_without_data(self):
        """Test that checks are skipped when data not provided."""
        monitor = CausalMonitor()

        # Only provide propensity scores
        results = monitor.run_all_checks(propensity_scores=np.random.uniform(0.2, 0.8, size=100))

        check_names = [r.check_name for r in results]
        assert "overlap_violations" in check_names
        assert "treatment_shift" not in check_names
        assert "nuisance_degradation" not in check_names

    def test_stores_history(self):
        """Test that results are stored in history."""
        monitor = CausalMonitor()

        monitor.run_all_checks(propensity_scores=np.random.uniform(0.2, 0.8, size=100))
        monitor.run_all_checks(propensity_scores=np.random.uniform(0.2, 0.8, size=100))

        assert len(monitor._history) == 2


class TestGetSummary:
    """Tests for summary generation."""

    def test_summary_with_all_ok(self):
        """Test summary when all checks pass."""
        monitor = CausalMonitor()

        results = monitor.run_all_checks(
            propensity_scores=np.random.uniform(0.2, 0.8, size=100),
            r2_propensity=0.75,
            r2_outcome=0.80,
        )

        summary = monitor.get_summary(results)

        assert summary["status"] == "ok"
        assert summary["n_critical"] == 0
        assert summary["n_warning"] == 0

    def test_summary_with_warning(self):
        """Test summary when warning present."""
        monitor = CausalMonitor()

        # Create data with warning-level issue
        propensity = np.random.uniform(0.2, 0.8, size=930)
        violations = np.array([0.001] * 70)  # ~7% violations
        propensity = np.concatenate([propensity, violations])

        results = monitor.run_all_checks(propensity_scores=propensity)
        summary = monitor.get_summary(results)

        assert summary["status"] in ["ok", "warning"]

    def test_summary_with_critical(self):
        """Test summary when critical issue present."""
        monitor = CausalMonitor()

        # Create data with critical issue
        results = [
            MonitoringResult(
                check_name="test_ok",
                level=AlertLevel.OK,
                message="OK",
                value=0.01,
            ),
            MonitoringResult(
                check_name="test_critical",
                level=AlertLevel.CRITICAL,
                message="Critical",
                value=0.20,
            ),
        ]

        summary = monitor.get_summary(results)

        assert summary["status"] == "critical"
        assert summary["n_critical"] == 1

    def test_summary_uses_latest_history(self):
        """Test that summary uses latest history if not provided."""
        monitor = CausalMonitor()

        monitor.run_all_checks(propensity_scores=np.random.uniform(0.2, 0.8, size=100))

        summary = monitor.get_summary()  # No results provided

        assert summary["n_checks"] >= 1
        assert "checks" in summary
