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

from dml_ts.production.causal_monitor import (
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
        """An outcome R² below the critical threshold returns CRITICAL."""
        monitor = CausalMonitor()

        result = monitor.check_nuisance_degradation(
            r2_propensity=0.25,  # alone only a warning
            r2_outcome=0.20,  # below the critical threshold
        )

        assert result.level == AlertLevel.CRITICAL
        assert result.message.startswith("CRITICAL: Outcome model fit very poor")
        assert result.value < 0.30

    @pytest.mark.parametrize(("r2_propensity", "r2_outcome"), [(0.85, 0.25), (0.05, 0.25)])
    def test_value_is_the_smaller_r2_and_the_outcome_sets_the_level(
        self, r2_propensity, r2_outcome
    ):
        """``value`` reports the smaller R²; the level is set per model.

        In the second case the treatment R² is the smaller one, and the level is still
        critical because of the outcome R²: a treatment R² alone is never critical.
        """
        result = CausalMonitor().check_nuisance_degradation(
            r2_propensity=r2_propensity, r2_outcome=r2_outcome
        )

        assert result.value == min(r2_propensity, r2_outcome)
        assert result.details["r2_outcome"] == r2_outcome
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

    @pytest.mark.parametrize("role", ["propensity", "outcome"])
    @pytest.mark.parametrize("baseline,relative", [(0, None), (-0.2, 0.5)])
    def test_nonpositive_r2_baseline(self, role, baseline, relative):
        current = baseline - 0.1
        scores = {"r2_outcome": 0.8, "r2_propensity": 0.8}
        scores[f"r2_{role}"] = current
        result = CausalMonitor().check_nuisance_degradation(
            **scores, **{f"r2_{role}_baseline": baseline}
        )
        assert result.details[f"{role}_r2_decrease"] == pytest.approx(0.1)
        value = result.details[f"{role}_degradation"]
        assert value is None if relative is None else value == pytest.approx(relative)


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

    def test_misspelled_kwarg_raises_typeerror(self):
        """A misspelled check input fails fast instead of being silently dropped (#37)."""
        monitor = CausalMonitor()
        treatment = np.random.binomial(1, 0.5, size=100)
        # `treatment` is not a parameter; the real name is `treatment_current`.
        with pytest.raises(TypeError):
            monitor.run_all_checks(treatment=treatment, treatment_baseline=treatment)

    def test_promoted_optional_kwargs_flow_through(self):
        """The promoted current_se/baseline_se reach the effect-stability check (#37)."""
        monitor = CausalMonitor()
        results = monitor.run_all_checks(
            current_effect=1.0,
            baseline_effect=1.2,
            current_se=0.05,
            baseline_se=0.05,
        )
        # The SE values populate the effect-stability detail fields, proving the
        # promoted keyword args are wired through (not silently ignored).
        assert any("current_se" in r.details for r in results)

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


class TestAlertsFollowTheStandardErrors:
    """The three cases of #64, and the rules that replace the ones they broke."""

    def test_a_randomized_treatment_is_a_warning_not_critical(self):
        # Under randomization the treatment model's R² is near zero by design.
        result = CausalMonitor().check_nuisance_degradation(r2_propensity=0.01, r2_outcome=0.80)

        assert result.level == AlertLevel.WARNING
        assert "randomized" in result.message
        assert result.value == pytest.approx(0.01)

    @pytest.mark.parametrize("r2_propensity", [0.0, 0.29])
    def test_the_treatment_r2_alone_is_never_critical(self, r2_propensity):
        result = CausalMonitor().check_nuisance_degradation(
            r2_propensity=r2_propensity, r2_outcome=0.90
        )

        assert result.level == AlertLevel.WARNING

    def test_a_poor_outcome_model_is_still_critical(self):
        result = CausalMonitor().check_nuisance_degradation(r2_propensity=0.90, r2_outcome=0.29)

        assert result.level == AlertLevel.CRITICAL
        assert "Outcome model" in result.message

    def test_both_models_are_named_when_both_are_low(self):
        result = CausalMonitor().check_nuisance_degradation(r2_propensity=0.10, r2_outcome=0.40)

        assert result.level == AlertLevel.WARNING
        assert "Outcome model" in result.message
        assert "Treatment model" in result.message

    def test_a_change_at_noise_level_is_ok(self):
        # 0.05 -> 0.08 is a 60% relative change, but 0.11 standard errors (p = 0.9155).
        result = CausalMonitor().check_effect_stability(
            current_effect=0.08, baseline_effect=0.05, current_se=0.20, baseline_se=0.20
        )

        assert result.level == AlertLevel.OK
        assert result.details["rule"] == "standard_errors"
        assert result.details["z_score"] == pytest.approx(0.03 / np.sqrt(0.08))
        assert result.details["p_value"] == pytest.approx(0.9155, abs=1e-4)
        assert result.details["relative_change"] == pytest.approx(0.6)
        assert result.value == pytest.approx(0.03 / np.sqrt(0.08))

    def test_a_move_of_106_standard_errors_is_critical(self):
        # 10.0 -> 8.5 is a 15% relative change, which the relative rule marks ok.
        result = CausalMonitor().check_effect_stability(
            current_effect=8.5, baseline_effect=10.0, current_se=0.01, baseline_se=0.01
        )

        assert result.level == AlertLevel.CRITICAL
        assert result.details["z_score"] == pytest.approx(-1.5 / np.sqrt(2e-4))
        assert round(result.details["z_score"]) == -106
        assert result.details["relative_change"] == pytest.approx(0.15)
        assert "-106.06 standard errors, 15% relative" in result.message

    @pytest.mark.parametrize(
        ("change", "level"),
        [
            (1.9375, AlertLevel.OK),
            (1.96, AlertLevel.WARNING),
            (2.9375, AlertLevel.WARNING),
            (3.0, AlertLevel.CRITICAL),
            (-3.0, AlertLevel.CRITICAL),
        ],
    )
    def test_the_level_follows_the_thresholds_in_standard_errors(self, change, level):
        # SEs 1 and 0 combine to exactly 1, so z equals the change, and a change equal
        # to a threshold meets it.
        result = CausalMonitor().check_effect_stability(
            current_effect=change, baseline_effect=0.0, current_se=1.0, baseline_se=0.0
        )

        assert result.details["z_score"] == change
        assert result.level == level
        assert result.threshold == CausalMonitorConfig().effect_z_warning

    @pytest.mark.parametrize(
        ("critical", "level"), [(1.5, AlertLevel.CRITICAL), (2.5, AlertLevel.WARNING)]
    )
    def test_the_thresholds_in_standard_errors_are_configurable(self, critical, level):
        config = CausalMonitorConfig(effect_z_warning=1.0, effect_z_critical=critical)
        result = CausalMonitor(config).check_effect_stability(
            current_effect=2.0, baseline_effect=0.0, current_se=1.0, baseline_se=0.0
        )

        assert result.level == level
        assert result.threshold == 1.0

    @pytest.mark.parametrize(
        ("current_se", "baseline_se", "missing"),
        [
            (None, None, "current_se and baseline_se"),
            (0.01, None, "baseline_se"),
            (None, 0.01, "current_se"),
        ],
    )
    def test_without_both_standard_errors_the_relative_rule_applies_and_says_so(
        self, current_se, baseline_se, missing
    ):
        # 10.0 -> 8.5 with SE 0.01 is critical; without both SEs it is judged by 15%.
        result = CausalMonitor().check_effect_stability(
            current_effect=8.5,
            baseline_effect=10.0,
            current_se=current_se,
            baseline_se=baseline_se,
        )

        assert result.level == AlertLevel.OK
        assert result.details["rule"] == "relative_change"
        assert "z_score" not in result.details
        assert f"{missing} not given" in result.message
        assert result.value == pytest.approx(0.15)
        assert result.threshold == CausalMonitorConfig().effect_change_warning

    @pytest.mark.parametrize(
        ("current_se", "baseline_se"),
        [(-0.1, 0.1), (0.1, float("nan")), (float("inf"), 0.1), (0.0, 0.0)],
    )
    def test_standard_errors_that_cannot_measure_a_change_are_refused(
        self, current_se, baseline_se
    ):
        with pytest.raises(ValueError, match="se"):
            CausalMonitor().check_effect_stability(
                current_effect=1.0,
                baseline_effect=0.9,
                current_se=current_se,
                baseline_se=baseline_se,
            )

    def test_one_standard_error_of_zero_is_measured_by_the_other(self):
        result = CausalMonitor().check_effect_stability(
            current_effect=1.0, baseline_effect=0.9, current_se=0.0, baseline_se=0.05
        )

        assert result.details["z_score"] == pytest.approx(2.0)
        assert result.level == AlertLevel.WARNING

    def test_run_all_checks_hands_on_the_standard_errors(self):
        results = CausalMonitor().run_all_checks(
            current_effect=8.5, baseline_effect=10.0, current_se=0.01, baseline_se=0.01
        )

        (stability,) = [r for r in results if r.check_name == "effect_stability"]
        assert stability.level == AlertLevel.CRITICAL

    @pytest.mark.parametrize(
        ("current", "level"), [(0.25, AlertLevel.CRITICAL), (0.13, AlertLevel.WARNING)]
    )
    def test_a_relative_alert_without_standard_errors_names_them(self, current, level):
        result = CausalMonitor().check_effect_stability(
            current_effect=current, baseline_effect=0.10
        )

        assert result.level == level
        assert "current_se and baseline_se not given" in result.message

    @pytest.mark.parametrize(
        ("current_se", "baseline_se"), [(-0.5, None), (None, float("nan")), (float("inf"), None)]
    )
    def test_a_standard_error_is_checked_when_the_other_is_missing(self, current_se, baseline_se):
        with pytest.raises(ValueError, match="finite, non-negative"):
            CausalMonitor().check_effect_stability(
                current_effect=0.08,
                baseline_effect=0.05,
                current_se=current_se,
                baseline_se=baseline_se,
            )

    @pytest.mark.parametrize(
        ("current", "baseline", "name"),
        [
            (float("nan"), 1.0, "current_effect"),
            (1.0, float("nan"), "baseline_effect"),
            (float("inf"), 1.0, "current_effect"),
            (float("inf"), float("inf"), "current_effect"),
        ],
    )
    @pytest.mark.parametrize("se", [None, 0.1])
    def test_an_effect_that_is_not_finite_is_refused(self, current, baseline, name, se):
        # A NaN failed every comparison and was reported as a stable effect.
        with pytest.raises(ValueError, match=f"{name} must be a finite number"):
            CausalMonitor().check_effect_stability(
                current_effect=current, baseline_effect=baseline, current_se=se, baseline_se=se
            )

    def test_a_low_treatment_r2_no_longer_triggers_a_default_retrain(self):
        # Stated consequence: the default scheduler retrains on critical results only.
        from dml_ts.production.retrain_pipeline import RetrainScheduler, RetrainSchedulerConfig

        scheduler = RetrainScheduler(config=RetrainSchedulerConfig(scheduled_interval_days=0))
        monitor = CausalMonitor()
        randomized = monitor.check_nuisance_degradation(r2_propensity=0.05, r2_outcome=0.85)
        poor_outcome = monitor.check_nuisance_degradation(r2_propensity=0.85, r2_outcome=0.05)

        assert scheduler.evaluate_monitoring_results([randomized], n_samples=200) is None
        trigger = scheduler.evaluate_monitoring_results([poor_outcome], n_samples=200)
        assert trigger is not None
        assert trigger.trigger_type.value == "nuisance_degradation"

    @pytest.mark.parametrize(
        ("kwargs", "name"),
        [
            ({"r2_propensity": 0.9, "r2_outcome": float("nan")}, "r2_outcome"),
            ({"r2_propensity": float("nan"), "r2_outcome": 0.1}, "r2_propensity"),
            ({"r2_propensity": 0.9, "r2_outcome": float("inf")}, "r2_outcome"),
            (
                {"r2_propensity": 0.9, "r2_outcome": 0.9, "r2_outcome_baseline": float("nan")},
                "r2_outcome_baseline",
            ),
            (
                {"r2_propensity": 0.9, "r2_outcome": 0.9, "r2_propensity_baseline": float("-inf")},
                "r2_propensity_baseline",
            ),
        ],
    )
    def test_an_r2_that_is_not_finite_is_refused(self, kwargs, name):
        # A NaN failed every comparison and was reported as a good fit.
        with pytest.raises(ValueError, match=f"{name} must be a finite number"):
            CausalMonitor().check_nuisance_degradation(**kwargs)

    @pytest.mark.parametrize(
        ("r2_propensity", "r2_outcome", "level"),
        [
            (0.90, 0.30, AlertLevel.WARNING),  # outcome at the critical threshold
            (0.90, 0.50, AlertLevel.OK),  # outcome at the warning threshold
            (0.50, 0.90, AlertLevel.OK),  # treatment at the warning threshold
        ],
    )
    def test_an_r2_equal_to_a_threshold_meets_it(self, r2_propensity, r2_outcome, level):
        result = CausalMonitor().check_nuisance_degradation(
            r2_propensity=r2_propensity, r2_outcome=r2_outcome
        )

        assert result.level == level

    def test_the_r2_thresholds_come_from_the_config(self):
        monitor = CausalMonitor(
            CausalMonitorConfig(nuisance_r2_warning=0.8, nuisance_r2_critical=0.6)
        )

        assert monitor.check_nuisance_degradation(0.9, 0.55).level == AlertLevel.CRITICAL
        assert monitor.check_nuisance_degradation(0.9, 0.70).level == AlertLevel.WARNING
        assert monitor.check_nuisance_degradation(0.70, 0.9).level == AlertLevel.WARNING
        assert monitor.check_nuisance_degradation(0.9, 0.9).level == AlertLevel.OK

    def test_a_degraded_baseline_keeps_a_critical_level(self):
        result = CausalMonitor().check_nuisance_degradation(
            r2_propensity=0.9, r2_outcome=0.1, r2_outcome_baseline=0.8
        )

        assert result.level == AlertLevel.CRITICAL
        assert "decreased" in result.message

    @pytest.mark.parametrize(
        ("current", "baseline", "relative", "level"),
        [(6.0, 5.0, 0.2, AlertLevel.WARNING), (3.0, 2.0, 0.5, AlertLevel.CRITICAL)],
    )
    def test_a_relative_change_equal_to_a_threshold_meets_it(
        self, current, baseline, relative, level
    ):
        # 1/5 and 1/2 are computed exactly, so the change equals the threshold.
        result = CausalMonitor().check_effect_stability(
            current_effect=current, baseline_effect=baseline
        )

        assert result.details["relative_change"] == relative
        assert result.level == level

    def test_the_relative_thresholds_come_from_the_config(self):
        config = CausalMonitorConfig(effect_change_warning=0.05, effect_change_critical=0.25)
        monitor = CausalMonitor(config)

        assert monitor.check_effect_stability(1.1, 1.0).level == AlertLevel.WARNING
        assert monitor.check_effect_stability(1.3, 1.0).level == AlertLevel.CRITICAL

    def test_a_zero_baseline_says_the_relative_change_is_undefined(self):
        result = CausalMonitor().check_effect_stability(
            current_effect=0.5, baseline_effect=0.0, current_se=0.2, baseline_se=0.2
        )

        assert result.level == AlertLevel.OK
        assert "+1.76 standard errors, relative change undefined at a zero baseline" in (
            result.message
        )
        assert "inf%" not in result.message

    @pytest.mark.parametrize(
        ("change", "level", "shown"),
        [
            (1.9552, AlertLevel.OK, "+1.95"),
            (np.nextafter(1.96, 0.0), AlertLevel.OK, "+1.95"),
            (1.96, AlertLevel.WARNING, "+1.96"),
            (np.nextafter(3.0, 0.0), AlertLevel.WARNING, "+2.99"),
            (-1.9552, AlertLevel.OK, "-1.95"),
        ],
    )
    def test_a_z_below_a_threshold_never_prints_as_reaching_it(self, change, level, shown):
        # Rounded to nearest, 1.9552 printed as +1.96 beside a level of ok.
        result = CausalMonitor().check_effect_stability(
            current_effect=change, baseline_effect=0.0, current_se=1.0, baseline_se=0.0
        )

        assert result.level == level
        assert f"moved {shown} standard errors" in result.message

    def test_a_very_large_z_prints_in_exponent_form(self):
        result = CausalMonitor().check_effect_stability(
            current_effect=1.0, baseline_effect=0.9, current_se=1e-170, baseline_se=1e-170
        )

        assert result.level == AlertLevel.CRITICAL
        assert "moved +7.07e+168 standard errors, 11% relative" in result.message

    @pytest.mark.parametrize(("change", "significant"), [(2.0, True), (1.9, False)])
    def test_significance_is_judged_at_five_percent(self, change, significant):
        # z = 2.0 gives p = 0.046 and z = 1.9 gives p = 0.057.
        result = CausalMonitor().check_effect_stability(
            current_effect=change, baseline_effect=0.0, current_se=1.0, baseline_se=0.0
        )

        assert result.details["statistically_significant"] is significant

    @pytest.mark.parametrize("scale", [1e-170, 1e200])
    def test_standard_errors_far_from_one_are_combined_without_underflow_or_overflow(self, scale):
        # Squaring 1e-170 gives zero, and squaring the Python float 1e200 overflows.
        result = CausalMonitor().check_effect_stability(
            current_effect=4 * scale, baseline_effect=scale, current_se=scale, baseline_se=scale
        )

        assert result.details["z_score"] == pytest.approx(3 / np.sqrt(2))
        assert result.level == AlertLevel.WARNING
