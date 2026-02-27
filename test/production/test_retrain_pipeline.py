"""
Tests for retraining triggers and scheduling.

Tests cover:
- Scheduled retraining triggers
- Monitoring-based triggers (overlap, treatment shift, nuisance degradation)
- Cooldown period handling
- Trigger evaluation and recording
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.production.causal_monitor import AlertLevel, CausalMonitor, MonitoringResult
from src.production.retrain_pipeline import (
    RetrainScheduler,
    RetrainSchedulerConfig,
    RetrainTrigger,
    TriggerType,
    create_airflow_dag_config,
    create_prefect_flow_config,
)


class TestRetrainTrigger:
    """Tests for RetrainTrigger dataclass."""

    def test_trigger_creation(self):
        """Test basic trigger creation."""
        trigger = RetrainTrigger(
            trigger_type=TriggerType.OVERLAP_VIOLATION,
            triggered_at=datetime.utcnow().isoformat(),
            severity=AlertLevel.CRITICAL,
            reason="15% overlap violations detected",
            metrics={"violation_rate": 0.15},
            model_version="dml-abc123",
        )

        assert trigger.trigger_type == TriggerType.OVERLAP_VIOLATION
        assert trigger.severity == AlertLevel.CRITICAL
        assert trigger.metrics["violation_rate"] == 0.15

    def test_to_dict_serialization(self):
        """Test trigger serialization to dict."""
        trigger = RetrainTrigger(
            trigger_type=TriggerType.SCHEDULED,
            triggered_at="2024-01-15T10:00:00",
            severity=AlertLevel.OK,
            reason="Scheduled retraining",
        )

        d = trigger.to_dict()

        assert d["trigger_type"] == "scheduled"
        assert d["severity"] == "ok"
        assert d["triggered_at"] == "2024-01-15T10:00:00"


class TestRetrainSchedulerConfig:
    """Tests for RetrainSchedulerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetrainSchedulerConfig()

        assert config.scheduled_interval_days == 30
        assert config.trigger_on_critical is True
        assert config.trigger_on_warning is False
        assert config.min_samples_for_trigger == 100
        assert config.cooldown_hours == 24

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetrainSchedulerConfig(
            scheduled_interval_days=7,
            trigger_on_warning=True,
            cooldown_hours=12,
        )

        assert config.scheduled_interval_days == 7
        assert config.trigger_on_warning is True
        assert config.cooldown_hours == 12


class TestRetrainScheduler:
    """Tests for RetrainScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = RetrainScheduler()

        assert scheduler.config is not None
        assert scheduler.monitor is not None
        assert scheduler._last_retrain is None
        assert scheduler._current_model_version is None

    def test_set_current_model(self):
        """Test setting current model version."""
        scheduler = RetrainScheduler()
        scheduler.set_current_model("dml-version-123")

        assert scheduler._current_model_version == "dml-version-123"

    def test_record_retrain(self):
        """Test recording a retraining event."""
        scheduler = RetrainScheduler()

        trigger = RetrainTrigger(
            trigger_type=TriggerType.MANUAL,
            triggered_at=datetime.utcnow().isoformat(),
            severity=AlertLevel.OK,
            reason="Manual trigger",
        )

        scheduler.record_retrain(trigger, "dml-new-version")

        assert scheduler._last_retrain is not None
        assert scheduler._current_model_version == "dml-new-version"
        assert len(scheduler._trigger_history) == 1

    def test_is_in_cooldown_false_initially(self):
        """Test cooldown is False when no previous retrain."""
        scheduler = RetrainScheduler()

        assert scheduler.is_in_cooldown() is False

    def test_is_in_cooldown_true_after_retrain(self):
        """Test cooldown is True immediately after retrain."""
        scheduler = RetrainScheduler()
        trigger = RetrainTrigger(
            trigger_type=TriggerType.MANUAL,
            triggered_at=datetime.utcnow().isoformat(),
            severity=AlertLevel.OK,
            reason="Test",
        )
        scheduler.record_retrain(trigger, "v1")

        assert scheduler.is_in_cooldown() is True

    def test_check_scheduled_retrain_no_previous(self):
        """Test scheduled check when no previous retrain."""
        scheduler = RetrainScheduler()

        trigger = scheduler.check_scheduled_retrain()

        assert trigger is not None
        assert trigger.trigger_type == TriggerType.SCHEDULED
        assert "Initial" in trigger.reason

    def test_check_scheduled_retrain_recent(self):
        """Test scheduled check when retrain was recent."""
        scheduler = RetrainScheduler()
        scheduler._last_retrain = datetime.utcnow()

        trigger = scheduler.check_scheduled_retrain()

        # Should not trigger if within interval
        assert trigger is None

    def test_check_scheduled_retrain_overdue(self):
        """Test scheduled check when retrain is overdue."""
        config = RetrainSchedulerConfig(scheduled_interval_days=7)
        scheduler = RetrainScheduler(config)

        # Set last retrain to 10 days ago
        scheduler._last_retrain = datetime.utcnow() - timedelta(days=10)

        trigger = scheduler.check_scheduled_retrain()

        assert trigger is not None
        assert trigger.trigger_type == TriggerType.SCHEDULED
        assert "10 days since" in trigger.reason

    def test_check_scheduled_disabled(self):
        """Test scheduled check when interval is 0 (disabled)."""
        config = RetrainSchedulerConfig(scheduled_interval_days=0)
        scheduler = RetrainScheduler(config)

        trigger = scheduler.check_scheduled_retrain()

        assert trigger is None


class TestEvaluateMonitoringResults:
    """Tests for monitoring result evaluation."""

    def test_no_trigger_on_ok_results(self):
        """Test no trigger when all results OK."""
        scheduler = RetrainScheduler()
        scheduler._last_retrain = datetime.utcnow()  # Avoid scheduled trigger

        results = [
            MonitoringResult(
                check_name="overlap_violations",
                level=AlertLevel.OK,
                message="OK",
                value=0.02,
            ),
            MonitoringResult(
                check_name="treatment_shift",
                level=AlertLevel.OK,
                message="OK",
                value=0.01,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=200)

        assert trigger is None

    def test_trigger_on_critical_overlap(self):
        """Test trigger on CRITICAL overlap violation."""
        scheduler = RetrainScheduler()

        results = [
            MonitoringResult(
                check_name="overlap_violations",
                level=AlertLevel.CRITICAL,
                message="Critical overlap violations",
                value=0.15,
                threshold=0.10,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=200)

        assert trigger is not None
        assert trigger.trigger_type == TriggerType.OVERLAP_VIOLATION
        assert trigger.severity == AlertLevel.CRITICAL

    def test_trigger_on_critical_treatment_shift(self):
        """Test trigger on CRITICAL treatment shift."""
        scheduler = RetrainScheduler()

        results = [
            MonitoringResult(
                check_name="treatment_shift",
                level=AlertLevel.CRITICAL,
                message="Treatment distribution shift",
                value=0.15,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=200)

        assert trigger is not None
        assert trigger.trigger_type == TriggerType.TREATMENT_SHIFT

    def test_no_trigger_on_warning_by_default(self):
        """Test no trigger on WARNING when trigger_on_warning is False."""
        scheduler = RetrainScheduler()
        scheduler._last_retrain = datetime.utcnow()

        results = [
            MonitoringResult(
                check_name="overlap_violations",
                level=AlertLevel.WARNING,
                message="Warning",
                value=0.08,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=200)

        assert trigger is None

    def test_trigger_on_warning_when_enabled(self):
        """Test trigger on WARNING when trigger_on_warning is True."""
        config = RetrainSchedulerConfig(trigger_on_warning=True)
        scheduler = RetrainScheduler(config)

        results = [
            MonitoringResult(
                check_name="nuisance_degradation",
                level=AlertLevel.WARNING,
                message="Degraded",
                value=0.45,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=200)

        assert trigger is not None
        assert trigger.trigger_type == TriggerType.NUISANCE_DEGRADATION

    def test_no_trigger_during_cooldown(self):
        """Test no trigger during cooldown period."""
        scheduler = RetrainScheduler()
        scheduler._last_retrain = datetime.utcnow()

        results = [
            MonitoringResult(
                check_name="overlap_violations",
                level=AlertLevel.CRITICAL,
                message="Critical",
                value=0.20,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=200)

        # In cooldown - should not trigger
        assert trigger is None

    def test_no_trigger_insufficient_samples(self):
        """Test no trigger when samples below minimum."""
        config = RetrainSchedulerConfig(min_samples_for_trigger=500)
        scheduler = RetrainScheduler(config)

        results = [
            MonitoringResult(
                check_name="overlap_violations",
                level=AlertLevel.CRITICAL,
                message="Critical",
                value=0.20,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=100)

        assert trigger is None

    def test_disabled_trigger_types_ignored(self):
        """Test that disabled trigger types are ignored."""
        config = RetrainSchedulerConfig(enabled_triggers=[TriggerType.SCHEDULED])  # Only scheduled
        scheduler = RetrainScheduler(config)

        results = [
            MonitoringResult(
                check_name="overlap_violations",
                level=AlertLevel.CRITICAL,
                message="Critical",
                value=0.20,
            ),
        ]

        trigger = scheduler.evaluate_monitoring_results(results, n_samples=200)

        # Overlap is not enabled, should fall through to scheduled
        assert trigger is None or trigger.trigger_type == TriggerType.SCHEDULED


class TestEvaluateAndMonitor:
    """Tests for combined evaluate_and_monitor method."""

    def test_runs_monitoring_and_evaluation(self):
        """Test that method runs both monitoring and evaluation."""
        scheduler = RetrainScheduler()

        propensity = np.random.uniform(0.2, 0.8, size=200)

        results, trigger = scheduler.evaluate_and_monitor(
            propensity_scores=propensity,
            r2_propensity=0.75,
            r2_outcome=0.80,
        )

        assert len(results) > 0
        assert all(isinstance(r, MonitoringResult) for r in results)

    def test_returns_trigger_when_critical(self):
        """Test that trigger is returned for critical issues."""
        scheduler = RetrainScheduler()

        # Create critical overlap violations
        propensity = np.random.uniform(0.2, 0.8, size=800)
        violations = np.array([0.001] * 100 + [0.999] * 100)
        propensity = np.concatenate([propensity, violations])  # 20% violations

        results, trigger = scheduler.evaluate_and_monitor(propensity_scores=propensity)

        # With 20% violations, should trigger
        assert trigger is not None
        assert trigger.trigger_type == TriggerType.OVERLAP_VIOLATION


class TestManualTrigger:
    """Tests for manual trigger creation."""

    def test_create_manual_trigger(self):
        """Test creating a manual trigger."""
        scheduler = RetrainScheduler()
        scheduler.set_current_model("dml-v1")

        trigger = scheduler.create_manual_trigger("Requested by data science team")

        assert trigger.trigger_type == TriggerType.MANUAL
        assert trigger.severity == AlertLevel.OK
        assert "Manual trigger" in trigger.reason
        assert trigger.model_version == "dml-v1"


class TestSchedulerStatus:
    """Tests for scheduler status reporting."""

    def test_get_status(self):
        """Test getting scheduler status."""
        scheduler = RetrainScheduler()
        scheduler.set_current_model("dml-v1")

        status = scheduler.get_status()

        assert status["current_model_version"] == "dml-v1"
        assert status["last_retrain"] is None
        assert status["in_cooldown"] is False
        assert "enabled_triggers" in status

    def test_get_trigger_history(self):
        """Test getting trigger history."""
        scheduler = RetrainScheduler()

        trigger1 = scheduler.create_manual_trigger("Test 1")
        scheduler.record_retrain(trigger1, "v1")

        trigger2 = scheduler.create_manual_trigger("Test 2")
        scheduler.record_retrain(trigger2, "v2")

        history = scheduler.get_trigger_history()

        assert len(history) == 2
        assert all("trigger_type" in t for t in history)


class TestDAGConfigs:
    """Tests for workflow configuration generation."""

    def test_create_airflow_dag_config(self):
        """Test Airflow DAG configuration generation."""
        scheduler = RetrainScheduler()
        config = create_airflow_dag_config(scheduler, dag_id="test_dag")

        assert config["dag_id"] == "test_dag"
        assert "tasks" in config
        assert len(config["tasks"]) == 5
        task_ids = [t["task_id"] for t in config["tasks"]]
        assert "check_monitoring" in task_ids
        assert "retrain_model" in task_ids

    def test_create_prefect_flow_config(self):
        """Test Prefect flow configuration generation."""
        scheduler = RetrainScheduler()
        config = create_prefect_flow_config(scheduler, flow_name="test_flow")

        assert config["flow_name"] == "test_flow"
        assert "tasks" in config
        assert "schedule" in config
        task_names = [t["name"] for t in config["tasks"]]
        assert "run_monitoring" in task_names
        assert "retrain_conditionally" in task_names
