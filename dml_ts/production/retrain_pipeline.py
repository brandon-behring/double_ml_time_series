"""
Retraining triggers and scheduling for DML pipelines.

This module provides causal-specific retraining logic that differs from
standard ML in critical ways:

Standard ML Retraining Triggers:
- Prediction accuracy degradation
- Feature drift
- Scheduled intervals

Causal-Specific Retraining Triggers:
- Overlap violation threshold exceeded (positivity breakdown)
- Treatment distribution shift (assignment mechanism change)
- Nuisance model R² degradation (NOT causal accuracy - unknowable!)
- Effect stability change beyond acceptable bounds

Key insight: We CANNOT retrain based on "treatment effect accuracy" because
counterfactual outcomes are never observed. We can only monitor proxies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from dml_ts.production.causal_monitor import AlertLevel, CausalMonitor, MonitoringResult


class TriggerType(Enum):
    """Types of retraining triggers."""

    SCHEDULED = "scheduled"
    OVERLAP_VIOLATION = "overlap_violation"
    TREATMENT_SHIFT = "treatment_shift"
    NUISANCE_DEGRADATION = "nuisance_degradation"
    EFFECT_INSTABILITY = "effect_instability"
    COVARIATE_SHIFT = "covariate_shift"
    MANUAL = "manual"


@dataclass
class RetrainTrigger:
    """
    A single retraining trigger event.

    Attributes:
        trigger_type: Type of trigger
        triggered_at: When the trigger fired
        severity: Alert level that triggered retraining
        reason: Human-readable explanation
        metrics: Relevant metrics at trigger time
        model_version: Version of model that triggered
    """

    trigger_type: TriggerType
    triggered_at: str
    severity: AlertLevel
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trigger_type": self.trigger_type.value,
            "triggered_at": self.triggered_at,
            "severity": self.severity.value,
            "reason": self.reason,
            "metrics": self.metrics,
            "model_version": self.model_version,
        }


@dataclass
class RetrainSchedulerConfig:
    """
    Configuration for retraining scheduler.

    Attributes:
        scheduled_interval_days: Days between scheduled retrains (0 = disabled)
        trigger_on_critical: Trigger retrain on CRITICAL alerts
        trigger_on_warning: Trigger retrain on WARNING alerts (use cautiously)
        min_samples_for_trigger: Minimum samples before triggering
        cooldown_hours: Minimum hours between retrains
        enabled_triggers: Which trigger types are active
    """

    scheduled_interval_days: int = 30
    trigger_on_critical: bool = True
    trigger_on_warning: bool = False
    min_samples_for_trigger: int = 100
    cooldown_hours: int = 24
    enabled_triggers: list[TriggerType] = field(
        default_factory=lambda: [
            TriggerType.SCHEDULED,
            TriggerType.OVERLAP_VIOLATION,
            TriggerType.TREATMENT_SHIFT,
            TriggerType.NUISANCE_DEGRADATION,
        ]
    )


class RetrainScheduler:
    """
    Intelligent retraining scheduler for DML pipelines.

    Combines scheduled and triggered retraining with causal-specific
    monitoring integration.

    Example:
        >>> scheduler = RetrainScheduler()
        >>> monitor = CausalMonitor()
        >>>
        >>> # Check if retraining needed
        >>> results = monitor.run_all_checks(...)
        >>> trigger = scheduler.evaluate_monitoring_results(results)
        >>> if trigger is not None:
        ...     print(f"Retrain triggered: {trigger.reason}")
        ...     # Execute retraining pipeline
        ...     scheduler.record_retrain(trigger, new_version_id)
    """

    def __init__(
        self,
        config: RetrainSchedulerConfig | None = None,
        monitor: CausalMonitor | None = None,
    ):
        """
        Initialize retrain scheduler.

        Args:
            config: Scheduler configuration (uses defaults if None)
            monitor: CausalMonitor instance (creates new if None)
        """
        self.config = config or RetrainSchedulerConfig()
        self.monitor = monitor or CausalMonitor()

        self._last_retrain: datetime | None = None
        self._last_scheduled_check: datetime | None = None
        self._trigger_history: list[RetrainTrigger] = []
        self._current_model_version: str | None = None

    def set_current_model(self, version_id: str) -> None:
        """Set the current production model version."""
        self._current_model_version = version_id

    def record_retrain(self, trigger: RetrainTrigger, new_version_id: str) -> None:
        """
        Record that retraining occurred.

        Args:
            trigger: The trigger that caused retraining
            new_version_id: Version ID of newly trained model
        """
        self._last_retrain = datetime.now(UTC)
        self._trigger_history.append(trigger)
        self._current_model_version = new_version_id

    def is_in_cooldown(self) -> bool:
        """Check if scheduler is in cooldown period."""
        if self._last_retrain is None:
            return False
        elapsed = datetime.now(UTC) - self._last_retrain
        return elapsed < timedelta(hours=self.config.cooldown_hours)

    def check_scheduled_retrain(self) -> RetrainTrigger | None:
        """
        Check if scheduled retraining is due.

        Returns:
            RetrainTrigger if scheduled retrain due, None otherwise
        """
        if self.config.scheduled_interval_days <= 0:
            return None

        if TriggerType.SCHEDULED not in self.config.enabled_triggers:
            return None

        # Check time since last retrain
        if self._last_retrain is None:
            # No previous retrain - schedule one
            return RetrainTrigger(
                trigger_type=TriggerType.SCHEDULED,
                triggered_at=datetime.now(UTC).isoformat(),
                severity=AlertLevel.OK,
                reason="Initial scheduled retraining (no previous retrain recorded)",
                model_version=self._current_model_version,
            )

        elapsed_days = (datetime.now(UTC) - self._last_retrain).days
        if elapsed_days >= self.config.scheduled_interval_days:
            return RetrainTrigger(
                trigger_type=TriggerType.SCHEDULED,
                triggered_at=datetime.now(UTC).isoformat(),
                severity=AlertLevel.OK,
                reason=(
                    f"Scheduled retraining: {elapsed_days} days since last retrain "
                    f"(threshold: {self.config.scheduled_interval_days} days)"
                ),
                metrics={"days_since_retrain": elapsed_days},
                model_version=self._current_model_version,
            )

        return None

    def evaluate_monitoring_results(
        self,
        results: list[MonitoringResult],
        n_samples: int | None = None,
    ) -> RetrainTrigger | None:
        """
        Evaluate monitoring results and determine if retraining needed.

        Args:
            results: List of MonitoringResult from CausalMonitor
            n_samples: Number of samples in evaluation (for min threshold)

        Returns:
            RetrainTrigger if retraining needed, None otherwise
        """
        # Check cooldown
        if self.is_in_cooldown():
            return None

        # Check minimum samples
        if n_samples is not None and n_samples < self.config.min_samples_for_trigger:
            return None

        # Map check names to trigger types
        check_to_trigger = {
            "overlap_violations": TriggerType.OVERLAP_VIOLATION,
            "treatment_shift": TriggerType.TREATMENT_SHIFT,
            "nuisance_degradation": TriggerType.NUISANCE_DEGRADATION,
            "effect_stability": TriggerType.EFFECT_INSTABILITY,
            "covariate_shift": TriggerType.COVARIATE_SHIFT,
        }

        # Find triggering results
        for result in results:
            trigger_type = check_to_trigger.get(result.check_name)
            if trigger_type is None:
                continue

            if trigger_type not in self.config.enabled_triggers:
                continue

            should_trigger = False
            if result.level == AlertLevel.CRITICAL and self.config.trigger_on_critical:
                should_trigger = True
            elif result.level == AlertLevel.WARNING and self.config.trigger_on_warning:
                should_trigger = True

            if should_trigger:
                return RetrainTrigger(
                    trigger_type=trigger_type,
                    triggered_at=datetime.now(UTC).isoformat(),
                    severity=result.level,
                    reason=result.message,
                    metrics={"value": result.value, "threshold": result.threshold or 0},
                    model_version=self._current_model_version,
                )

        # Check scheduled retraining
        scheduled_trigger = self.check_scheduled_retrain()
        if scheduled_trigger is not None:
            return scheduled_trigger

        return None

    def evaluate_and_monitor(
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
        **kwargs: Any,
    ) -> tuple[list[MonitoringResult], RetrainTrigger | None]:
        """
        Run monitoring and evaluate for retraining in one call.

        Convenience method combining CausalMonitor.run_all_checks with
        evaluate_monitoring_results.

        Args:
            propensity_scores: Estimated propensity scores
            treatment_current: Current treatment values
            treatment_baseline: Baseline treatment values
            r2_propensity: Propensity model R²
            r2_outcome: Outcome model R²
            current_effect: Current ATE estimate
            baseline_effect: Baseline ATE estimate
            X_current: Current covariates
            X_baseline: Baseline covariates
            **kwargs: Additional args for monitor

        Returns:
            Tuple of (monitoring_results, retrain_trigger)
        """
        n_samples = None
        if propensity_scores is not None:
            n_samples = len(propensity_scores)
        elif treatment_current is not None:
            n_samples = len(treatment_current)
        elif X_current is not None:
            n_samples = len(X_current)

        # Run monitoring
        results = self.monitor.run_all_checks(
            propensity_scores=propensity_scores,
            treatment_current=treatment_current,
            treatment_baseline=treatment_baseline,
            r2_propensity=r2_propensity,
            r2_outcome=r2_outcome,
            current_effect=current_effect,
            baseline_effect=baseline_effect,
            X_current=X_current,
            X_baseline=X_baseline,
            **kwargs,
        )

        # Evaluate for retraining
        trigger = self.evaluate_monitoring_results(results, n_samples)

        return results, trigger

    def create_manual_trigger(self, reason: str) -> RetrainTrigger:
        """
        Create a manual retraining trigger.

        Args:
            reason: Explanation for manual trigger

        Returns:
            RetrainTrigger for manual retraining
        """
        return RetrainTrigger(
            trigger_type=TriggerType.MANUAL,
            triggered_at=datetime.now(UTC).isoformat(),
            severity=AlertLevel.OK,
            reason=f"Manual trigger: {reason}",
            model_version=self._current_model_version,
        )

    def get_trigger_history(self) -> list[dict[str, Any]]:
        """Return history of retraining triggers as dicts."""
        return [t.to_dict() for t in self._trigger_history]

    def get_status(self) -> dict[str, Any]:
        """
        Get current scheduler status.

        Returns:
            Dict with scheduler state information
        """
        return {
            "current_model_version": self._current_model_version,
            "last_retrain": (self._last_retrain.isoformat() if self._last_retrain else None),
            "in_cooldown": self.is_in_cooldown(),
            "cooldown_remaining_hours": (
                max(
                    0,
                    self.config.cooldown_hours
                    - (datetime.now(UTC) - self._last_retrain).total_seconds() / 3600,
                )
                if self._last_retrain
                else 0
            ),
            "n_triggers_total": len(self._trigger_history),
            "scheduled_interval_days": self.config.scheduled_interval_days,
            "enabled_triggers": [t.value for t in self.config.enabled_triggers],
        }

    def __repr__(self) -> str:
        return (
            f"RetrainScheduler("
            f"model={self._current_model_version}, "
            f"last_retrain={self._last_retrain}, "
            f"n_triggers={len(self._trigger_history)})"
        )


def create_airflow_dag_config(
    scheduler: RetrainScheduler,
    dag_id: str = "dml_retrain_pipeline",
    schedule: str = "@daily",
) -> dict[str, Any]:
    """
    Create Airflow DAG configuration for DML retraining.

    This is a configuration template - actual DAG implementation
    would be in your Airflow codebase.

    Args:
        scheduler: RetrainScheduler instance
        dag_id: Airflow DAG identifier
        schedule: Airflow schedule interval

    Returns:
        Dict with DAG configuration
    """
    return {
        "dag_id": dag_id,
        "schedule_interval": schedule,
        "default_args": {
            "owner": "causal-ml",
            "depends_on_past": False,
            "retries": 1,
            "retry_delay_minutes": 5,
        },
        "tasks": [
            {
                "task_id": "check_monitoring",
                "description": "Run causal monitoring checks",
                "python_callable": "run_causal_monitoring",
            },
            {
                "task_id": "evaluate_retrain",
                "description": "Evaluate if retraining needed",
                "python_callable": "evaluate_retrain_trigger",
                "depends_on": ["check_monitoring"],
            },
            {
                "task_id": "retrain_model",
                "description": "Retrain DML model if triggered",
                "python_callable": "retrain_dml_pipeline",
                "depends_on": ["evaluate_retrain"],
                "trigger_rule": "none_failed",
            },
            {
                "task_id": "validate_model",
                "description": "Validate retrained model",
                "python_callable": "validate_new_model",
                "depends_on": ["retrain_model"],
            },
            {
                "task_id": "promote_staging",
                "description": "Promote to staging if validation passes",
                "python_callable": "promote_to_staging",
                "depends_on": ["validate_model"],
            },
        ],
        "scheduler_config": scheduler.config.__dict__,
    }


def create_prefect_flow_config(
    scheduler: RetrainScheduler,
    flow_name: str = "dml-retrain",
) -> dict[str, Any]:
    """
    Create Prefect flow configuration for DML retraining.

    This is a configuration template - actual flow implementation
    would be in your Prefect codebase.

    Args:
        scheduler: RetrainScheduler instance
        flow_name: Prefect flow name

    Returns:
        Dict with flow configuration
    """
    return {
        "flow_name": flow_name,
        "schedule": {
            "cron": "0 6 * * *",  # Daily at 6 AM
            "timezone": "UTC",
        },
        "tasks": [
            {
                "name": "load_data",
                "description": "Load inference data for monitoring",
                "retries": 2,
            },
            {
                "name": "run_monitoring",
                "description": "Execute causal monitoring checks",
                "depends_on": ["load_data"],
            },
            {
                "name": "evaluate_trigger",
                "description": "Determine if retraining needed",
                "depends_on": ["run_monitoring"],
            },
            {
                "name": "retrain_conditionally",
                "description": "Retrain model if trigger fired",
                "depends_on": ["evaluate_trigger"],
                "conditional": True,
            },
            {
                "name": "validate_and_promote",
                "description": "Validate and promote to staging",
                "depends_on": ["retrain_conditionally"],
            },
        ],
        "scheduler_config": scheduler.config.__dict__,
    }
