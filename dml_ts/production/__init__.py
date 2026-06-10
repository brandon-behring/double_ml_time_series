"""
Research/demo utilities for Double Machine Learning pipeline organization.

This module provides tools for:
- Model serialization and versioning (DMLModelVersion)
- Causal-specific monitoring (CausalMonitor)
- Intelligent retraining triggers (RetrainScheduler)
- End-to-end DML pipelines (InsuranceDMLPipeline)

Conceptual differences from standard ML deployment:
1. Monitor treatment distribution shifts (not just feature drift)
2. Track overlap violations (positivity assumption)
3. Validate nuisance model quality (propensity/outcome R²)
4. Detect effect stability changes over time

These are book-companion utilities, not deployment-ready infrastructure.
"""

from dml_ts.production.causal_monitor import CausalMonitor, MonitoringResult
from dml_ts.production.dml_pipeline import InsuranceDMLPipeline, PipelineConfig
from dml_ts.production.model_registry import DMLModelRegistry, DMLModelVersion
from dml_ts.production.retrain_pipeline import RetrainScheduler, RetrainTrigger

__all__ = [
    "DMLModelVersion",
    "DMLModelRegistry",
    "CausalMonitor",
    "MonitoringResult",
    "RetrainScheduler",
    "RetrainTrigger",
    "InsuranceDMLPipeline",
    "PipelineConfig",
]
