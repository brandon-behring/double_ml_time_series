"""
Production module for deploying Double Machine Learning pipelines.

This module provides tools for:
- Model serialization and versioning (DMLModelVersion)
- Causal-specific monitoring (CausalMonitor)
- Intelligent retraining triggers (RetrainScheduler)
- End-to-end DML pipelines (InsuranceDMLPipeline)

Key differences from standard ML production:
1. Monitor treatment distribution shifts (not just feature drift)
2. Track overlap violations (positivity assumption)
3. Validate nuisance model quality (propensity/outcome R²)
4. Detect effect stability changes over time
"""

from src.production.model_registry import DMLModelVersion, DMLModelRegistry
from src.production.causal_monitor import CausalMonitor, MonitoringResult
from src.production.retrain_pipeline import RetrainScheduler, RetrainTrigger
from src.production.dml_pipeline import InsuranceDMLPipeline, PipelineConfig

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
