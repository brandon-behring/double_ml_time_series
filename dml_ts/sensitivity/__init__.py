"""Sensitivity analysis tools for causal inference.

This module provides methods to assess the robustness of causal effect
estimates to violations of the unconfoundedness assumption.

Available Methods
-----------------
- Rosenbaum Bounds: Classic bounds for hidden bias assessment
- Parametric Sensitivity: Chernozhukov-style sensitivity analysis
"""

from .rosenbaum import RosenbaumBounds, SensitivityResult, compute_sensitivity_for_dml

__all__ = ["RosenbaumBounds", "SensitivityResult", "compute_sensitivity_for_dml"]
