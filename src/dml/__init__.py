"""Double Machine Learning for Time Series Causal Inference.

This package implements:
1. Frisch-Waugh-Lovell (FWL) - Linear residualization baseline
2. Robinson Estimator - FWL with ML nuisances
3. Double Machine Learning - Cross-fitted Robinson for debiased inference
4. Dynamic DML - Sequential g-estimation for time series

The progression FWL → Robinson → DML is pedagogically essential:
- FWL demonstrates WHY residualization works
- Robinson shows WHEN linear fails (nonlinear confounding)
- DML solves the regularization bias problem via cross-fitting
"""

from .fwl import (
    fwl_estimate,
    fwl_residualize,
    fwl_vs_ols_comparison,
)
from .robinson import robinson_estimator
from .double_ml import double_ml, DMLResult

__all__ = [
    "fwl_estimate",
    "fwl_residualize",
    "fwl_vs_ols_comparison",
    "robinson_estimator",
    "double_ml",
    "DMLResult",
]
