"""Double Machine Learning companion estimators.

This package implements:
1. Frisch-Waugh-Lovell (FWL) - Linear residualization baseline
2. Robinson Estimator - FWL with ML nuisances
3. Double Machine Learning - Cross-fitted Robinson for debiased inference
4. TemporalPLRDML - Scalar temporal PLR DML with HAC inference

The progression FWL -> Robinson -> DML is pedagogically essential:
- FWL demonstrates WHY residualization works
- Robinson shows WHEN linear fails (nonlinear confounding)
- DML solves the regularization bias problem via cross-fitting
"""

from ._dynamic_econml import econml_available, fit_econml_reference
from .cv_factory import create_time_series_cv
from .double_ml import DMLResult, double_ml
from .dynamic_g_estimation import DynamicGEstimationDML, DynamicGEstimationResult
from .fwl import (
    fwl_estimate,
    fwl_residualize,
    fwl_vs_ols_comparison,
)
from .robinson import robinson_estimator
from .temporal_plr_dml import (
    PanelDML,
    RollingWindowDML,
    TemporalPLRDML,
    TemporalPLRDMLResult,
)

__all__ = [
    "create_time_series_cv",
    "fwl_estimate",
    "fwl_residualize",
    "fwl_vs_ols_comparison",
    "robinson_estimator",
    "double_ml",
    "DMLResult",
    "TemporalPLRDML",
    "TemporalPLRDMLResult",
    "RollingWindowDML",
    "PanelDML",
    "DynamicGEstimationDML",
    "DynamicGEstimationResult",
    "fit_econml_reference",
    "econml_available",
]
