"""Data loading utilities for DML applications.

This module provides standardized data loaders for benchmark datasets
used in causal inference validation and demonstration.

Available Datasets
------------------
- OJ Dataset: Dominick's Orange Juice store-level panel data
  - Price elasticity estimation
  - Standard benchmark for DML applications

- FRED Macro Data: Federal Reserve Economic Data
  - GDP, CPI, unemployment, interest rates
  - Macroeconomic controls for time series DML
"""

from .oj_loader import OJDataset, OJDataLoader
from .fred_loader import (
    FREDLoader,
    FREDSeries,
    MacroControlsResult,
    create_synthetic_fred_data,
    STANDARD_MACRO_SERIES,
    MACRO_CONTROL_SETS,
)

__all__ = [
    # OJ Dataset
    "OJDataset",
    "OJDataLoader",
    # FRED Data
    "FREDLoader",
    "FREDSeries",
    "MacroControlsResult",
    "create_synthetic_fred_data",
    "STANDARD_MACRO_SERIES",
    "MACRO_CONTROL_SETS",
]
