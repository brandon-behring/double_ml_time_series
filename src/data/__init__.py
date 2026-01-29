"""Data loading utilities for DML applications.

This module provides standardized data loaders for benchmark datasets
used in causal inference validation and demonstration.

Available Datasets
------------------
- OJ Dataset: Dominick's Orange Juice store-level panel data
  - Price elasticity estimation
  - Standard benchmark for DML applications
"""

from .oj_loader import OJDataset, OJDataLoader

__all__ = ["OJDataset", "OJDataLoader"]
