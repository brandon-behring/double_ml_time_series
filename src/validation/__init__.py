"""Validation infrastructure for Double ML methods."""

from .dgp_generator import DGPGenerator, DGPResult
from .validation_result import ValidationResult
from .storage import ResultStorage
from . import plotting
from .parallel import (
    parallel_map,
    parallel_monte_carlo,
    parallelize,
    ParallelExecutor,
    chunk_workload,
    get_optimal_n_jobs,
)
from .bias_validation import BiasValidation

# Phase 2A stubs (not yet implemented)
from .stationarity import (
    StationarityDiagnostic,
    StationarityResult,
    ComprehensiveStationarityResult,
)

__all__ = [
    "DGPGenerator",
    "DGPResult",
    "ValidationResult",
    "ResultStorage",
    "plotting",
    "parallel_map",
    "parallel_monte_carlo",
    "parallelize",
    "ParallelExecutor",
    "chunk_workload",
    "get_optimal_n_jobs",
    "BiasValidation",
    # Phase 2A stubs
    "StationarityDiagnostic",
    "StationarityResult",
    "ComprehensiveStationarityResult",
]
