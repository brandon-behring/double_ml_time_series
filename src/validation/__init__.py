"""Validation infrastructure for Double ML methods."""

from .dgp_generator import DGPGenerator, DGPResult
from .dgp_generator_ts import (
    TimeSeriesDGPGenerator,
    TimeSeriesDGPResult,
    BreakDGPGenerator,
    BreakDGPResult,
    create_ar_dgp,
    create_panel_dgp,
    create_break_dgp,
)
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
    # i.i.d. DGP
    "DGPGenerator",
    "DGPResult",
    # Time Series DGP
    "TimeSeriesDGPGenerator",
    "TimeSeriesDGPResult",
    "BreakDGPGenerator",
    "BreakDGPResult",
    "create_ar_dgp",
    "create_panel_dgp",
    "create_break_dgp",
    # Other
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
