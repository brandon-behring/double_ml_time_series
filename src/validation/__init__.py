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

# Phase 2A (fully implemented)
from .stationarity import (
    StationarityDiagnostic,
    StationarityResult,
    ComprehensiveStationarityResult,
)
from .insurance_dgp import (
    InsuranceDGPParams,
    InsuranceDGPResult,
    create_insurance_dgp,
    validate_dgp_recovery,
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
    # Phase 2A
    "StationarityDiagnostic",
    "StationarityResult",
    "ComprehensiveStationarityResult",
    # Insurance DGP
    "InsuranceDGPParams",
    "InsuranceDGPResult",
    "create_insurance_dgp",
    "validate_dgp_recovery",
]
