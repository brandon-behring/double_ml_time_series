"""Validation infrastructure for Double ML methods."""

from . import plotting
from .bias_validation import BiasValidation
from .dgp_generator import DGPGenerator, DGPResult
from .dgp_generator_ts import (
    BreakDGPGenerator,
    BreakDGPResult,
    TimeSeriesDGPGenerator,
    TimeSeriesDGPResult,
    create_ar_dgp,
    create_break_dgp,
    create_panel_dgp,
)
from .dynamic_dgp import DynamicTreatmentDGP, DynamicTreatmentDGPResult
from .insurance_dgp import (
    InsuranceDGPParams,
    InsuranceDGPResult,
    create_insurance_dgp,
    validate_dgp_recovery,
)
from .parallel import (
    ParallelExecutor,
    chunk_workload,
    get_optimal_n_jobs,
    parallel_map,
    parallel_monte_carlo,
    parallelize,
)
from .storage import ResultStorage
from .validation_result import ValidationResult

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
    # Dynamic-treatment DGP (known per-period blips, for g-estimation)
    "DynamicTreatmentDGP",
    "DynamicTreatmentDGPResult",
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
    # Insurance DGP
    "InsuranceDGPParams",
    "InsuranceDGPResult",
    "create_insurance_dgp",
    "validate_dgp_recovery",
]
