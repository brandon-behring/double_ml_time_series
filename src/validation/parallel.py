"""
Parallel execution framework for Monte Carlo simulations.

Optimized for 64-core AMD Threadripper 3990X with intelligent workload distribution.
Provides decorators and utilities for parallelizing validation methods.
"""

from typing import Callable, List, Any, Optional, TypeVar, ParamSpec
from functools import wraps
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# Type variables for generic function signatures
P = ParamSpec("P")
T = TypeVar("T")


# Hardware configuration (from archimedes_lever)
DEFAULT_N_JOBS = 48  # Leave 16 cores for system (64 total)
MIN_TASKS_PER_CORE = 2  # Minimum tasks per core for efficient parallelization


def get_optimal_n_jobs(n_tasks: int, n_jobs: int = DEFAULT_N_JOBS) -> int:
    """
    Calculate optimal number of parallel jobs based on task count.

    Args:
        n_tasks: Total number of tasks to execute
        n_jobs: Maximum number of parallel jobs (default: 48)

    Returns:
        Optimal number of jobs (1 if sequential is better)

    Examples:
        >>> get_optimal_n_jobs(100, n_jobs=48)
        48
        >>> get_optimal_n_jobs(10, n_jobs=48)
        5
        >>> get_optimal_n_jobs(2, n_jobs=48)
        1
    """
    if n_tasks < MIN_TASKS_PER_CORE:
        return 1  # Sequential execution for tiny workloads

    # Use at most n_tasks // MIN_TASKS_PER_CORE cores
    optimal = min(n_jobs, n_tasks // MIN_TASKS_PER_CORE)

    return max(1, optimal)


def parallel_map(
    func: Callable[[Any], T],
    items: List[Any],
    n_jobs: int = DEFAULT_N_JOBS,
    show_progress: bool = True,
    desc: str = "Processing",
    backend: str = "loky",
) -> List[T]:
    """
    Parallel map with automatic optimization and progress tracking.

    Args:
        func: Function to apply to each item
        items: List of items to process
        n_jobs: Maximum number of parallel jobs (default: 48)
        show_progress: Show tqdm progress bar (default: True)
        desc: Progress bar description
        backend: Joblib backend ('loky', 'threading', 'multiprocessing')

    Returns:
        List of results in same order as input

    Examples:
        >>> def square(x): return x ** 2
        >>> results = parallel_map(square, [1, 2, 3, 4], n_jobs=2)
        >>> results
        [1, 4, 9, 16]
    """
    n_tasks = len(items)
    optimal_jobs = get_optimal_n_jobs(n_tasks, n_jobs)

    if optimal_jobs == 1:
        # Sequential execution
        if show_progress:
            return [func(item) for item in tqdm(items, desc=desc)]
        else:
            return [func(item) for item in items]

    # Parallel execution
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        if show_progress:
            # Use tqdm for progress tracking
            results = Parallel(n_jobs=optimal_jobs, backend=backend)(
                delayed(func)(item) for item in tqdm(items, desc=desc)
            )
        else:
            results = Parallel(n_jobs=optimal_jobs, backend=backend)(
                delayed(func)(item) for item in items
            )

    return results  # type: ignore[no-any-return]


def parallel_monte_carlo(
    simulation_func: Callable[..., T],
    n_simulations: int,
    n_jobs: int = DEFAULT_N_JOBS,
    show_progress: bool = True,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> List[T]:
    """
    Execute Monte Carlo simulations in parallel with reproducibility.

    Args:
        simulation_func: Function to run for each simulation
        n_simulations: Number of Monte Carlo runs
        n_jobs: Maximum number of parallel jobs (default: 48)
        show_progress: Show progress bar (default: True)
        random_state: Random seed for reproducibility (default: None)
        **kwargs: Additional arguments passed to simulation_func

    Returns:
        List of simulation results

    Examples:
        >>> def simulate(sim_id, mean=0.0):
        ...     rng = np.random.RandomState(sim_id)
        ...     return rng.normal(mean, 1.0)
        >>> results = parallel_monte_carlo(simulate, 1000, mean=2.0, n_jobs=4)
        >>> len(results)
        1000
    """
    # Generate simulation IDs with offsets for reproducibility
    if random_state is not None:
        sim_ids = [random_state + i for i in range(n_simulations)]
    else:
        sim_ids = list(range(n_simulations))

    # Create partial function with kwargs
    def run_single(sim_id: int) -> T:
        return simulation_func(sim_id=sim_id, **kwargs)

    # Execute in parallel
    results = parallel_map(
        run_single,
        sim_ids,
        n_jobs=n_jobs,
        show_progress=show_progress,
        desc="Monte Carlo",
    )

    return results


def parallelize(
    n_jobs: int = DEFAULT_N_JOBS,
    show_progress: bool = True,
    desc: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, List[T]]]:
    """
    Decorator to automatically parallelize a function over a list argument.

    The decorated function should take an iterable as its first argument.

    Args:
        n_jobs: Maximum number of parallel jobs (default: 48)
        show_progress: Show progress bar (default: True)
        desc: Progress bar description (default: function name)

    Returns:
        Decorated function that executes in parallel

    Examples:
        >>> @parallelize(n_jobs=4)
        ... def process_batch(items):
        ...     def process_single(item):
        ...         return item ** 2
        ...     return process_single
        >>> # Usage: results = process_batch([1, 2, 3, 4])
    """

    def decorator(func: Callable[P, Callable[[Any], T]]) -> Callable[P, List[T]]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> List[T]:
            # Get the processing function and items
            items = args[0] if args else kwargs.get("items", [])

            if not isinstance(items, (list, tuple, np.ndarray)):
                raise TypeError(f"First argument must be iterable, got {type(items).__name__}")

            # Get the single-item processing function
            process_func = func(*args, **kwargs)

            # Execute in parallel
            progress_desc = desc or func.__name__
            results = parallel_map(
                process_func,
                list(items),
                n_jobs=n_jobs,
                show_progress=show_progress,
                desc=progress_desc,
            )

            return results

        return wrapper

    return decorator  # type: ignore[return-value]


class ParallelExecutor:
    """
    Context manager for parallel execution with resource cleanup.

    Provides controlled parallel execution with automatic cleanup and
    error handling for long-running validation workflows.

    Examples:
        >>> with ParallelExecutor(n_jobs=4) as executor:
        ...     results = executor.map(lambda x: x**2, [1, 2, 3, 4])
        >>> results
        [1, 4, 9, 16]
    """

    def __init__(
        self,
        n_jobs: int = DEFAULT_N_JOBS,
        backend: str = "loky",
        show_progress: bool = True,
    ):
        """
        Initialize parallel executor.

        Args:
            n_jobs: Maximum number of parallel jobs (default: 48)
            backend: Joblib backend ('loky', 'threading', 'multiprocessing')
            show_progress: Show progress bars (default: True)
        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.show_progress = show_progress
        self._parallel: Optional[Parallel] = None

    def __enter__(self) -> "ParallelExecutor":
        """Enter context manager."""
        self._parallel = Parallel(n_jobs=self.n_jobs, backend=self.backend)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup resources."""
        if self._parallel is not None:
            # Force cleanup of worker pool
            self._parallel = None

    def map(
        self,
        func: Callable[[Any], T],
        items: List[Any],
        desc: str = "Processing",
    ) -> List[T]:
        """
        Map function over items in parallel.

        Args:
            func: Function to apply
            items: Items to process
            desc: Progress bar description

        Returns:
            List of results
        """
        if self._parallel is None:
            raise RuntimeError("Executor not initialized (use 'with' statement)")

        optimal_jobs = get_optimal_n_jobs(len(items), self.n_jobs)

        if optimal_jobs == 1:
            # Sequential
            if self.show_progress:
                return [func(item) for item in tqdm(items, desc=desc)]
            else:
                return [func(item) for item in items]

        # Parallel
        if self.show_progress:
            results = self._parallel(delayed(func)(item) for item in tqdm(items, desc=desc))
        else:
            results = self._parallel(delayed(func)(item) for item in items)

        return results  # type: ignore[no-any-return]


def chunk_workload(n_tasks: int, n_chunks: int, balance: bool = True) -> List[tuple[int, int]]:
    """
    Split workload into chunks for parallel processing.

    Args:
        n_tasks: Total number of tasks
        n_chunks: Number of chunks to create
        balance: Balance chunk sizes (default: True)

    Returns:
        List of (start, end) tuples for each chunk

    Examples:
        >>> chunk_workload(100, 4)
        [(0, 25), (25, 50), (50, 75), (75, 100)]
        >>> chunk_workload(10, 3)
        [(0, 4), (4, 7), (7, 10)]
    """
    if n_tasks <= 0:
        raise ValueError(f"n_tasks must be positive, got {n_tasks}")
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")

    if n_chunks > n_tasks:
        n_chunks = n_tasks  # No point in more chunks than tasks

    if balance:
        # Balanced chunks (some may be 1 larger)
        base_size = n_tasks // n_chunks
        remainder = n_tasks % n_chunks

        chunks = []
        start = 0
        for i in range(n_chunks):
            # First 'remainder' chunks get 1 extra task
            chunk_size = base_size + (1 if i < remainder else 0)
            end = start + chunk_size
            chunks.append((start, end))
            start = end

        return chunks
    else:
        # Simple equal chunks (last may be larger)
        chunk_size = n_tasks // n_chunks
        chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_chunks - 1)]
        chunks.append(((n_chunks - 1) * chunk_size, n_tasks))
        return chunks
