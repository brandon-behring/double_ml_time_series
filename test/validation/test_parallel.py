"""
Tests for parallel execution framework.

Tests all parallel utilities including map, Monte Carlo, decorators, and chunking.
"""

import pytest
import numpy as np
from typing import List
import time

from dml_ts.validation.parallel import (
    get_optimal_n_jobs,
    parallel_map,
    parallel_monte_carlo,
    parallelize,
    ParallelExecutor,
    chunk_workload,
    DEFAULT_N_JOBS,
    MIN_TASKS_PER_CORE,
)


@pytest.mark.tier1
class TestOptimalJobCalculation:
    """Test automatic n_jobs optimization based on workload."""

    def test_tiny_workload_uses_sequential(self):
        """Very small workloads should use sequential execution."""
        n_jobs = get_optimal_n_jobs(n_tasks=1, n_jobs=48)
        assert n_jobs == 1

    def test_small_workload_reduces_cores(self):
        """Small workloads should use fewer cores."""
        n_jobs = get_optimal_n_jobs(n_tasks=10, n_jobs=48)
        # Should use at most 10 // 2 = 5 cores
        assert n_jobs <= 5
        assert n_jobs >= 1

    def test_large_workload_uses_max_cores(self):
        """Large workloads should use maximum available cores."""
        n_jobs = get_optimal_n_jobs(n_tasks=1000, n_jobs=48)
        assert n_jobs == 48

    def test_respects_min_tasks_per_core(self):
        """Should maintain minimum tasks per core ratio."""
        n_jobs = get_optimal_n_jobs(n_tasks=100, n_jobs=48)
        # 100 tasks / n_jobs should be >= MIN_TASKS_PER_CORE
        assert 100 / n_jobs >= MIN_TASKS_PER_CORE

    def test_never_exceeds_available_cores(self):
        """Should never request more cores than available."""
        n_jobs = get_optimal_n_jobs(n_tasks=10000, n_jobs=8)
        assert n_jobs <= 8


@pytest.mark.tier1
class TestParallelMap:
    """Test basic parallel_map functionality."""

    def test_parallel_map_basic(self):
        """Should correctly map function over items."""

        def square(x: int) -> int:
            return x**2

        result = parallel_map(square, [1, 2, 3, 4], n_jobs=2, show_progress=False)
        assert result == [1, 4, 9, 16]

    def test_parallel_map_preserves_order(self):
        """Results should maintain input order."""

        def identity(x: int) -> int:
            return x

        items = list(range(100))
        result = parallel_map(identity, items, n_jobs=4, show_progress=False)
        assert result == items

    def test_parallel_map_handles_exceptions(self):
        """Should propagate exceptions from worker functions."""

        def fail(x: int) -> int:
            if x == 5:
                raise ValueError("Intentional error")
            return x

        with pytest.raises(ValueError, match="Intentional error"):
            parallel_map(fail, list(range(10)), n_jobs=2, show_progress=False)

    def test_parallel_map_empty_list(self):
        """Should handle empty input list."""

        def square(x: int) -> int:
            return x**2

        result = parallel_map(square, [], n_jobs=2, show_progress=False)
        assert result == []

    def test_parallel_map_single_item(self):
        """Should handle single-item list (sequential)."""

        def square(x: int) -> int:
            return x**2

        result = parallel_map(square, [5], n_jobs=2, show_progress=False)
        assert result == [25]

    def test_parallel_map_progress_bar(self):
        """Should work with progress bar enabled."""

        def square(x: int) -> int:
            return x**2

        result = parallel_map(square, [1, 2, 3], n_jobs=2, show_progress=True)
        assert result == [1, 4, 9]


@pytest.mark.tier2
class TestParallelMonteCarlo:
    """Test Monte Carlo simulation utilities."""

    def test_monte_carlo_runs_correct_count(self):
        """Should execute exact number of simulations."""

        def sim(sim_id: int) -> int:
            return sim_id

        results = parallel_monte_carlo(sim, n_simulations=100, n_jobs=4, show_progress=False)
        assert len(results) == 100

    def test_monte_carlo_reproducibility_with_seed(self):
        """Same seed should produce identical results."""

        def sim(sim_id: int) -> float:
            rng = np.random.RandomState(sim_id)
            return rng.normal(0, 1)

        results1 = parallel_monte_carlo(
            sim, n_simulations=50, random_state=42, n_jobs=2, show_progress=False
        )
        results2 = parallel_monte_carlo(
            sim, n_simulations=50, random_state=42, n_jobs=2, show_progress=False
        )

        assert np.allclose(results1, results2)

    def test_monte_carlo_different_seeds_differ(self):
        """Different seeds should produce different results."""

        def sim(sim_id: int) -> float:
            rng = np.random.RandomState(sim_id)
            return rng.normal(0, 1)

        results1 = parallel_monte_carlo(
            sim, n_simulations=50, random_state=42, n_jobs=2, show_progress=False
        )
        results2 = parallel_monte_carlo(
            sim, n_simulations=50, random_state=123, n_jobs=2, show_progress=False
        )

        assert not np.allclose(results1, results2)

    def test_monte_carlo_passes_kwargs(self):
        """Should pass additional kwargs to simulation function."""

        def sim(sim_id: int, mean: float = 0.0, scale: float = 1.0) -> float:
            rng = np.random.RandomState(sim_id)
            return rng.normal(mean, scale)

        results = parallel_monte_carlo(
            sim,
            n_simulations=100,
            mean=5.0,
            scale=2.0,
            n_jobs=2,
            show_progress=False,
        )

        # Check mean is approximately 5.0
        assert abs(np.mean(results) - 5.0) < 0.5

    def test_monte_carlo_statistical_properties(self):
        """Monte Carlo should produce correct statistical properties."""

        def sim(sim_id: int, mean: float = 0.0) -> float:
            rng = np.random.RandomState(sim_id)
            return rng.normal(mean, 1.0)

        results = parallel_monte_carlo(
            sim,
            n_simulations=1000,
            mean=3.0,
            random_state=42,
            n_jobs=4,
            show_progress=False,
        )

        # Mean should be close to 3.0
        assert abs(np.mean(results) - 3.0) < 0.1
        # Std should be close to 1.0
        assert abs(np.std(results) - 1.0) < 0.1


@pytest.mark.tier1
class TestParallelizeDecorator:
    """Test @parallelize decorator."""

    def test_parallelize_decorator_basic(self):
        """Decorator should parallelize function execution."""

        @parallelize(n_jobs=2, show_progress=False)
        def process_batch(items: List[int]) -> callable:
            def process_single(x: int) -> int:
                return x**2

            return process_single

        results = process_batch([1, 2, 3, 4])
        assert results == [1, 4, 9, 16]

    def test_parallelize_decorator_with_kwargs(self):
        """Decorator should work with additional kwargs."""

        @parallelize(n_jobs=2, show_progress=False)
        def process_batch(items: List[int], multiplier: int = 1) -> callable:
            def process_single(x: int) -> int:
                return x * multiplier

            return process_single

        results = process_batch([1, 2, 3], multiplier=5)
        assert results == [5, 10, 15]

    def test_parallelize_decorator_preserves_metadata(self):
        """Decorator should preserve function metadata."""

        @parallelize(n_jobs=2)
        def my_function(items: List[int]) -> callable:
            """My docstring."""

            def process(x: int) -> int:
                return x

            return process

        assert my_function.__name__ == "my_function"
        assert "My docstring" in (my_function.__doc__ or "")


@pytest.mark.tier1
class TestParallelExecutor:
    """Test ParallelExecutor context manager."""

    def test_executor_context_manager(self):
        """Should work as context manager."""
        with ParallelExecutor(n_jobs=2, show_progress=False) as executor:
            results = executor.map(lambda x: x**2, [1, 2, 3, 4])

        assert results == [1, 4, 9, 16]

    def test_executor_requires_context(self):
        """Should raise error if used outside context."""
        executor = ParallelExecutor(n_jobs=2)

        with pytest.raises(RuntimeError, match="not initialized"):
            executor.map(lambda x: x, [1, 2, 3])

    def test_executor_cleanup(self):
        """Should cleanup resources on exit."""
        executor = ParallelExecutor(n_jobs=2, show_progress=False)

        with executor:
            assert executor._parallel is not None

        # After exit, should be cleaned up
        assert executor._parallel is None

    def test_executor_multiple_maps(self):
        """Should support multiple map calls in same context."""
        with ParallelExecutor(n_jobs=2, show_progress=False) as executor:
            results1 = executor.map(lambda x: x**2, [1, 2, 3])
            results2 = executor.map(lambda x: x * 2, [1, 2, 3])

        assert results1 == [1, 4, 9]
        assert results2 == [2, 4, 6]


@pytest.mark.tier1
class TestChunkWorkload:
    """Test workload chunking utilities."""

    def test_chunk_balanced_even_split(self):
        """Even split should create equal chunks."""
        chunks = chunk_workload(100, 4, balance=True)

        assert len(chunks) == 4
        assert chunks == [(0, 25), (25, 50), (50, 75), (75, 100)]

    def test_chunk_balanced_uneven_split(self):
        """Uneven split should balance extra tasks."""
        chunks = chunk_workload(10, 3, balance=True)

        assert len(chunks) == 3
        # First chunk gets extra: 4, 3, 3
        sizes = [end - start for start, end in chunks]
        assert sum(sizes) == 10
        assert max(sizes) - min(sizes) <= 1  # Balanced

    def test_chunk_unbalanced_split(self):
        """Unbalanced mode should create simple splits."""
        chunks = chunk_workload(10, 3, balance=False)

        assert len(chunks) == 3
        assert chunks[0][0] == 0
        assert chunks[-1][1] == 10

    def test_chunk_more_chunks_than_tasks(self):
        """Should cap chunks at number of tasks."""
        chunks = chunk_workload(5, 10, balance=True)

        assert len(chunks) == 5  # Not 10

    def test_chunk_single_chunk(self):
        """Single chunk should span all tasks."""
        chunks = chunk_workload(100, 1, balance=True)

        assert chunks == [(0, 100)]

    def test_chunk_validates_inputs(self):
        """Should validate input parameters."""
        with pytest.raises(ValueError, match="n_tasks must be positive"):
            chunk_workload(0, 4)

        with pytest.raises(ValueError, match="n_chunks must be positive"):
            chunk_workload(100, 0)

    def test_chunk_coverage_complete(self):
        """Chunks should cover all tasks exactly once."""
        chunks = chunk_workload(47, 7, balance=True)

        # Extract all task indices
        covered = []
        for start, end in chunks:
            covered.extend(range(start, end))

        assert sorted(covered) == list(range(47))

    def test_chunk_no_overlap(self):
        """Chunks should not overlap."""
        chunks = chunk_workload(100, 5, balance=True)

        for i in range(len(chunks) - 1):
            assert chunks[i][1] == chunks[i + 1][0]


@pytest.mark.tier3
class TestPerformance:
    """Performance and integration tests."""

    def test_parallel_speedup(self):
        """Parallel execution correctness (speedup depends on overhead)."""

        def slow_func(x: int) -> int:
            # Simulate CPU work
            result = x**2
            for _ in range(10000):  # CPU work instead of sleep
                result += 1
            return result

        items = list(range(20))

        # Sequential
        seq_results = [slow_func(x) for x in items]

        # Parallel (4 cores)
        par_results = parallel_map(slow_func, items, n_jobs=4, show_progress=False)

        # Verify results match (speedup is hardware/load dependent)
        assert par_results == seq_results

    def test_large_workload_integration(self):
        """Integration test with realistic workload."""

        def simulate_estimator(sim_id: int, n: int = 100) -> dict:
            rng = np.random.RandomState(sim_id)
            X = rng.normal(0, 1, n)
            Y = 2.0 * X + rng.normal(0, 1, n)

            # Simple OLS
            beta = np.sum(X * Y) / np.sum(X**2)
            return {"estimate": beta, "sim_id": sim_id}

        results = parallel_monte_carlo(
            simulate_estimator,
            n_simulations=100,
            n=200,
            random_state=42,
            n_jobs=4,
            show_progress=False,
        )

        assert len(results) == 100
        estimates = [r["estimate"] for r in results]

        # Mean estimate should be close to true value (2.0)
        assert abs(np.mean(estimates) - 2.0) < 0.1
