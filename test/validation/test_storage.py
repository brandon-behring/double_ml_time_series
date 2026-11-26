"""Regression tests for archiving scientific outputs across workers."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pytest

from dml_ts._persistence import write_json_atomically
from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.storage import ResultStorage
from dml_ts.validation.validation_result import ValidationResult

pytestmark = pytest.mark.tier2


def sample_result():
    return ValidationResult(
        method="trial",
        status="PASS",
        bias=0.0,
        mse=0.01,
        coverage=1.0,
        ci_lower=-0.1,
        ci_upper=0.1,
        n_simulations=2,
        timestamp=datetime(2026, 1, 1),
        metadata={"design": {"folds": np.int64(2)}},
        point_estimates=np.array([0.9, 1.1]),
        ci_estimates=np.array([[0.8, 1.0], [1.0, 1.2]]),
    )


def test_concurrent_writers_keep_all_result_payloads_and_metadata(tmp_path, monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr("dml_ts.validation.storage.datetime", FixedClock)
    result = sample_result()
    stores = [ResultStorage(tmp_path) for _ in range(8)]

    def save(i):
        return stores[i].save_result(result, metadata={"draw": np.int64(i)})

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        paths = list(pool.map(save, range(len(stores))))
    assert len(set(paths)) == len(stores)
    for store in (stores[0], ResultStorage(tmp_path)):
        results = store.load_results(method="trial")
        assert sorted(r.storage_metadata["draw"] for r in results) == list(range(len(stores)))
        for r in results:
            np.testing.assert_array_equal(r.ci_estimates, result.ci_estimates)
            np.testing.assert_array_equal(r.point_estimates, result.point_estimates)
            assert r.metadata == {"design": {"folds": 2}}
        assert store.get_summary()["total_results"] == len(stores)


def test_failed_archive_preserves_prior_index_and_payload(tmp_path):
    storage = ResultStorage(tmp_path)
    path = storage.save_result(sample_result())
    with pytest.raises(TypeError):
        storage.save_result(sample_result(), metadata={"unserializable": object()})
    assert list(storage.results_dir.iterdir()) == [path]
    assert len(storage.load_results()) == 1


def test_failed_index_update_discards_only_new_payload(tmp_path, monkeypatch):
    storage = ResultStorage(tmp_path)
    previous = storage.save_result(sample_result(), metadata={"draw": 1})

    def interrupt(path, data):
        if path == storage.index_file:
            raise OSError("simulated write failure")
        write_json_atomically(path, data)

    monkeypatch.setattr("dml_ts.validation.storage.write_json_atomically", interrupt)
    with pytest.raises(OSError, match="simulated write failure"):
        storage.save_result(sample_result(), metadata={"draw": 2})
    assert list(storage.results_dir.iterdir()) == [previous]
    assert [r.storage_metadata for r in storage.load_results()] == [{"draw": 1}]


def test_real_bias_validation_json_round_trip(tmp_path):
    pytest.importorskip("econml")
    result = BiasValidation(n_simulations=2, random_state=42).validate(
        DGPGenerator(n=100, p=2, true_effect=2.0, random_state=42)
    )
    storage = ResultStorage(tmp_path)
    restored = storage.load_result(storage.save_result(result, metadata={"seed": 42}))
    assert restored.storage_metadata == {"seed": 42}
    assert result.storage_metadata == {}
    for copy in (restored, ValidationResult.from_json(result.to_json())):
        np.testing.assert_array_equal(copy.ci_estimates, result.ci_estimates)
        np.testing.assert_array_equal(copy.point_estimates, result.point_estimates)
        assert copy.metadata == result.metadata
        assert copy.status == result.status
