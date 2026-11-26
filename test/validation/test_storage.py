"""Regression tests for archiving scientific outputs across workers."""

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

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


def test_an_error_after_the_index_update_keeps_the_payload(tmp_path, monkeypatch):
    """Once the index names a payload, a later error must not delete that payload."""

    class ReleaseFails:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            raise OSError("simulated failure releasing the lock")

    storage = ResultStorage(tmp_path)
    monkeypatch.setattr("dml_ts.validation.storage.FileLock", ReleaseFails)
    with pytest.raises(OSError, match="releasing the lock"):
        storage.save_result(sample_result(), metadata={"draw": 1})
    monkeypatch.undo()
    assert [r.storage_metadata for r in ResultStorage(tmp_path).load_results()] == [{"draw": 1}]


def test_a_moved_store_still_loads_its_results(tmp_path):
    """The index records where a payload was saved; a renamed project keeps it beside the index.

    With absolute recorded paths and no fallback, ``load_results`` on the moved store
    returned an empty list with no error while ``get_summary`` still counted the results.
    """
    original = ResultStorage(tmp_path / "project" / "store")
    original.save_result(sample_result(), metadata={"draw": 1})
    original.save_result(sample_result(), metadata={"draw": 2})
    (tmp_path / "project").rename(tmp_path / "renamed")
    moved = ResultStorage(tmp_path / "renamed" / "store")
    assert moved.get_summary()["total_results"] == 2
    assert sorted(r.storage_metadata["draw"] for r in moved.load_results()) == [1, 2]


def test_a_relative_store_survives_a_change_of_directory(tmp_path, monkeypatch):
    """The default ``base_dir`` is relative; the object must keep meaning the store it opened."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    storage = ResultStorage(Path("store"))
    storage.save_result(sample_result(), metadata={"draw": 1})
    monkeypatch.chdir(tmp_path / "b")
    assert len(storage.load_results()) == 1
    storage.save_result(sample_result(), metadata={"draw": 2})
    assert storage.get_summary()["total_results"] == 2
    assert list((tmp_path / "b").iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_published_files_have_the_permissions_of_an_ordinary_file(tmp_path):
    """``tempfile`` creates files only their owner can read, and a rename keeps that mode."""
    storage = ResultStorage(tmp_path)
    payload = storage.save_result(sample_result())
    umask = os.umask(0)
    os.umask(umask)
    for published in (payload, storage.index_file):
        assert stat.S_IMODE(published.stat().st_mode) == 0o666 & ~umask


def test_an_interrupted_write_leaves_the_previous_file_and_no_temporary_one(tmp_path, monkeypatch):
    """The content reaches its name only by a rename of a complete file."""
    target = tmp_path / "document.json"
    write_json_atomically(target, {"version": 1})

    def crash(self, destination):
        raise OSError("simulated crash before the rename")

    monkeypatch.setattr(Path, "replace", crash)
    with pytest.raises(OSError, match="before the rename"):
        write_json_atomically(target, {"version": 2})
    assert json.loads(target.read_text()) == {"version": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["document.json"]


def test_a_document_that_cannot_be_serialized_creates_no_file(tmp_path):
    with pytest.raises(TypeError):
        write_json_atomically(tmp_path / "document.json", {"value": object()})
    assert list(tmp_path.iterdir()) == []


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
