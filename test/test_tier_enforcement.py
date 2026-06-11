"""The tier-marker collection gate must fire loudly on unmarked tests.

Unit-tests the conftest hook with fake items, plus one end-to-end subprocess
test pinning REAL pytest collection semantics.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import conftest
import pytest

pytestmark = pytest.mark.tier1

_SENTINEL = pytest.mark.tier1.mark  # any Mark object works as a fake marker


class FakeItem:
    """Minimal stand-in exposing the item surface the hook touches."""

    def __init__(self, nodeid: str, marker_names: tuple[str, ...] = ()) -> None:
        self.nodeid = nodeid
        self._markers: dict[str, Any] = dict.fromkeys(marker_names, _SENTINEL)
        self.added: list[Any] = []

    def get_closest_marker(self, name: str) -> Any:
        return self._markers.get(name)

    def add_marker(self, marker: Any) -> None:
        self.added.append(marker)
        self._markers[marker.name] = marker


def test_marked_item_gets_tier_timeout() -> None:
    item = FakeItem("test_x.py::test_a", ("tier2",))
    conftest.pytest_collection_modifyitems(config=None, items=[item])

    (added,) = item.added
    assert added.name == "timeout"
    assert added.args == (conftest.TIER_CONFIGS["tier2"].timeout,)


def test_highest_tier_wins() -> None:
    """Overlapping class/method markers resolve to the HIGHEST tier."""
    item = FakeItem("test_x.py::test_b", ("tier2", "tier3"))
    conftest.pytest_collection_modifyitems(config=None, items=[item])

    (added,) = item.added
    assert added.args == (conftest.TIER_CONFIGS["tier3"].timeout,)


def test_explicit_timeout_not_overridden() -> None:
    item = FakeItem("test_x.py::test_c", ("tier1", "timeout"))
    conftest.pytest_collection_modifyitems(config=None, items=[item])

    assert item.added == []


def test_unmarked_item_fails_collection() -> None:
    good = FakeItem("test_x.py::test_ok", ("tier1",))
    bad = FakeItem("test_x.py::test_unmarked")
    with pytest.raises(pytest.UsageError, match=r"test_x\.py::test_unmarked"):
        conftest.pytest_collection_modifyitems(config=None, items=[good, bad])


@pytest.mark.tier2
def test_real_collection_rejects_unmarked_test(tmp_path: Path) -> None:
    """End-to-end: REAL pytest collection fails loudly on an unmarked test.

    Runs a subprocess pytest over a temp dir whose conftest loads the actual
    hook from this repo's test/conftest.py — pinning real collection
    semantics, not the FakeItem model.
    """
    repo_conftest = Path(__file__).resolve().parent / "conftest.py"
    (tmp_path / "conftest.py").write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('repo_conftest', {str(repo_conftest)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "pytest_collection_modifyitems = mod.pytest_collection_modifyitems\n"
    )
    (tmp_path / "test_unmarked_probe.py").write_text("def test_probe():\n    assert True\n")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )

    assert proc.returncode == 4, f"expected usage-error exit 4, got {proc.returncode}"
    combined = proc.stdout + proc.stderr
    assert "lack a tier marker" in combined
    assert "test_unmarked_probe.py::test_probe" in combined
