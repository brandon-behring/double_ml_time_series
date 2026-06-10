"""The tier-marker collection gate must fire loudly on unmarked tests.

Unit-tests the conftest hook directly with fake items; the wiring itself is
exercised by every real pytest run (the CI lint job's --collect-only step sees
every test regardless of -m filtering).
"""

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


def test_most_specific_tier_wins() -> None:
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
