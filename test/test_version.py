"""Version-consistency gate: every release-facing version site must agree.

Deliberately file-parsing (not importlib.metadata): metadata goes stale under
editable installs, while the files are what a release actually ships.
"""

import re
from pathlib import Path

import pytest

import dml_ts

pytestmark = pytest.mark.tier1

ROOT = Path(__file__).resolve().parents[1]


def _extract(relpath: str, pattern: str) -> str:
    text = (ROOT / relpath).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"version pattern {pattern!r} not found in {relpath}"
    return match.group(1)


def test_version_sites_agree() -> None:
    """pyproject, sphinx conf, CITATION.cff, and the package must match."""
    pyproject = _extract("pyproject.toml", r'^version = "([^"]+)"')
    sphinx_conf = _extract("docs/sphinx/conf.py", r'^release = "([^"]+)"')
    citation = _extract("CITATION.cff", r'^version: "([^"]+)"')

    assert dml_ts.__version__ == pyproject
    assert sphinx_conf == pyproject
    assert citation == pyproject


def test_temporalcv_pin_resolves() -> None:
    """The temporalcv git pin must import at the pinned release version."""
    import temporalcv

    assert temporalcv.__version__ == "2.0.0"
