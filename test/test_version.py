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
    """The installed temporalcv must satisfy the declared range (>=2.0.0,<3).

    Exact behavioral agreement is enforced separately by the golden-parity
    suite; this gate only catches a wrong-major or pre-v2 environment.
    """
    import temporalcv

    version = temporalcv.__version__
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    assert match is not None, f"unparseable temporalcv version: {version!r}"
    major, minor, patch = (int(g) for g in match.groups())
    assert major == 2, f"temporalcv major must be 2, got {version}"
    assert (major, minor, patch) >= (2, 0, 0)
