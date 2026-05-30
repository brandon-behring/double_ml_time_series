"""Tier-1 tests for the LaTeX<->MDX provenance drift guard.

Exercises ``scripts/check_tex_mdx_drift.py`` against synthetic chapter pairs in
``tmp_path`` (so the canonical .tex files are never touched), plus one regression
assertion against the real Chapter 1 MDX (the W3 baseline).
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1


# The guard lives in scripts/, which is not an importable package; load by path.
_GUARD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_tex_mdx_drift.py"
_spec = importlib.util.spec_from_file_location("check_tex_mdx_drift", _GUARD_PATH)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _write_pair(
    tmp_path: Path,
    tex_body: str,
    *,
    record_sha: str | None = "auto",
    include_source_file: bool = True,
) -> tuple[Path, Path]:
    """Create a (repo_root, mdx_dir) layout with one chapter pair under tmp_path.

    ``record_sha="auto"`` stamps the true hash of ``tex_body``; pass an explicit
    hex string to force drift, or ``None`` to omit the ``source_sha256`` key.
    Returns ``(repo_root, mdx_dir)``.
    """
    repo_root = tmp_path
    chapters = repo_root / "chapters"
    chapters.mkdir()
    (chapters / "chapter_07.tex").write_text(tex_body, encoding="utf-8")

    mdx_dir = repo_root / "web" / "src" / "content" / "chapters"
    mdx_dir.mkdir(parents=True)

    if record_sha == "auto":
        record_sha = hashlib.sha256(tex_body.encode("utf-8")).hexdigest()

    lines = ["---", "title: Demo", "provenance:", "  ai_tools:", "    - Claude Code"]
    if include_source_file:
        lines.append("  source_file: chapters/chapter_07.tex")
    if record_sha is not None:
        lines.append(f"  source_sha256: {record_sha}")
    lines += ["---", "", "# Demo chapter", ""]
    (mdx_dir / "chapter07-demo.mdx").write_text("\n".join(lines), encoding="utf-8")
    return repo_root, mdx_dir


def test_matching_hash_has_no_problems(tmp_path: Path) -> None:
    """A recorded hash equal to the current .tex hash is clean."""
    repo_root, mdx_dir = _write_pair(tmp_path, "Original tex body.\n")
    assert guard.run_check(mdx_dir, repo_root) == []


def test_mutated_tex_reports_drift(tmp_path: Path) -> None:
    """Editing the .tex after the hash was recorded is flagged as drift."""
    repo_root, mdx_dir = _write_pair(tmp_path, "Original tex body.\n")
    (repo_root / "chapters" / "chapter_07.tex").write_text("Edited body.\n", encoding="utf-8")

    problems = guard.run_check(mdx_dir, repo_root)

    assert len(problems) == 1
    assert problems[0].kind == "drift"


def test_mdx_without_provenance_is_skipped(tmp_path: Path) -> None:
    """An MDX that declares no source (unported placeholder) is not checked."""
    repo_root = tmp_path
    mdx_dir = repo_root / "web" / "src" / "content" / "chapters"
    mdx_dir.mkdir(parents=True)
    (mdx_dir / "chapter02-plain.mdx").write_text(
        "---\ntitle: Plain\n---\n\n# No provenance\n", encoding="utf-8"
    )

    assert guard.run_check(mdx_dir, repo_root) == []


def test_source_file_without_sha_is_malformed(tmp_path: Path) -> None:
    """Declaring source_file but omitting source_sha256 is a hard error."""
    repo_root, mdx_dir = _write_pair(tmp_path, "Body.\n", record_sha=None)

    problems = guard.run_check(mdx_dir, repo_root)

    assert len(problems) == 1
    assert problems[0].kind == "malformed"


def test_missing_source_file_is_reported(tmp_path: Path) -> None:
    """A source_file pointing at a non-existent .tex is reported."""
    repo_root, mdx_dir = _write_pair(tmp_path, "Body.\n", record_sha="de" * 32)
    (repo_root / "chapters" / "chapter_07.tex").unlink()

    problems = guard.run_check(mdx_dir, repo_root)

    assert len(problems) == 1
    assert problems[0].kind == "missing-source"


def test_update_restamps_and_then_passes(tmp_path: Path) -> None:
    """--update rewrites the stale hash; a subsequent check is clean."""
    repo_root, mdx_dir = _write_pair(tmp_path, "Original.\n")
    (repo_root / "chapters" / "chapter_07.tex").write_text("Edited body.\n", encoding="utf-8")

    assert len(guard.run_check(mdx_dir, repo_root)) == 1  # drift exists
    assert guard.run_check(mdx_dir, repo_root, update=True) == []  # re-stamped
    assert guard.run_check(mdx_dir, repo_root) == []  # persisted + clean

    mdx_text = (mdx_dir / "chapter07-demo.mdx").read_text(encoding="utf-8")
    assert hashlib.sha256(b"Edited body.\n").hexdigest() in mdx_text


def test_real_chapter01_matches_recorded_hash() -> None:
    """The live Ch1 MDX must match chapters/chapter_01.tex (W3 baseline regression)."""
    problems = guard.run_check(guard.MDX_CHAPTERS_DIR, guard.REPO_ROOT)
    assert problems == [], f"unexpected provenance drift in real chapters: {problems}"
