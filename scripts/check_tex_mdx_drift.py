#!/usr/bin/env python3
"""
LaTeX <-> MDX provenance drift guard for the Double ML book companion.

Each ported web chapter (``web/src/content/chapters/*.mdx``) declares, in its
frontmatter ``provenance`` block, the canonical LaTeX file it was ported from
(``source_file``) and the SHA-256 of that file at port time (``source_sha256``).
This guard recomputes the hash of every declared ``source_file`` and fails when
it no longer matches the recorded ``source_sha256`` -- i.e. the canonical .tex
changed without the ported MDX being re-synced (or at least re-stamped).

The check is stateless: it depends only on current file contents, so it runs
identically in a pre-commit hook, in CI, or ad hoc. Chapters that are not yet
ported simply have no MDX and are never checked.

Usage:
    python scripts/check_tex_mdx_drift.py            # check (CI / pre-commit)
    python scripts/check_tex_mdx_drift.py --update   # re-stamp drifted hashes

Exit codes:
    0: every ported chapter's source_sha256 matches its .tex (or --update fixed it)
    1: drift or malformed provenance detected (and not --update'd)
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MDX_CHAPTERS_DIR = REPO_ROOT / "web" / "src" / "content" / "chapters"

# Frontmatter is the leading ``---`` ... ``---`` block. The two provenance keys
# we read are unique scalars, so targeted line regexes suffice (no YAML dep).
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---", re.DOTALL)
_SOURCE_FILE_RE = re.compile(r"^\s*source_file:\s*(.+?)\s*$", re.MULTILINE)
_SOURCE_SHA_RE = re.compile(r"^\s*source_sha256:\s*([0-9a-fA-F]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Problem:
    """A single provenance issue found in a chapter MDX file.

    Attributes:
        mdx: Path to the offending MDX file.
        kind: One of ``"drift"``, ``"malformed"``, or ``"missing-source"``.
        detail: Human-readable explanation of the issue.
    """

    mdx: Path
    kind: str
    detail: str


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``'s bytes."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def extract_provenance(mdx_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract ``(source_file, source_sha256)`` from an MDX file's frontmatter.

    Only the leading frontmatter block is scanned. Either element is ``None``
    when its key is absent (or, for the hash, malformed/non-hex). Surrounding
    quotes on ``source_file`` are stripped; the hash is lower-cased.
    """
    match = _FRONTMATTER_RE.search(mdx_text)
    if match is None:
        return None, None
    frontmatter = match.group(1)

    file_match = _SOURCE_FILE_RE.search(frontmatter)
    sha_match = _SOURCE_SHA_RE.search(frontmatter)
    source_file = file_match.group(1).strip().strip("\"'") if file_match else None
    source_sha = sha_match.group(1).strip().lower() if sha_match else None
    return source_file, source_sha


def check_mdx_file(mdx_path: Path, repo_root: Path, update: bool) -> Optional[Problem]:
    """Check one chapter MDX against its declared LaTeX source.

    Returns a :class:`Problem` describing any drift or malformed provenance, or
    ``None`` when the file is consistent (or legitimately unported and skipped).
    When ``update`` is true and the only issue is a stale hash, the MDX's
    ``source_sha256`` is rewritten in place and ``None`` is returned.
    """
    text = mdx_path.read_text(encoding="utf-8")
    source_file, source_sha = extract_provenance(text)

    # No provenance source at all -> not a tracked port; skip silently.
    if source_file is None and source_sha is None:
        return None

    # Half-declared provenance is a hard error (cannot be auto-fixed safely).
    if source_file is None:
        return Problem(mdx_path, "malformed", "source_sha256 present without source_file")
    if source_sha is None:
        return Problem(mdx_path, "malformed", "source_file present without source_sha256")

    tex_path = (repo_root / source_file).resolve()
    if not tex_path.is_file():
        return Problem(mdx_path, "missing-source", f"source_file does not exist: {source_file}")

    actual_sha = compute_sha256(tex_path)
    if actual_sha == source_sha:
        return None

    if update:
        new_text = _SOURCE_SHA_RE.sub(
            lambda m: m.group(0).replace(m.group(1), actual_sha), text, count=1
        )
        mdx_path.write_text(new_text, encoding="utf-8")
        return None

    return Problem(
        mdx_path,
        "drift",
        f"{source_file} changed: recorded {source_sha[:12]}..., actual {actual_sha[:12]}...",
    )


def run_check(mdx_dir: Path, repo_root: Path, update: bool = False) -> List[Problem]:
    """Check every ``*.mdx`` in ``mdx_dir`` against its declared LaTeX source.

    ``repo_root`` is the base each ``source_file`` is resolved against. Returns
    the list of unresolved :class:`Problem`s (empty == all consistent). A
    non-existent ``mdx_dir`` yields an empty list (nothing ported yet).
    """
    problems: List[Problem] = []
    if not mdx_dir.is_dir():
        return problems
    for mdx_path in sorted(mdx_dir.glob("*.mdx")):
        problem = check_mdx_file(mdx_path, repo_root, update)
        if problem is not None:
            problems.append(problem)
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    """Run the drift guard over the repo's web chapters."""
    parser = argparse.ArgumentParser(
        description="Enforce LaTeX <-> MDX provenance: each ported chapter MDX's "
        "source_sha256 must match the current hash of its source_file .tex.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite each drifted source_sha256 to the current .tex hash",
    )
    args = parser.parse_args(argv)

    problems = run_check(MDX_CHAPTERS_DIR, REPO_ROOT, update=args.update)
    rel_dir = MDX_CHAPTERS_DIR.relative_to(REPO_ROOT)

    if not problems:
        print(f"tex<->mdx drift guard: OK -- ported chapters in {rel_dir} match their .tex.")
        return 0

    print(f"tex<->mdx drift guard: {len(problems)} problem(s) in {rel_dir}:")
    for problem in problems:
        try:
            label = problem.mdx.relative_to(REPO_ROOT)
        except ValueError:
            label = problem.mdx
        print(f"  [{problem.kind}] {label}: {problem.detail}")
    print()
    print("Re-port the MDX to match the .tex, then record the new hash with:")
    print("    venv/bin/python scripts/check_tex_mdx_drift.py --update")
    return 1


if __name__ == "__main__":
    sys.exit(main())
