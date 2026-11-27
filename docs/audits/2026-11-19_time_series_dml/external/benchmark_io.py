"""Checksum-pinned, atomic downloads for the opt-in external replications."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


def _verify(path: Path, expected: str) -> None:
    with path.open("rb") as source:
        measured = hashlib.file_digest(source, "sha256").hexdigest()
    if measured != expected:
        raise ValueError(f"{path.name}: expected SHA-256 {expected}, got {measured}")


def fetch(
    url: str,
    cache_dir: Path,
    filename: str,
    sha256: str,
    *,
    user_agent: str | None = "dml-ts empirical replication audit/1.0",
) -> Path:
    """Reuse verified bytes or download and verify before publishing to the cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.exists():
        _verify(target, sha256)
        return target
    request = Request(url, headers={"User-Agent": user_agent} if user_agent else {})
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        with urlopen(request, timeout=90) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        _verify(temp_path, sha256)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target
