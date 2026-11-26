"""Publish complete files: the validation store, the model registry, a data cache."""

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_bytes_atomically(path: Path, content: bytes) -> None:
    """Readers see either the previous complete file or its replacement.

    The content is written to a temporary file beside ``path`` and moved over it. The
    temporary file is created as ``open()`` would create it (mode 0666 less the umask):
    ``tempfile`` creates files that only their owner can read, and the published file
    would inherit that mode.
    """
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with open(descriptor, "wb") as file:
            file.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomically(path: Path, data: dict[str, Any]) -> None:
    """Publish ``data`` as one complete JSON document; a document that cannot be
    serialized raises before any file is created."""
    write_bytes_atomically(path, json.dumps(data, indent=2).encode("utf-8"))
