"""Publish complete JSON files for the validation store and model registry."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_json_atomically(path: Path, data: dict[str, Any]) -> None:
    """Readers see either the previous complete document or its replacement."""
    file = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(file.name)
    try:
        with file:
            json.dump(data, file, indent=2)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
