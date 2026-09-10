"""Persistent result-file helpers shared by benchmark workflows."""

import json
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON without exposing a partially written result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)
