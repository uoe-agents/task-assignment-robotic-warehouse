"""JSONL logger."""

from __future__ import annotations

import json
from typing import Dict, TextIO


class JSONLLogger:
    def __init__(self, path: str) -> None:
        self._file: TextIO = open(path, "w", encoding="utf-8")

    def log(self, row: Dict[str, object]) -> None:
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
