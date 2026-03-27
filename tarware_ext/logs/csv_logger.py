"""CSV logger."""

from __future__ import annotations

import csv
from typing import Dict, Iterable, TextIO


class CSVLogger:
    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self._file: TextIO = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
