"""Feature engineering helpers."""

from __future__ import annotations

from typing import Tuple


def normalize_coordinates(coord: Tuple[int, int], grid_size: Tuple[int, int]) -> Tuple[float, float]:
    y, x = coord
    h, w = grid_size
    if h <= 1 or w <= 1:
        return 0.0, 0.0
    return y / (h - 1), x / (w - 1)
