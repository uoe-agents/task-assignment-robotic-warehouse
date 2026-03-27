"""Seed utilities for reproducible runs."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
