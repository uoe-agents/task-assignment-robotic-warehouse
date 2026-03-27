"""Training utilities."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
