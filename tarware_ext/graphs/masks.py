"""Action mask helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_action_masks(env: Any) -> np.ndarray:
    if hasattr(env, "compute_valid_action_masks"):
        return env.compute_valid_action_masks()
    raise NotImplementedError("Env does not expose compute_valid_action_masks.")
