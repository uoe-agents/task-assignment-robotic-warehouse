"""Helpers to list and validate Gymnasium env ids."""

from __future__ import annotations

from typing import List

import gymnasium as gym


def _all_specs() -> list:
    registry = gym.envs.registry
    if hasattr(registry, "values"):
        return list(registry.values())
    return list(registry)


def list_env_ids(prefix: str | None = None) -> List[str]:
    ids = []
    for spec in _all_specs():
        env_id = spec.id if hasattr(spec, "id") else str(spec)
        ids.append(env_id)
    ids = sorted(set(ids))
    if prefix:
        ids = [env_id for env_id in ids if env_id.startswith(prefix)]
    return ids


def filter_env_ids(prefix: str) -> List[str]:
    return list_env_ids(prefix=prefix)


def is_env_id_valid(env_id: str) -> bool:
    return env_id in set(list_env_ids())
