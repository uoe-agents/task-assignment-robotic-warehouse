"""Metrics helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def summarize_episode(
    infos: Iterable[Dict[str, Any]],
    global_episode_return: float,
    episode_returns: Iterable[float],
) -> Dict[str, float | int | list]:
    total_deliveries = 0.0
    total_clashes = 0.0
    total_stuck = 0.0
    info_list = list(infos)
    for info in info_list:
        total_deliveries += float(info.get("shelf_deliveries", 0))
        total_clashes += float(info.get("clashes", 0))
        total_stuck += float(info.get("stucks", 0))

    episode_length = len(info_list)
    pick_rate = 0.0
    if episode_length > 0:
        pick_rate = total_deliveries * 3600.0 / (5.0 * episode_length)

    return {
        "episode_length": episode_length,
        "global_episode_return": float(global_episode_return),
        "episode_returns": list(episode_returns),
        "total_deliveries": total_deliveries,
        "total_clashes": total_clashes,
        "total_stuck": total_stuck,
        "shelf_deliveries": total_deliveries,
        "clashes": total_clashes,
        "stucks": total_stuck,
        "pick_rate": pick_rate,
    }
