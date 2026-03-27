"""Batch evaluation helper."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from .rollout import run_episode


def evaluate(
    env_fn: Callable[[], Any],
    policy: Any,
    episodes: int,
    max_steps: int,
    render: bool = False,
    seed: int | None = None,
) -> Dict[str, float] | List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for ep in range(episodes):
        env = env_fn()
        ep_seed = None if seed is None else seed + ep
        result = run_episode(env, policy, max_steps=max_steps, render=render, seed=ep_seed)
        env.close()
        result["episode"] = ep
        result["seed"] = ep_seed
        results.append(result)

    if not results:
        return []

    mean_return = float(np.mean([r.get("global_episode_return", 0.0) for r in results]))
    mean_deliveries = float(np.mean([r.get("total_deliveries", 0.0) for r in results]))
    mean_clashes = float(np.mean([r.get("total_clashes", 0.0) for r in results]))
    mean_stuck = float(np.mean([r.get("total_stuck", 0.0) for r in results]))
    mean_pick_rate = float(np.mean([r.get("pick_rate", 0.0) for r in results]))
    mean_episode_length = float(np.mean([r.get("episode_length", 0.0) for r in results]))
    mean_fps = float(np.mean([r.get("fps", 0.0) for r in results]))

    total_deliveries = float(np.sum([r.get("total_deliveries", 0.0) for r in results]))
    total_episode_length = float(np.sum([r.get("episode_length", 0.0) for r in results]))
    overall_pick_rate = 0.0
    if total_episode_length > 0:
        overall_pick_rate = total_deliveries * 3600.0 / (5.0 * total_episode_length)

    summary = {
        "episodes": float(episodes),
        "mean_return": mean_return,
        "mean_deliveries": mean_deliveries,
        "mean_clashes": mean_clashes,
        "mean_stuck": mean_stuck,
        "mean_pick_rate": mean_pick_rate,
        "overall_pick_rate": overall_pick_rate,
        "mean_episode_length": mean_episode_length,
        "mean_fps": mean_fps,
    }
    return {"summary": summary, "episodes": results}
