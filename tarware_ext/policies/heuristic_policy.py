"""Heuristic policy wrapper."""

from __future__ import annotations

import time
from typing import Any, Dict

from tarware.heuristic import heuristic_episode
from tarware_ext.runners.metrics import summarize_episode


class HeuristicPolicy:
    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self) -> None:
        return None

    def act(self, obs: Any) -> Any:
        raise NotImplementedError("HeuristicPolicy uses run_episode in episodic mode.")

    def run_episode(
        self,
        env: Any,
        seed: int | None = None,
        render: bool = False,
        max_steps: int | None = None,
    ) -> Dict[str, Any]:
        _ = max_steps
        start = time.time()
        infos, global_episode_return, episode_returns = heuristic_episode(
            env.unwrapped, render=render, seed=seed
        )
        metrics = summarize_episode(
            infos=infos,
            global_episode_return=float(global_episode_return),
            episode_returns=episode_returns,
        )
        duration_s = max(time.time() - start, 1e-9)
        metrics["fps"] = float(metrics["episode_length"] / duration_s) if metrics["episode_length"] else 0.0
        return metrics
