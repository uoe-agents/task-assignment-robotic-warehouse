"""Episode rollout helper."""

from __future__ import annotations

import time
from typing import Any, Dict, Sequence

import numpy as np

from tarware_ext.envs import Transition
from .metrics import summarize_episode


def _as_seq(x: Any) -> Sequence:
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    return [x]

# Note: this function is not intended to be a general-purpose rollout helper; it's designed to work with the specific policies and envs in this codebase. It can be extended or modified as needed for different use cases.
def run_episode(env: Any, policy: Any, max_steps: int, render: bool = False, seed: int | None = None) -> Dict[str, Any]:
    if hasattr(policy, "run_episode"):
        return policy.run_episode(env, seed=seed, render=render, max_steps=max_steps)

    obs, _info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        try:
            policy.reset(env.unwrapped if hasattr(env, "unwrapped") else env)
        except TypeError:
            policy.reset()

    episode_returns = None
    infos = []
    global_episode_return = 0.0
    start = time.time()
    # Note: we break on done_all, which is a heuristic for "episode over" that may not be perfectly aligned with env-specific episode termination conditions. For example, some envs may have time limits or other conditions that end the episode even if not all agents are done. In practice, this should be sufficient for most cases, but if needed, this can be made more robust by also checking env-specific info flags or conditions.
    for _ in range(max_steps):
        if getattr(policy, "uses_env", False):
            action = policy.act(env.unwrapped if hasattr(env, "unwrapped") else env)
        else:
            action = policy.act(obs)
        step_out = env.step(action)

        if isinstance(step_out, Transition):
            obs = step_out.obs
            reward_team = step_out.reward_team
            reward_by_agent = step_out.reward_by_agent
            done_all = step_out.done_all
            info = step_out.info
        else:
            obs, reward, terminated, truncated, info = step_out
            reward_by_agent = [float(x) for x in _as_seq(reward)]
            reward_team = float(np.sum(reward_by_agent))
            done_all = all(bool(x) for x in _as_seq(terminated)) or all(
                bool(x) for x in _as_seq(truncated)
            )

        if episode_returns is None:
            episode_returns = np.zeros(len(reward_by_agent), dtype=np.float64)
        episode_returns += np.array(reward_by_agent, dtype=np.float64)
        global_episode_return += reward_team
        infos.append(info)

        if render:
            env.render()

        if done_all:
            break

    duration_s = max(time.time() - start, 1e-9)
    metrics = summarize_episode(
        infos=infos,
        global_episode_return=global_episode_return,
        episode_returns=episode_returns if episode_returns is not None else [],
    )
    metrics["fps"] = float(metrics["episode_length"] / duration_s) if metrics["episode_length"] else 0.0
    return metrics
