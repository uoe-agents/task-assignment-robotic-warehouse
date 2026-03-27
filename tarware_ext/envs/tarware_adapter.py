"""Adapter that normalizes reset/step outputs for multi-agent envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np


def _as_seq(x: Any) -> Sequence:
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    return [x]


def _infer_num_agents(env: Any, fallback: int | None = None) -> int:
    for attr in ("num_agents",):
        if hasattr(env, attr):
            return int(getattr(env, attr))
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "num_agents"):
        return int(env.unwrapped.num_agents)
    if fallback is not None:
        return int(fallback)
    return 1


def _as_list(x: Any, count: int) -> list:
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x for _ in range(count)]


@dataclass
class Transition:
    obs: Any
    reward_by_agent: list[float]
    reward_team: float
    terminated_by_agent: list[bool]
    truncated_by_agent: list[bool]
    done_by_agent: list[bool]
    done_all: bool
    info: dict


class TarwareAdapter:
    def __init__(self, env: Any, reward_team: bool = False, done_all: bool = True) -> None:
        self.env = env
        self.reward_team = reward_team
        self.done_all = done_all

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        if options is None:
            reset_out = self.env.reset(seed=seed)
        else:
            try:
                reset_out = self.env.reset(seed=seed, options=options)
            except TypeError:
                reset_out = self.env.reset(seed=seed)

        if isinstance(reset_out, tuple):
            if len(reset_out) >= 2:
                return reset_out[0], reset_out[1]
            return reset_out[0], {}
        return reset_out, {}

    def step(self, action: Any) -> Transition:
        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step return length: {len(step_out)}")

        num_agents = _infer_num_agents(self.env, fallback=len(_as_seq(reward)))
        reward_by_agent = [float(x) for x in _as_list(reward, num_agents)]
        terminated_by_agent = [bool(x) for x in _as_list(terminated, num_agents)]
        truncated_by_agent = [bool(x) for x in _as_list(truncated, num_agents)]
        done_by_agent = [t or tr for t, tr in zip(terminated_by_agent, truncated_by_agent)]

        reward_team = float(sum(reward_by_agent))
        done_all = all(done_by_agent)

        if self.reward_team:
            reward_by_agent = [reward_team for _ in range(num_agents)]

        if self.done_all:
            terminated_by_agent = [done_all for _ in range(num_agents)]
            truncated_by_agent = [done_all for _ in range(num_agents)]
            done_by_agent = [done_all for _ in range(num_agents)]

        return Transition(
            obs=obs,
            reward_by_agent=reward_by_agent,
            reward_team=reward_team,
            terminated_by_agent=terminated_by_agent,
            truncated_by_agent=truncated_by_agent,
            done_by_agent=done_by_agent,
            done_all=done_all,
            info=info,
        )

    def render(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.env.render(*args, **kwargs)
        except TypeError:
            # Some Gym wrappers (e.g. OrderEnforcing) have a render() that
            # doesn't accept keyword arguments like `mode`. Attempt to find an
            # underlying env that supports the requested signature and call
            # its render. Fall back to calling render without kwargs.
            target = self.env
            for _ in range(6):
                try:
                    if hasattr(target, "unwrapped") and getattr(target, "unwrapped") is not target:
                        target = getattr(target, "unwrapped")
                        continue
                except Exception:
                    pass
                try:
                    if hasattr(target, "env") and getattr(target, "env") is not target:
                        target = getattr(target, "env")
                        continue
                except Exception:
                    pass
                break

            # Prefer calling render on the unwrapped target with kwargs
            try:
                return target.render(*args, **kwargs)
            except TypeError:
                try:
                    return target.render(*args)
                except Exception:
                    # Last resort: call wrapper's render without kwargs
                    return self.env.render(*args)

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
