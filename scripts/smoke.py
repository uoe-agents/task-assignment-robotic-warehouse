"""Smoke test runner for TA-RWARE."""

from __future__ import annotations

import argparse
import time
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

import tarware  # noqa: F401


def _brief(x: Any) -> str:
    if isinstance(x, np.ndarray):
        return f"ndarray shape={x.shape} dtype={x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__} len={len(x)}"
    return f"{type(x).__name__}"


def _as_seq(x: Any) -> Sequence:
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    return [x]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    if args.render:
        try:
            env = gym.make(args.env_id, render_mode="human")
        except TypeError:
            env = gym.make(args.env_id)
    else:
        env = gym.make(args.env_id)

    print("ENV:", args.env_id)
    print("action_space:", env.action_space)

    reset_out = env.reset(seed=args.seed)
    if isinstance(reset_out, tuple):
        if len(reset_out) >= 2:
            obs, info = reset_out[0], reset_out[1]
        else:
            obs, info = reset_out[0], {}
    else:
        obs, info = reset_out, {}
    print("\nRESET:")
    print("obs:", _brief(obs))
    print("info keys:", list(info.keys()) if isinstance(info, dict) else type(info))

    obs_seq = _as_seq(obs)
    print("num_agents (from obs):", len(obs_seq))
    if len(obs_seq) > 0:
        print("obs[0]:", _brief(obs_seq[0]))

    print("\nSTEP LOOP:")
    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if t < 3 or t == args.steps - 1:
            r = np.sum(_as_seq(reward))
            term_all = all(bool(x) for x in _as_seq(terminated))
            trunc_all = all(bool(x) for x in _as_seq(truncated))
            print(
                f"t={t:03d} action={action} reward(sum)={r:.3f} term_all={term_all} trunc_all={trunc_all}"
            )

        if args.render:
            env.render()
            time.sleep(1.0 / 30.0)

        term_all = all(bool(x) for x in _as_seq(terminated))
        trunc_all = all(bool(x) for x in _as_seq(truncated))
        if term_all or trunc_all:
            print(f"Episode ended at t={t} (term_all={term_all}, trunc_all={trunc_all})")
            break

    env.close()
    if args.render:
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
