from __future__ import annotations

import gymnasium as gym
import numpy as np

import tarware  # noqa: F401
from tarware.heuristic import heuristic_episode

from tarware_ext.controllers import HeuristicController
from tarware_ext.runners.metrics import summarize_episode


def _rollout_with_controller(env, seed: int):
    _ = env.reset(seed=seed)
    controller = HeuristicController()
    controller.reset(env, seed=seed)

    done = False
    infos = []
    global_episode_return = 0.0
    episode_returns = np.zeros(env.unwrapped.num_agents, dtype=np.float64)

    while not done:
        actions = controller.step(env, rl_agv_assignments=None)
        _obs, reward, terminated, truncated, info = env.step(actions)

        episode_returns += np.array(reward, dtype=np.float64)
        global_episode_return += float(np.sum(reward))
        infos.append(info)

        done_flags = np.array(terminated, dtype=bool) | np.array(truncated, dtype=bool)
        done = bool(np.all(done_flags))

    return infos, global_episode_return, episode_returns


def test_heuristic_controller_matches_heuristic_baseline_metrics() -> None:
    env_id = "tarware-small-2agvs-1pickers-globalobs-v1"
    seed = 21

    env_ref = gym.make(env_id, disable_env_checker=True)
    env_ctl = gym.make(env_id, disable_env_checker=True)
    try:
        infos_ref, ret_ref, ep_ret_ref = heuristic_episode(env_ref.unwrapped, render=False, seed=seed)
        infos_ctl, ret_ctl, ep_ret_ctl = _rollout_with_controller(env_ctl, seed=seed)

        m_ref = summarize_episode(infos_ref, ret_ref, ep_ret_ref)
        m_ctl = summarize_episode(infos_ctl, ret_ctl, ep_ret_ctl)

        assert int(m_ctl["episode_length"]) == int(m_ref["episode_length"])
        assert float(m_ctl["shelf_deliveries"]) == float(m_ref["shelf_deliveries"])
        assert float(m_ctl["clashes"]) == float(m_ref["clashes"])
        assert float(m_ctl["stucks"]) == float(m_ref["stucks"])
        assert np.isclose(
            float(m_ctl["global_episode_return"]),
            float(m_ref["global_episode_return"]),
            atol=1e-12,
        )
    finally:
        env_ref.close()
        env_ctl.close()
