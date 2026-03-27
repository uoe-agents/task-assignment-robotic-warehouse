from __future__ import annotations

import gymnasium as gym

import tarware  # noqa: F401

from tarware_ext.controllers import HeuristicController


def test_heuristic_controller_returns_valid_joint_action() -> None:
    env = gym.make("tarware-small-2agvs-1pickers-globalobs-v1", disable_env_checker=True)
    try:
        env.reset(seed=21)
        controller = HeuristicController()
        controller.reset(env, seed=21)

        actions = controller.step(env, rl_agv_assignments=[1, 0])
        assert len(actions) == env.unwrapped.num_agents

        step_out = env.step(actions)
        assert isinstance(step_out, tuple)
        assert len(step_out) in (4, 5)
    finally:
        env.close()
