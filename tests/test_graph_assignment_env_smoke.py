from __future__ import annotations

import pytest

from tarware_ext.sb3 import GraphAssignmentConfig, GraphAssignmentEnv


@pytest.mark.parametrize("obs_backend", ["assignment", "graph", "graph_dict"])
def test_graph_assignment_env_reset_step_smoke(obs_backend: str) -> None:
    env = GraphAssignmentEnv(
        GraphAssignmentConfig(
            env_id="tarware-small-2agvs-1pickers-globalobs-v1",
            obs_backend=obs_backend,
            max_request_slots=20,
            max_steps=20,
            seed=21,
            verbose=False,
        )
    )
    try:
        obs, info = env.reset(seed=21)
        if obs_backend == "graph_dict":
            assert isinstance(obs, dict)
            assert set(obs.keys()) == {
                "node_features",
                "edge_index",
                "edge_attr",
                "action_mask",
                "n_nodes",
                "n_edges",
                "n_tasks",
            }
            assert obs["node_features"].dtype.name == "float32"
            assert obs["edge_index"].dtype.name == "int32"
            assert obs["edge_attr"].dtype.name == "float32"
            assert obs["action_mask"].dtype.name == "int8"
        else:
            assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        if obs_backend == "graph_dict":
            assert isinstance(next_obs, dict)
            assert next_obs["node_features"].shape == env.observation_space["node_features"].shape
        else:
            assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()
