import gymnasium as gym
import sys
from pathlib import Path

# Ensure repo root is on sys.path so local packages (tarware_ext, tarware) can be imported
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tarware_ext.envs import TarwareAdapter
from tarware_ext.graphs.builder_v0 import GraphBuilderV0


def test_builder_shapes_and_mappings():
    env_id = "tarware-small-2agvs-1pickers-globalobs-v1"
    env = TarwareAdapter(gym.make(env_id))
    builder = GraphBuilderV0()
    # Builder expects access to the underlying Warehouse object. When the
    # gym env is wrapped (OrderEnforcing etc.) the concrete attributes live
    # on `env.env.unwrapped`. Use that here to be robust in test environments.
    underlying = getattr(env, "env", None)
    if underlying is not None and hasattr(underlying, "unwrapped"):
        target_env = underlying.unwrapped
    else:
        target_env = env
    g = builder.build(target_env)

    # Basic shape checks
    # Use the unwrapped env for introspection (wrappers may hide attributes)
    n_agents = len(target_env.agents)
    n_tasks = len(g.task_node_ids)
    assert g.node_features.shape[0] == n_agents + n_tasks
    assert len(g.task_loc_ids) == n_tasks

    # If action_mask is provided, check its shape
    if g.action_mask is not None:
        assert g.action_mask.shape == (n_agents, n_tasks)

    # task_loc_ids must be present in env.action_id_to_coords_map
    for lid in g.task_loc_ids:
        assert int(lid) in env.action_id_to_coords_map
