import pytest
import gymnasium as gym
import sys
from pathlib import Path

# Ensure repo root is on sys.path so local packages (tarware_ext, tarware) can be imported
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from tarware_ext.envs import TarwareAdapter
from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.policies import GraphScorePolicy
from scripts.train_graph_rl import _train_torch_from_samples


def test_torch_gnn_training_smoke(tmp_path):
    env_id = "tarware-small-2agvs-1pickers-globalobs-v1"
    env = TarwareAdapter(gym.make(env_id))
    underlying = getattr(env, "env", None)
    if underlying is not None and hasattr(underlying, "unwrapped"):
        target_env = underlying.unwrapped
    else:
        target_env = env

    builder = GraphBuilderV0()
    policy = GraphScorePolicy(distance_mode="manhattan", assigner="greedy")
    # Ensure policy initialized
    policy.reset(target_env)

    g = builder.build(target_env)
    actions = policy.act(target_env)

    # Convert actions loc_id -> teacher task indices
    teacher_indices = []
    for a in actions:
        if int(a) == 0:
            teacher_indices.append(-1)
            continue
        try:
            task_idx = g.task_loc_ids.index(int(a))
        except ValueError:
            task_idx = -1
        teacher_indices.append(int(task_idx))

    sample = {
        "node_features": g.node_features.copy(),
        "edge_index": g.edge_index.copy(),
        "agent_node_ids": list(g.agent_node_ids),
        "task_node_ids": list(g.task_node_ids),
        "task_loc_ids": list(g.task_loc_ids),
        "action_mask": None if g.action_mask is None else g.action_mask.copy(),
        "metadata": dict(g.metadata) if g.metadata is not None else {},
        "teacher_task_indices": list(teacher_indices),
    }

    samples = [sample] * 3
    out_model = str(tmp_path / "torch_gnn_smoke.pth")

    # Should not raise and should save a model file
    _train_torch_from_samples(samples, out_model, epochs=1, lr=1e-3, device="cpu")
    assert Path(out_model).exists()
