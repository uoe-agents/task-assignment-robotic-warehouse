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
from tarware_ext.policies.torch_gnn_policy import TorchGNNPolicy


def test_torch_gnn_forward_act():
    env_id = "tarware-small-2agvs-1pickers-globalobs-v1"
    env = TarwareAdapter(gym.make(env_id))
    underlying = getattr(env, "env", None)
    if underlying is not None and hasattr(underlying, "unwrapped"):
        target_env = underlying.unwrapped
    else:
        target_env = env

    builder = GraphBuilderV0()
    g = builder.build(target_env)

    policy = TorchGNNPolicy(builder=builder, device="cpu")
    policy.reset(target_env)
    actions = policy.act(target_env)
    assert isinstance(actions, list)
    assert len(actions) == len(target_env.agents)
    for a in actions:
        assert isinstance(a, int)
