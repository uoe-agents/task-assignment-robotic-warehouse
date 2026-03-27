from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tarware_ext.graphs.builder_assignment_v1 import AssignmentGraphBuilder
from tarware_ext.sb3.graph_assignment_obs_encoder import encode_graph_assignment_obs

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def test_encode_graph_assignment_obs_manual_shape_and_finite() -> None:
    env = gym.make("tarware-small-2agvs-1pickers-globalobs-v1", disable_env_checker=True)
    try:
        env.reset(seed=21)
        graph = AssignmentGraphBuilder().build(env.unwrapped, controller=None)

        obs_shape = (2 * (6 + 20 * 7) + 4,)
        obs = encode_graph_assignment_obs(
            graph,
            max_request_slots=20,
            num_agvs=2,
            obs_shape=obs_shape,
            agv_feat_dim=6,
            slot_feat_dim=7,
            global_feat_dim=4,
            encoder_mode="manual",
        )

        assert obs.shape == obs_shape
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
    finally:
        env.close()


@pytest.mark.skipif(not HAS_TORCH, reason="torch is required")
@pytest.mark.parametrize("gnn_arch", ["sage", "gcn", "gat"])
def test_encode_graph_assignment_obs_gnn_shape_and_finite(gnn_arch: str) -> None:
    env = gym.make("tarware-small-2agvs-1pickers-globalobs-v1", disable_env_checker=True)
    try:
        env.reset(seed=21)
        graph = AssignmentGraphBuilder().build(env.unwrapped, controller=None)

        obs_shape = (2 * (6 + 20 * 7) + 4,)
        obs = encode_graph_assignment_obs(
            graph,
            max_request_slots=20,
            num_agvs=2,
            obs_shape=obs_shape,
            agv_feat_dim=6,
            slot_feat_dim=7,
            global_feat_dim=4,
            encoder_mode="gnn",
            gnn_arch=gnn_arch,
        )

        assert obs.shape == obs_shape
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
    finally:
        env.close()
