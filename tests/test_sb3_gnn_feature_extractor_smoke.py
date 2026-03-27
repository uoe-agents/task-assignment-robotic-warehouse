from __future__ import annotations

import torch
import pytest

from tarware_ext.sb3 import GraphAssignmentConfig, GraphAssignmentEnv
from tarware_ext.sb3.gnn_feature_extractor import GnnFeatureExtractor


@pytest.mark.parametrize("architecture", ["sage", "gcn", "gat"])
def test_gnn_feature_extractor_smoke_graph_dict_obs(architecture: str) -> None:
    env = GraphAssignmentEnv(
        GraphAssignmentConfig(
            env_id="tarware-small-2agvs-1pickers-globalobs-v1",
            obs_backend="graph_dict",
            max_request_slots=20,
            max_steps=20,
            seed=21,
            verbose=False,
        )
    )
    try:
        obs, _ = env.reset(seed=21)
        assert isinstance(obs, dict)

        extractor = GnnFeatureExtractor(
            env.observation_space,
            emb_dim=32,
            gnn_layers=2,
            dropout=0.0,
            architecture=architecture,
        )

        batch_obs = {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in obs.items()
        }

        z = extractor(batch_obs)
        assert z.shape == (1, extractor.features_dim)
        assert torch.isfinite(z).all()
    finally:
        env.close()
