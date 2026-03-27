from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from tarware_ext.graphs.builder_assignment_v1 import AssignmentGraphBuilder
from tarware_ext.graphs.gnn_minimal import GraphBatch, GnnAssignmentModel, masked_softmax


@pytest.mark.parametrize("architecture", ["sage", "gcn", "gat"])
def test_gnn_minimal_forward_shapes(architecture: str) -> None:
    env = gym.make("tarware-small-2agvs-1pickers-globalobs-v1", disable_env_checker=True)
    try:
        env.reset(seed=21)
        unwrapped = env.unwrapped
        builder = AssignmentGraphBuilder()
        graph = builder.build(unwrapped, controller=None)

        batch = GraphBatch.from_graph_state(graph)
        model = GnnAssignmentModel(
            node_in_dim=graph.node_features.shape[1],
            emb_dim=32,
            edge_dim=2,
            architecture=architecture,
        )

        agv_emb, task_emb, logits, probs = model(batch)

        expected_a = int(len(graph.metadata.get("agv_agent_indices", [])))
        expected_t = int(len(graph.task_node_ids))

        assert agv_emb.shape == (expected_a, 32)
        assert task_emb.shape == (expected_t, 32)
        assert logits.shape == (expected_a, expected_t)
        assert probs.shape == (expected_a, expected_t)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(probs).all()
        if expected_t > 0:
            assert torch.allclose(probs.sum(dim=-1), torch.ones((expected_a,), dtype=probs.dtype), atol=1e-5)
    finally:
        env.close()


def test_masked_softmax_relaxes_fully_masked_rows() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False, True], [False, False, False]], dtype=torch.bool)

    probs = masked_softmax(logits, mask, dim=-1)

    assert probs.shape == logits.shape
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones((2,), dtype=probs.dtype), atol=1e-6)
