"""GNNPolicy scaffold.

This file provides a minimal, dependency-light scaffold for a graph-based
policy. The implementation below uses a simple numpy-based node encoder and
pairwise scoring function (dot-product of node embeddings) so it can be used
for quick prototyping without requiring `torch` or `torch_geometric`.

The class is intentionally pluggable: when you later add a PyTorch-based
GNN, keep the same `reset(self, env)` / `act(self, env)` interface so it can
drop in as a replacement for this scaffold.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from tarware.definitions import AgentType
from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.graphs.schema import GraphState, NodeType


class GNNPolicy:
    """Minimal graph policy scaffold.

    - Uses a tiny numpy encoder to produce node embeddings from
      `GraphState.node_features` and scores agent->task pairs by dot-product.
    - Respects `GraphState.action_mask` and `metadata['top_k_candidates']`
      if present.
    - Produces actions as a list of `loc_id` values (0 = NOOP) in the same
      order as `env.agents`.
    """

    uses_env = True

    def __init__(self, builder: Optional[GraphBuilderV0] = None, hidden_dim: int = 32, assigner: str = "greedy") -> None:
        self.builder = builder or GraphBuilderV0()
        self.hidden_dim = int(hidden_dim)
        self.assigner = assigner
        self._initialized = False
        # simple deterministic RNG for weight init so behaviour is stable
        self._rng = np.random.RandomState(0)
        # input feature dim is 4 in builder_v0's node_features (y, x, busy, carrying)
        self._in_dim = 4
        self.W = self._rng.normal(scale=0.1, size=(self._in_dim, self.hidden_dim))
        self.b = np.zeros((self.hidden_dim,), dtype=float)

    def reset(self, env: Any) -> None:
        self._initialized = True

    def _greedy_assign(self, scores: np.ndarray, valid_mask: Optional[np.ndarray]) -> List[int]:
        n_agvs, n_tasks = scores.shape
        assigned_task = [-1] * n_agvs
        agent_free = [True] * n_agvs
        task_free = [True] * n_tasks

        pairs = []
        for i in range(n_agvs):
            for j in range(n_tasks):
                if valid_mask is not None and not valid_mask[i, j]:
                    continue
                pairs.append((scores[i, j], i, j))
        pairs.sort(key=lambda x: -x[0])

        for score, i, j in pairs:
            if agent_free[i] and task_free[j]:
                assigned_task[i] = j
                agent_free[i] = False
                task_free[j] = False
        return assigned_task

    def act(self, env: Any) -> List[int]:
        if not self._initialized:
            self.reset(env)

        # Prefer the unwrapped env for builders
        target_env = env.unwrapped if hasattr(env, "unwrapped") else env
        g: GraphState = self.builder.build(target_env)

        n_agents = len(g.agent_node_ids)
        n_tasks = len(g.task_node_ids)

        if n_tasks == 0:
            return [0 for _ in range(len(g.agent_node_ids))]

        # Identify AGV agent indices
        agv_indices = [i for i, nid in enumerate(g.agent_node_ids) if g.node_types[nid] == NodeType.AGV]

        # Node embeddings: simple linear encoder
        nf = np.array(g.node_features, dtype=float)
        embeddings = nf.dot(self.W) + self.b  # (N_nodes, hidden_dim)

        # Build score matrix for AGVs x tasks
        scores = np.zeros((len(agv_indices), n_tasks), dtype=float)
        for ai_idx, agent_node_idx in enumerate(agv_indices):
            a_emb = embeddings[agent_node_idx]
            for tj, task_node_idx in enumerate(g.task_node_ids):
                t_emb = embeddings[task_node_idx]
                scores[ai_idx, tj] = float(a_emb.dot(t_emb))

        # Valid mask
        if g.action_mask is not None:
            valid_mask = g.action_mask[agv_indices, :]
        else:
            valid_mask = np.ones((len(agv_indices), n_tasks), dtype=bool)

        # Respect builder-provided top_k_candidates when present
        top_k = None
        if g.metadata is not None:
            top_k = g.metadata.get("top_k")
            top_k_candidates = g.metadata.get("top_k_candidates")
        else:
            top_k_candidates = None

        if top_k_candidates is not None:
            candidate_mask = np.zeros((n_agents, n_tasks), dtype=bool)
            for i in range(n_agents):
                for j in top_k_candidates[i]:
                    candidate_mask[i, j] = True
            candidate_mask_agvs = candidate_mask[agv_indices, :]
            valid_mask = np.logical_and(valid_mask, candidate_mask_agvs)

        # If an agent has no valid candidates, relax mask for that agent
        for i_row in range(valid_mask.shape[0]):
            if not valid_mask[i_row].any():
                valid_mask[i_row, :] = True

        # Assignment: greedy
        assigned_for_agvs = self._greedy_assign(scores, valid_mask)

        # Map assigned tasks (agv-level) back to loc_id and build final action list
        actions = [0 for _ in range(len(g.agent_node_ids))]
        for agv_pos, agv_node_idx in enumerate(g.agent_node_ids):
            if g.node_types[agv_node_idx] != NodeType.AGV:
                continue
            try:
                pos_in_agv_list = agv_indices.index(agv_pos)
            except ValueError:
                continue
            task_idx = assigned_for_agvs[pos_in_agv_list]
            if task_idx is None or task_idx < 0:
                continue
            loc_id = g.task_loc_ids[int(task_idx)]
            actions[agv_pos] = int(loc_id)

        return actions
"""GNN policy placeholder."""
