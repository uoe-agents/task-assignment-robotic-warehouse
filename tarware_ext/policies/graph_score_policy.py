"""Dummy graph-based policy: scores agent->task pairs and assigns.

This policy is a lightweight bridge to validate the GraphBuilder -> Policy
integration. It does not learn: it scores pairs (by negative manhattan
distance) and produces a global assignment. The implementation favours a
Hungarian solver when SciPy is available and falls back to a greedy solver
otherwise.

The class mirrors the minimal policy contract used by the runner:
- `reset(self, env)`
- `act(self, env) -> List[int]` (returns macro-action `loc_id` per agent in
  the same order as `env.agents`).

This file is intentionally small and well-documented to be a drop-in for
experimentation before implementing a trainable GNN policy.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from tarware.definitions import AgentType
from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.graphs.schema import GraphState, NodeType

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class GraphScorePolicy:
    """Non-learning graph policy for prototyping.

    Parameters
    - builder: GraphBuilderV0 instance (or compatible) used to create GraphState
    - assigner: 'hungarian' or 'greedy'
    - distance_mode: forwarded to builder (placeholder)
    - top_k: optional number of candidates per agent (unused in builder_v0)
    """

    uses_env = True

    def __init__(
        self,
        builder: Optional[GraphBuilderV0] = None,
        assigner: str = "hungarian",
        distance_mode: str = "manhattan",
        top_k: Optional[int] = 2,
    ) -> None:
        self.builder = builder or GraphBuilderV0(distance_mode=distance_mode, top_k=top_k)
        self.assigner = assigner
        self._initialized = False
        self._agents = []

    def reset(self, env: Any) -> None:
        self._agents = list(env.agents)
        self._initialized = True

    def _greedy_assign(self, scores: np.ndarray, valid_mask: Optional[np.ndarray]) -> List[int]:
        # scores: (n_agvs, n_tasks) high->good
        n_agvs, n_tasks = scores.shape
        assigned_task = [-1] * n_agvs
        agent_free = [True] * n_agvs
        task_free = [True] * n_tasks

        # Flatten and sort pairs by score desc
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

    def _hungarian_assign(self, scores: np.ndarray, valid_mask: Optional[np.ndarray]) -> List[int]:
        # Convert to cost matrix for Hungarian (minimize)
        # Invalid pairs get very large cost
        n_agvs, n_tasks = scores.shape
        cost = np.full((n_agvs, n_tasks), 1e6, dtype=float)
        for i in range(n_agvs):
            for j in range(n_tasks):
                if valid_mask is not None and not valid_mask[i, j]:
                    continue
                cost[i, j] = -float(scores[i, j])  # maximize score -> minimize negative

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = [-1] * n_agvs
        for r, c in zip(row_ind, col_ind):
            # Only accept assignments with finite cost
            if cost[r, c] < 1e5:
                assigned[r] = int(c)
        return assigned

    def act(self, env: Any) -> List[int]:
        if not self._initialized:
            self.reset(env)

        g: GraphState = self.builder.build(env)

        n_agents = len(g.agent_node_ids)
        n_tasks = len(g.task_node_ids)

        # Quick exit when no tasks
        if n_tasks == 0:
            return [0 for _ in env.agents]

        # Identify AGV agent indices in the agent list (we only assign AGVs)
        agv_indices = [i for i, nid in enumerate(g.agent_node_ids) if g.node_types[nid] == NodeType.AGV]

        # Retrieve any precomputed distances and candidate lists from the builder
        scoring_distances = None
        if g.metadata is not None:
            scoring_distances = g.metadata.get("scoring_distances")
            top_k_candidates = g.metadata.get("top_k_candidates")
        else:
            top_k_candidates = None

        # Build score matrix for AGVs x tasks. Prefer builder-provided scoring
        # distances (which may contain path lengths for top-k candidates).
        scores = np.zeros((len(agv_indices), n_tasks), dtype=float)
        if scoring_distances is not None:
            for ai_idx, agent_node_idx in enumerate(agv_indices):
                for tj in range(n_tasks):
                    d = float(scoring_distances[agent_node_idx, tj])
                    if not np.isfinite(d):
                        # large negative score for unreachable/uncomputed
                        scores[ai_idx, tj] = -1e6
                    else:
                        scores[ai_idx, tj] = -d
        else:
            # Fallback: compute negative Manhattan from node_features
            for ai_idx, agent_node_idx in enumerate(agv_indices):
                ay, ax = g.node_features[agent_node_idx, 0], g.node_features[agent_node_idx, 1]
                for tj, task_node_idx in enumerate(g.task_node_ids):
                    ty, tx = g.node_features[task_node_idx, 0], g.node_features[task_node_idx, 1]
                    scores[ai_idx, tj] = - (abs(float(ay) - float(ty)) + abs(float(ax) - float(tx)))

        # Build valid mask combining env-provided action_mask and top-k candidate restriction
        valid_mask = None
        if g.action_mask is not None:
            valid_mask = g.action_mask[agv_indices, :]
        else:
            valid_mask = np.ones((len(agv_indices), n_tasks), dtype=bool)

        if top_k_candidates is not None:
            candidate_mask = np.zeros((n_agents, n_tasks), dtype=bool)
            for i in range(n_agents):
                for j in top_k_candidates[i]:
                    candidate_mask[i, j] = True
            candidate_mask_agvs = candidate_mask[agv_indices, :]
            valid_mask = np.logical_and(valid_mask, candidate_mask_agvs)

        # If some agents have no valid candidates after masking, relax to allow all
        for i_row in range(valid_mask.shape[0]):
            if not valid_mask[i_row].any():
                valid_mask[i_row, :] = True

        # Assignment
        if self.assigner == "hungarian" and HAS_SCIPY:
            assigned_task_for_agv = self._hungarian_assign(scores, valid_mask)
        else:
            assigned_task_for_agv = self._greedy_assign(scores, valid_mask)

        # Map assigned tasks (agv-level) back to loc_id and build final action list
        # Initialize all actions with NOOP (0)
        actions = [0 for _ in env.agents]

        # Map agv index in g.agent_node_ids -> env agent index (same order was used by builder)
        # builder uses env.agents order for agent nodes
        for agv_pos, agv_node_idx in enumerate(g.agent_node_ids):
            # check if this node is AGV type
            if g.node_types[agv_node_idx] != NodeType.AGV:
                continue
            # find its position in the agv_indices list
            try:
                pos_in_agv_list = agv_indices.index(agv_pos)
            except ValueError:
                continue
            task_idx = assigned_task_for_agv[pos_in_agv_list]
            if task_idx is None or task_idx < 0:
                continue
            loc_id = g.task_loc_ids[int(task_idx)]
            # Map to correct env agent ordering: builder used env.agents order -> same index
            actions[agv_pos] = int(loc_id)

        return actions
