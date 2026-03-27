# ============================================================
# FILE: tarware_ext/sb3/graph_obs_encoder.py
# ============================================================
"""
MVP graph -> fixed vector encoder (feature engineering).

SB3 (PPO) expects fixed-size observations (gym.spaces.Box).
GraphState is variable-sized, so we encode:

Per AGV:
  [agv_y, agv_x, busy, carrying]
  For each of K candidate tasks:
    [task_y, task_x, manhattan_dist, valid_flag]

Global:
  [num_tasks, num_free_agvs]

This is intentionally simple for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from tarware_ext.graphs.schema import GraphState, NodeType


@dataclass(frozen=True)
class GraphObsSpec:
    num_agvs: int
    top_k: int

    agv_feat_dim: int = 4
    task_feat_dim: int = 4
    global_feat_dim: int = 2

    @property
    def obs_dim(self) -> int:
        return self.num_agvs * (self.agv_feat_dim + self.top_k * self.task_feat_dim) + self.global_feat_dim


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def encode_graph_obs(g: GraphState, spec: GraphObsSpec) -> np.ndarray:
    """
    Encode GraphState into a fixed-size float32 vector.

    Assumes builder uses node_features columns: [y, x, busy, carrying, ...]
    """
    obs = np.zeros((spec.obs_dim,), dtype=np.float32)

    # Collect AGV nodes in the order of g.agent_node_ids
    agv_node_ids: List[int] = [int(nid) for nid in g.agent_node_ids if g.node_types[int(nid)] == NodeType.AGV]
    agv_node_ids = agv_node_ids[: spec.num_agvs]
    while len(agv_node_ids) < spec.num_agvs:
        agv_node_ids.append(agv_node_ids[-1] if agv_node_ids else 0)

    top_k_candidates = None
    if g.metadata:
        top_k_candidates = g.metadata.get("top_k_candidates")

    num_tasks = len(g.task_node_ids)
    offset = 0

    for i in range(spec.num_agvs):
        agv_nid = agv_node_ids[i]

        y = float(g.node_features[agv_nid, 0]) if g.node_features.size else 0.0
        x = float(g.node_features[agv_nid, 1]) if g.node_features.size else 0.0
        busy = float(g.node_features[agv_nid, 2]) if g.node_features.shape[1] > 2 else 0.0
        carrying = float(g.node_features[agv_nid, 3]) if g.node_features.shape[1] > 3 else 0.0

        obs[offset : offset + spec.agv_feat_dim] = np.array([y, x, busy, carrying], dtype=np.float32)
        offset += spec.agv_feat_dim

        cand_tasks: Sequence[int] = []
        if top_k_candidates is not None and i < len(top_k_candidates):
            cand_tasks = top_k_candidates[i]
        cand_tasks = list(cand_tasks)[: spec.top_k]
        while len(cand_tasks) < spec.top_k:
            cand_tasks.append(-1)

        agv_yx = (int(y), int(x))

        for c in cand_tasks:
            if c is None or int(c) < 0 or int(c) >= num_tasks:
                obs[offset : offset + spec.task_feat_dim] = 0.0
                offset += spec.task_feat_dim
                continue

            task_node_id = int(g.task_node_ids[int(c)])
            ty = float(g.node_features[task_node_id, 0])
            tx = float(g.node_features[task_node_id, 1])
            dist = float(_manhattan(agv_yx, (int(ty), int(tx))))
            valid = 1.0  # MVP: si está en candidates lo consideramos válido

            obs[offset : offset + spec.task_feat_dim] = np.array([ty, tx, dist, valid], dtype=np.float32)
            offset += spec.task_feat_dim

    # Global summary
    num_free_agvs = 0
    for nid in agv_node_ids:
        busy = float(g.node_features[nid, 2]) if g.node_features.shape[1] > 2 else 0.0
        if busy <= 0.0:
            num_free_agvs += 1

    obs[offset : offset + spec.global_feat_dim] = np.array(
        [float(num_tasks), float(num_free_agvs)], dtype=np.float32
    )
    return obs