# ============================================================
# FILE: tarware_ext/sb3/assignment_obs_encoder.py
# ============================================================
"""Observation encoder for GraphAssignmentEnv.

This module keeps the assignment observation logic separate from the Gym env
wrapper so it can evolve independently.
"""

from __future__ import annotations

from typing import Any, Dict, Set, Tuple

import numpy as np

from tarware.definitions import AgentType


def encode_assignment_obs(
    env: Any,
    *,
    controller: Any = None,
    max_request_slots: int,
    num_agvs: int,
    obs_shape: Tuple[int, ...],
    agv_feat_dim: int = 6,
    slot_feat_dim: int = 7,
    global_feat_dim: int = 4,
) -> np.ndarray:
    """Encode explicit request slots so action indices and observation slots align."""
    agents = list(getattr(env, "agents", []))
    agvs = [a for a in agents if getattr(a, "type", None) == AgentType.AGV]
    request_queue = list(getattr(env, "request_queue", []))

    snapshot: Dict[str, Any] = {}
    if controller is not None and hasattr(controller, "get_assignment_snapshot"):
        try:
            snap = controller.get_assignment_snapshot()
            if isinstance(snap, dict):
                snapshot = snap
        except Exception:
            snapshot = {}

    agv_is_assigned = snapshot.get("agv_is_assigned", {})
    agv_mission_type = snapshot.get("agv_mission_type", {})
    assigned_item_ids: Set[int] = set(int(x) for x in snapshot.get("assigned_item_ids", []))

    agv_positions = [(int(getattr(agv, "y", 0)), int(getattr(agv, "x", 0))) for agv in agvs]

    obs = np.zeros(obs_shape, dtype=np.float32)
    offset = 0

    for i in range(num_agvs):
        if i < len(agvs):
            agv = agvs[i]
            y = float(getattr(agv, "y", 0.0))
            x = float(getattr(agv, "x", 0.0))
            busy = 1.0 if bool(getattr(agv, "busy", False)) else 0.0
            carrying = 1.0 if bool(getattr(agv, "carrying_shelf", None)) else 0.0
            assigned = float(agv_is_assigned.get(id(agv), 0.0))
            mission_type = float(agv_mission_type.get(id(agv), 0.0))
        else:
            y = 0.0
            x = 0.0
            busy = 0.0
            carrying = 0.0
            assigned = 0.0
            mission_type = 0.0

        agv_values = [y, x, busy, carrying, assigned, mission_type]
        agv_feats = np.zeros((agv_feat_dim,), dtype=np.float32)
        agv_feats[: min(agv_feat_dim, len(agv_values))] = np.array(agv_values[:agv_feat_dim], dtype=np.float32)
        obs[offset : offset + agv_feat_dim] = agv_feats
        offset += agv_feat_dim

        agv_yx = (int(y), int(x))
        for slot in range(max_request_slots):
            if slot < len(request_queue):
                item = request_queue[slot]
                ty = float(getattr(item, "y", 0.0))
                tx = float(getattr(item, "x", 0.0))
                dist = float(abs(agv_yx[0] - int(ty)) + abs(agv_yx[1] - int(tx)))
                other_dists = [
                    abs(y_other - int(ty)) + abs(x_other - int(tx))
                    for agv_idx, (y_other, x_other) in enumerate(agv_positions)
                    if agv_idx != i
                ]
                min_other_dist = float(min(other_dists)) if other_dists else dist
                num_other_closer = float(sum(1 for d in other_dists if d < dist))
                item_id = int(getattr(item, "id", -1))
                assigned_slot = 1.0 if item_id in assigned_item_ids else 0.0
                valid = 1.0

                slot_values = [
                    ty,
                    tx,
                    dist,
                    min_other_dist,
                    num_other_closer,
                    valid,
                    assigned_slot,
                ]
                slot_feats = np.zeros((slot_feat_dim,), dtype=np.float32)
                slot_feats[: min(slot_feat_dim, len(slot_values))] = np.array(
                    slot_values[:slot_feat_dim], dtype=np.float32
                )
                obs[offset : offset + slot_feat_dim] = slot_feats
            else:
                obs[offset : offset + slot_feat_dim] = 0.0
            offset += slot_feat_dim

    num_tasks = float(len(request_queue))
    num_free_agvs = float(sum(1 for agv in agvs if not bool(getattr(agv, "busy", False))))
    num_busy_agvs = float(max(0, len(agvs) - int(num_free_agvs)))
    num_assigned_requests = float(len(assigned_item_ids))
    global_values = [num_tasks, num_free_agvs, num_busy_agvs, num_assigned_requests]
    global_feats = np.zeros((global_feat_dim,), dtype=np.float32)
    global_feats[: min(global_feat_dim, len(global_values))] = np.array(
        global_values[:global_feat_dim], dtype=np.float32
    )
    obs[offset : offset + global_feat_dim] = global_feats
    return obs
