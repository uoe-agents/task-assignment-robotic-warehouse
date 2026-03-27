"""Projection helpers from assignment GraphState to slot-aligned tensors."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .schema import GraphState


def project_graph_to_request_slots(
    graph: GraphState,
    *,
    max_request_slots: int,
    num_agvs: int,
) -> Dict[str, np.ndarray]:
    """Project assignment graph to fixed-size AGV/slot/global feature blocks."""
    metadata: Dict[str, Any] = graph.metadata if isinstance(graph.metadata, dict) else {}

    agv_agent_indices = list(metadata.get("agv_agent_indices", []))
    slot_to_node = metadata.get("request_slot_to_node_id", {})

    agv_is_assigned_values = list(metadata.get("agv_is_assigned_values", []))
    agv_mission_type_values = list(metadata.get("agv_mission_type_values", []))

    task_node_ids = list(graph.task_node_ids)
    num_tasks = len(task_node_ids)

    agv_feats = np.zeros((num_agvs, 6), dtype=np.float32)
    slot_feats = np.zeros((num_agvs, max_request_slots, 7), dtype=np.float32)

    # AGV features
    for i in range(num_agvs):
        if i >= len(agv_agent_indices):
            continue
        agv_node_idx = int(agv_agent_indices[i])
        y = float(graph.node_features[agv_node_idx, 0])
        x = float(graph.node_features[agv_node_idx, 1])
        busy = float(graph.node_features[agv_node_idx, 2])
        carrying = float(graph.node_features[agv_node_idx, 3])
        assigned = float(agv_is_assigned_values[i]) if i < len(agv_is_assigned_values) else 0.0
        mission_type = float(agv_mission_type_values[i]) if i < len(agv_mission_type_values) else 0.0
        agv_feats[i, :] = np.array([y, x, busy, carrying, assigned, mission_type], dtype=np.float32)

    # Slot features (per AGV, per request slot)
    agv_positions = [(int(agv_feats[i, 0]), int(agv_feats[i, 1])) for i in range(num_agvs)]
    for i in range(num_agvs):
        ay, ax = agv_positions[i]
        for slot in range(max_request_slots):
            if slot >= num_tasks:
                continue

            task_node_id = int(slot_to_node.get(slot, task_node_ids[slot]))
            ty = float(graph.node_features[task_node_id, 0])
            tx = float(graph.node_features[task_node_id, 1])
            valid = float(graph.node_features[task_node_id, 3])
            assigned_slot = float(graph.node_features[task_node_id, 4])

            dist = float(abs(ay - int(ty)) + abs(ax - int(tx)))
            other_dists = [
                abs(other_y - int(ty)) + abs(other_x - int(tx))
                for agv_idx, (other_y, other_x) in enumerate(agv_positions)
                if agv_idx != i
            ]
            min_other_dist = float(min(other_dists)) if other_dists else dist
            num_other_closer = float(sum(1 for d in other_dists if d < dist))

            slot_feats[i, slot, :] = np.array(
                [ty, tx, dist, min_other_dist, num_other_closer, valid, assigned_slot],
                dtype=np.float32,
            )

    # Global features
    free_mask = agv_feats[:, 2] <= 0.0
    num_free_agvs = float(np.sum(free_mask))
    num_busy_agvs = float(max(0, num_agvs - int(num_free_agvs)))
    assigned_requests = 0.0
    if num_tasks > 0:
        # Same value for each AGV row, take AGV row 0.
        assigned_requests = float(np.sum(slot_feats[0, : min(num_tasks, max_request_slots), 6] > 0.0))

    global_feats = np.array(
        [float(num_tasks), num_free_agvs, num_busy_agvs, assigned_requests],
        dtype=np.float32,
    )

    return {
        "agv_features": agv_feats,
        "slot_features": slot_feats,
        "global_features": global_feats,
    }
