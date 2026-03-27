"""GraphState-based observation encoder for GraphAssignmentEnv."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from tarware_ext.graphs.schema import GraphState
from tarware_ext.graphs.slot_projection import project_graph_to_request_slots

_GNN_MODEL_CACHE = {}


def _resolve_agv_node_ids(graph: GraphState) -> list[int]:
    metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    agv_ids = metadata.get("agv_agent_indices")
    if isinstance(agv_ids, list) and agv_ids:
        return [int(x) for x in agv_ids]

    out = []
    for node_id in graph.agent_node_ids:
        nid = int(node_id)
        if 0 <= nid < len(graph.node_types) and str(graph.node_types[nid]).lower().endswith("agv"):
            out.append(nid)
    return out


def _encode_graph_assignment_obs_manual(
    graph: GraphState,
    *,
    max_request_slots: int,
    num_agvs: int,
    obs_shape: Tuple[int, ...],
    agv_feat_dim: int,
    slot_feat_dim: int,
    global_feat_dim: int,
) -> np.ndarray:
    proj = project_graph_to_request_slots(
        graph,
        max_request_slots=max_request_slots,
        num_agvs=num_agvs,
    )
    agv_feats = proj["agv_features"]
    slot_feats = proj["slot_features"]
    global_feats = proj["global_features"]

    obs = np.zeros(obs_shape, dtype=np.float32)
    offset = 0
    for i in range(num_agvs):
        obs[offset : offset + agv_feat_dim] = agv_feats[i, :agv_feat_dim]
        offset += agv_feat_dim
        for slot in range(max_request_slots):
            obs[offset : offset + slot_feat_dim] = slot_feats[i, slot, :slot_feat_dim]
            offset += slot_feat_dim

    obs[offset : offset + global_feat_dim] = global_feats[:global_feat_dim]
    return obs


def _get_cached_gnn_model(node_feature_dim: int, gnn_arch: str):
    key = (int(node_feature_dim), str(gnn_arch).strip().lower())
    model = _GNN_MODEL_CACHE.get(key)
    if model is not None:
        return model

    from tarware_ext.graphs.gnn_minimal import build_default_gnn_for_assignment
    import torch

    torch.manual_seed(0)
    model = build_default_gnn_for_assignment(node_feature_dim=node_feature_dim, architecture=gnn_arch)
    model.eval()
    _GNN_MODEL_CACHE[key] = model
    return model


def _encode_graph_assignment_obs_gnn(
    graph: GraphState,
    *,
    max_request_slots: int,
    num_agvs: int,
    obs_shape: Tuple[int, ...],
    agv_feat_dim: int,
    slot_feat_dim: int,
    global_feat_dim: int,
    gnn_arch: str,
) -> np.ndarray:
    from tarware_ext.graphs.gnn_minimal import GraphBatch
    import torch

    model = _get_cached_gnn_model(node_feature_dim=int(graph.node_features.shape[1]), gnn_arch=gnn_arch)
    batch = GraphBatch.from_graph_state(graph)

    with torch.no_grad():
        agv_emb_t, task_emb_t, logits_t, probs_t = model(batch)

    agv_emb = agv_emb_t.detach().cpu().numpy()
    task_emb = task_emb_t.detach().cpu().numpy()
    logits = logits_t.detach().cpu().numpy()
    probs = probs_t.detach().cpu().numpy()

    if batch.agv_action_mask is not None:
        agv_action_mask = batch.agv_action_mask.detach().cpu().numpy().astype(np.float32)
    else:
        agv_action_mask = np.ones((agv_emb.shape[0], task_emb.shape[0]), dtype=np.float32)

    obs = np.zeros(obs_shape, dtype=np.float32)
    offset = 0

    agv_rows = min(num_agvs, agv_emb.shape[0])
    task_cols = min(max_request_slots, task_emb.shape[0])

    agv_max_prob = np.max(probs, axis=1) if probs.size else np.zeros((agv_emb.shape[0],), dtype=np.float32)
    agv_valid_ratio = (
        np.mean(agv_action_mask > 0.0, axis=1)
        if agv_action_mask.size
        else np.zeros((agv_emb.shape[0],), dtype=np.float32)
    )

    task_node_ids = list(graph.task_node_ids)
    task_assigned = np.zeros((task_cols,), dtype=np.float32)
    for j in range(task_cols):
        node_id = int(task_node_ids[j])
        if 0 <= node_id < graph.node_features.shape[0] and graph.node_features.shape[1] > 4:
            task_assigned[j] = float(graph.node_features[node_id, 4] > 0.5)

    for i in range(num_agvs):
        agv_vec = np.zeros((agv_feat_dim,), dtype=np.float32)
        if i < agv_rows:
            emb = agv_emb[i]
            agv_vec[: min(4, agv_feat_dim, emb.shape[0])] = emb[: min(4, agv_feat_dim, emb.shape[0])]
            if agv_feat_dim > 4:
                agv_vec[4] = float(agv_valid_ratio[i])
            if agv_feat_dim > 5:
                agv_vec[5] = float(agv_max_prob[i])
        obs[offset : offset + agv_feat_dim] = agv_vec
        offset += agv_feat_dim

        for slot in range(max_request_slots):
            slot_vec = np.zeros((slot_feat_dim,), dtype=np.float32)
            if i < agv_rows and slot < task_cols:
                emb = task_emb[slot]
                slot_vec[0] = float(emb[0]) if emb.shape[0] > 0 else 0.0
                slot_vec[1] = float(emb[1]) if emb.shape[0] > 1 else 0.0
                if slot_feat_dim > 2:
                    slot_vec[2] = float(logits[i, slot])
                if slot_feat_dim > 3:
                    slot_vec[3] = float(probs[i, slot])
                if slot_feat_dim > 4:
                    slot_vec[4] = float(np.max(probs[:, slot])) if probs.size else 0.0
                if slot_feat_dim > 5:
                    slot_vec[5] = float(agv_action_mask[i, slot] > 0.0)
                if slot_feat_dim > 6:
                    slot_vec[6] = float(task_assigned[slot])

            obs[offset : offset + slot_feat_dim] = slot_vec
            offset += slot_feat_dim

    agv_node_ids = _resolve_agv_node_ids(graph)
    busy_values = []
    for nid in agv_node_ids:
        if 0 <= int(nid) < graph.node_features.shape[0] and graph.node_features.shape[1] > 2:
            busy_values.append(float(graph.node_features[int(nid), 2]))
    num_busy = float(sum(1 for x in busy_values if x > 0.5))
    num_free = float(max(0, len(busy_values) - int(num_busy)))
    num_tasks = float(len(graph.task_node_ids))
    num_assigned = float(np.sum(task_assigned)) if task_assigned.size else 0.0

    global_values = np.array([num_tasks, num_free, num_busy, num_assigned], dtype=np.float32)
    obs[offset : offset + global_feat_dim] = global_values[:global_feat_dim]
    return obs


def encode_graph_assignment_obs(
    graph: GraphState,
    *,
    max_request_slots: int,
    num_agvs: int,
    obs_shape: Tuple[int, ...],
    agv_feat_dim: int = 6,
    slot_feat_dim: int = 7,
    global_feat_dim: int = 4,
    encoder_mode: str = "manual",
    gnn_arch: str = "sage",
) -> np.ndarray:
    """Encode assignment GraphState into fixed-size PPO observation vector."""
    mode = str(encoder_mode).strip().lower()
    if mode not in {"manual", "gnn"}:
        raise ValueError("encoder_mode must be 'manual' or 'gnn'.")

    if mode == "manual":
        return _encode_graph_assignment_obs_manual(
            graph,
            max_request_slots=max_request_slots,
            num_agvs=num_agvs,
            obs_shape=obs_shape,
            agv_feat_dim=agv_feat_dim,
            slot_feat_dim=slot_feat_dim,
            global_feat_dim=global_feat_dim,
        )

    try:
        return _encode_graph_assignment_obs_gnn(
            graph,
            max_request_slots=max_request_slots,
            num_agvs=num_agvs,
            obs_shape=obs_shape,
            agv_feat_dim=agv_feat_dim,
            slot_feat_dim=slot_feat_dim,
            global_feat_dim=global_feat_dim,
            gnn_arch=gnn_arch,
        )
    except Exception as exc:
        warnings.warn(
            f"Falling back to manual graph encoder because GNN encoder failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _encode_graph_assignment_obs_manual(
            graph,
            max_request_slots=max_request_slots,
            num_agvs=num_agvs,
            obs_shape=obs_shape,
            agv_feat_dim=agv_feat_dim,
            slot_feat_dim=slot_feat_dim,
            global_feat_dim=global_feat_dim,
        )
