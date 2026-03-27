"""GraphBuilder V0 - simple, incremental graph builder for MRTA.

This builder is intentionally conservative: it only creates nodes for agents
and for items currently in `env.request_queue`. Edges connect agents -> tasks
and carry a simple Manhattan distance attribute. The goal is to have a
workable GraphState for initial experiments and for a dummy policy.

Design goals:
- Minimal dependencies and clear, documented features.
- Produce `task_loc_ids` so policies can map a selected task -> env loc_id.
- Provide an action_mask when the environment exposes one.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from .schema import GraphState, NodeType
from .utils import yx_to_loc_id, loc_id_to_yx, PathCache
from tarware.definitions import AgentType


class GraphBuilderV0:
    """Simple graph builder that maps env -> GraphState.

    Parameters
    - distance_mode: currently only "manhattan" is supported (placeholder
      for future FIND_PATH distances).
    - top_k: when provided, can be used by policies to limit candidate tasks
      per agent (builder still returns full graph for now).
    """

    def __init__(self, distance_mode: str = "manhattan", top_k: int | None = 2):
        self.distance_mode = distance_mode
        self.top_k = top_k

    def build(self, env: Any) -> GraphState:
        agents = list(env.agents)
        request_queue = list(env.request_queue)

        # Map request_queue items to loc_id (skip items we cannot map)
        tasks: List[Tuple[Any, int, Tuple[int, int]]] = []  # (item, loc_id, (y,x))
        for item in request_queue:
            item_yx = (int(item.y), int(item.x))
            loc_id = yx_to_loc_id(env, item_yx)
            if loc_id is None:
                # Skip items that are not present in action_id_to_coords_map
                continue
            tasks.append((item, int(loc_id), item_yx))

        n_agents = len(agents)
        n_tasks = len(tasks)

        # Node features: [y, x, busy_flag, carrying_flag]
        node_features = np.zeros((n_agents + n_tasks, 4), dtype=float)
        node_types: List[NodeType] = []

        for i, agent in enumerate(agents):
            node_features[i, 0] = float(agent.y)
            node_features[i, 1] = float(agent.x)
            node_features[i, 2] = 1.0 if getattr(agent, "busy", False) else 0.0
            node_features[i, 3] = 1.0 if getattr(agent, "carrying_shelf", None) else 0.0
            # Map agent types
            if agent.type == AgentType.AGV:
                node_types.append(NodeType.AGV)
            elif agent.type == AgentType.PICKER:
                node_types.append(NodeType.PICKER)
            else:
                node_types.append(NodeType.AGV)

        for j, (_item, loc_id, (y, x)) in enumerate(tasks):
            idx = n_agents + j
            node_features[idx, 0] = float(y)
            node_features[idx, 1] = float(x)
            node_features[idx, 2] = 0.0
            node_features[idx, 3] = 0.0
            node_types.append(NodeType.SHELF)

        # Build edges: agent -> task fully connected (source = agent_idx, target = task_idx)
        if n_agents == 0 or n_tasks == 0:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_attr = np.zeros((0, 1), dtype=float)
        else:
            sources = []
            targets = []
            attrs = []
            for i in range(n_agents):
                ay, ax = int(node_features[i, 0]), int(node_features[i, 1])
                for j in range(n_tasks):
                    tj = n_agents + j
                    ty, tx = int(node_features[tj, 0]), int(node_features[tj, 1])
                    sources.append(i)
                    targets.append(tj)
                    manhattan = abs(ay - ty) + abs(ax - tx)
                    attrs.append([float(manhattan)])

            edge_index = np.vstack([np.array(sources, dtype=int), np.array(targets, dtype=int)])
            edge_attr = np.array(attrs, dtype=float)

        # Compute Manhattan distances matrix (n_agents, n_tasks)
        manhattan = np.zeros((n_agents, n_tasks), dtype=float) if n_agents and n_tasks else np.zeros((n_agents, n_tasks), dtype=float)
        for i in range(n_agents):
            for j in range(n_tasks):
                ajy, ajx = float(node_features[i, 0]), float(node_features[i, 1])
                tj = n_agents + j
                tjy, tjx = float(node_features[tj, 0]), float(node_features[tj, 1])
                manhattan[i, j] = abs(ajy - tjy) + abs(ajx - tjx)

        # Path distances (computed selectively using top_k and cached per build call)
        path_cache = PathCache()
        path_distances = np.full((n_agents, n_tasks), np.inf, dtype=float)

        # Determine per-agent candidate tasks (top-k based on Manhattan). If
        # top_k is None and distance_mode == 'find_path' we compute paths for
        # all tasks; otherwise only for top_k nearest by Manhattan.
        top_k_candidates: List[List[int]] = []
        for i in range(n_agents):
            if n_tasks == 0:
                top_k_candidates.append([])
                continue
            if self.top_k is None:
                if self.distance_mode == "find_path":
                    candidate_idxs = list(range(n_tasks))
                else:
                    candidate_idxs = []
            else:
                k = min(self.top_k, n_tasks)
                candidate_idxs = list(np.argsort(manhattan[i])[:k])
            top_k_candidates.append(candidate_idxs)

        # Compute path lengths only for candidate pairs and cache results.
        for i in range(n_agents):
            agent = agents[i]
            agent_type = getattr(agent, "type", None)
            agent_type_key = agent_type.name if agent_type is not None else None
            start = (int(node_features[i, 0]), int(node_features[i, 1]))
            # Skip expensive path computations for PICKER agents and for AGVs
            # that are currently busy; they will fall back to Manhattan
            # distances in scoring.
            if agent_type == AgentType.PICKER or getattr(agent, "busy", False):
                continue
            for j in top_k_candidates[i]:
                tj = n_agents + j
                goal = (int(node_features[tj, 0]), int(node_features[tj, 1]))
                cached = path_cache.get(start, goal, True, agent_type_key)
                if cached is not None:
                    path_len = cached
                else:
                    try:
                        p = env.find_path(start, goal, agent, care_for_agents=True)
                    except Exception:
                        p = None
                    if p:
                        # use number of steps (edges) as distance
                        path_len = max(0, len(p) - 1)
                    else:
                        path_len = float("inf")
                    path_cache.set(start, goal, True, path_len, agent_type_key)
                path_distances[i, j] = float(path_len) if path_len != float("inf") else np.inf

        # Combine scoring distances: by default use Manhattan, but replace
        # entries with path distances where available (and when requested).
        scoring_distances = manhattan.copy()
        if self.distance_mode == "find_path":
            # Prefer path distances when computed, otherwise keep Manhattan
            mask = np.isfinite(path_distances)
            scoring_distances[mask] = path_distances[mask]
        else:
            # If top_k candidates have path distances available, use them for
            # scoring for those entries.
            mask = np.isfinite(path_distances)
            scoring_distances[mask] = path_distances[mask]

        # Build action_mask (n_agents, n_tasks) by asking the env, if possible
        action_mask = None
        try:
            valid_masks = env.compute_valid_action_masks()
            # valid_masks shape: (num_agents, action_size)
            # For each task, use its loc_id column in valid_masks
            mask = np.ones((n_agents, n_tasks), dtype=bool)
            for i in range(n_agents):
                for j, (_item, loc_id, _yx) in enumerate(tasks):
                    # valid_masks uses action id indices directly
                    mask[i, j] = bool(valid_masks[i, int(loc_id)])
            action_mask = mask
        except Exception:
            # Env may not expose masks in some wrappers; leave None
            action_mask = None

        task_loc_ids = [loc_id for (_item, loc_id, _yx) in tasks]

        metadata = {
            "grid_size_x": int(env.grid_size[1]),
            "grid_size_y": int(env.grid_size[0]),
            "num_agents": n_agents,
            "num_tasks": n_tasks,
            "distance_mode": self.distance_mode,
            "top_k": int(self.top_k) if self.top_k is not None else None,
            "manhattan_distances": manhattan,
            "path_distances": path_distances,
            "scoring_distances": scoring_distances,
            "top_k_candidates": top_k_candidates,
        }

        return GraphState(
            node_features=node_features,
            edge_index=edge_index,
            node_types=node_types,
            metadata=metadata,
            agent_node_ids=list(range(n_agents)),
            task_node_ids=list(range(n_agents, n_agents + n_tasks)),
            task_loc_ids=task_loc_ids,
            action_mask=action_mask,
        )
