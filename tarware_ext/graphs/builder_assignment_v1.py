"""Assignment-oriented graph builder with explicit request-slot traceability.

This builder preserves request_queue slot order in task nodes and exposes
metadata mappings so assignment actions (slot indices) stay semantically aligned
with graph nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from tarware.definitions import AgentType

from .schema import GraphState, NodeType


class AssignmentGraphBuilder:
    """Build GraphState for assignment workflows.

    Contract highlights:
    - Task nodes preserve `request_queue` order (slot 0..N-1).
    - Metadata includes bidirectional slot<->node mappings.
    - `task_loc_ids[slot]` maps to env loc_id, or -1 when unmappable.
    """

    def _unwrap_env(self, env: Any) -> Any:
        cand = env
        for _ in range(6):
            if hasattr(cand, "unwrapped") and getattr(cand, "unwrapped") is not cand:
                cand = getattr(cand, "unwrapped")
                continue
            if hasattr(cand, "env") and getattr(cand, "env") is not cand:
                cand = getattr(cand, "env")
                continue
            break
        return cand

    def build(self, env: Any, controller: Any = None) -> GraphState:
        target = self._unwrap_env(env)
        agents = list(getattr(target, "agents", []))
        request_queue = list(getattr(target, "request_queue", []))

        n_agents = len(agents)
        n_tasks = len(request_queue)

        # Features:
        # Agent nodes: [y, x, busy, carrying, is_agv, is_picker]
        # Task nodes:  [y, x, slot_idx, valid_loc, assigned_flag, item_id]
        node_features = np.zeros((n_agents + n_tasks, 6), dtype=float)
        node_types: List[NodeType] = []

        agv_agent_indices: List[int] = []
        picker_agent_indices: List[int] = []

        for i, agent in enumerate(agents):
            is_agv = bool(getattr(agent, "type", None) == AgentType.AGV)
            is_picker = bool(getattr(agent, "type", None) == AgentType.PICKER)
            node_features[i, 0] = float(getattr(agent, "y", 0.0))
            node_features[i, 1] = float(getattr(agent, "x", 0.0))
            node_features[i, 2] = 1.0 if bool(getattr(agent, "busy", False)) else 0.0
            node_features[i, 3] = 1.0 if bool(getattr(agent, "carrying_shelf", None)) else 0.0
            node_features[i, 4] = 1.0 if is_agv else 0.0
            node_features[i, 5] = 1.0 if is_picker else 0.0

            if is_agv:
                node_types.append(NodeType.AGV)
                agv_agent_indices.append(i)
            elif is_picker:
                node_types.append(NodeType.PICKER)
                picker_agent_indices.append(i)
            else:
                node_types.append(NodeType.AGV)

        snapshot: Dict[str, Any] = {}
        if controller is not None and hasattr(controller, "get_assignment_snapshot"):
            try:
                snap = controller.get_assignment_snapshot()
                if isinstance(snap, dict):
                    snapshot = snap
            except Exception:
                snapshot = {}
        assigned_item_ids = {int(x) for x in snapshot.get("assigned_item_ids", [])}
        agv_is_assigned_by_id = snapshot.get("agv_is_assigned", {})
        agv_mission_type_by_id = snapshot.get("agv_mission_type", {})

        agv_is_assigned_values: List[float] = []
        agv_mission_type_values: List[float] = []
        for agv_idx in agv_agent_indices:
            agv = agents[agv_idx]
            agv_is_assigned_values.append(float(agv_is_assigned_by_id.get(id(agv), 0.0)))
            agv_mission_type_values.append(float(agv_mission_type_by_id.get(id(agv), 0.0)))

        coords_to_loc_id = {
            (int(yx[0]), int(yx[1])): int(loc_id)
            for loc_id, yx in dict(getattr(target, "action_id_to_coords_map", {})).items()
        }

        task_loc_ids: List[int] = []
        task_item_ids: List[int] = []
        task_valid_loc_mask: List[bool] = []
        request_slot_to_node_id: Dict[int, int] = {}
        node_id_to_request_slot: Dict[int, int] = {}

        for slot, item in enumerate(request_queue):
            task_node_id = n_agents + slot
            ty = int(getattr(item, "y", 0))
            tx = int(getattr(item, "x", 0))
            item_id = int(getattr(item, "id", -1))
            loc_id = coords_to_loc_id.get((ty, tx), -1)
            valid_loc = loc_id >= 0
            assigned_flag = item_id in assigned_item_ids

            node_features[task_node_id, 0] = float(ty)
            node_features[task_node_id, 1] = float(tx)
            node_features[task_node_id, 2] = float(slot)
            node_features[task_node_id, 3] = 1.0 if valid_loc else 0.0
            node_features[task_node_id, 4] = 1.0 if assigned_flag else 0.0
            node_features[task_node_id, 5] = float(item_id)
            node_types.append(NodeType.SHELF)

            request_slot_to_node_id[slot] = task_node_id
            node_id_to_request_slot[task_node_id] = slot
            task_loc_ids.append(int(loc_id))
            task_item_ids.append(item_id)
            task_valid_loc_mask.append(valid_loc)

        # Dense AGV->task edges.
        if not agv_agent_indices or n_tasks == 0:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_attr = np.zeros((0, 2), dtype=float)
        else:
            sources: List[int] = []
            targets: List[int] = []
            attrs: List[List[float]] = []
            for agv_agent_idx in agv_agent_indices:
                ay = int(node_features[agv_agent_idx, 0])
                ax = int(node_features[agv_agent_idx, 1])
                for slot in range(n_tasks):
                    task_node_id = n_agents + slot
                    ty = int(node_features[task_node_id, 0])
                    tx = int(node_features[task_node_id, 1])
                    manhattan = abs(ay - ty) + abs(ax - tx)
                    valid_loc = bool(task_valid_loc_mask[slot])
                    sources.append(agv_agent_idx)
                    targets.append(task_node_id)
                    attrs.append([float(manhattan), 1.0 if valid_loc else 0.0])
            edge_index = np.array([sources, targets], dtype=int)
            edge_attr = np.array(attrs, dtype=float)

        # Agent-task action mask aligned with env.agents x request_queue slots.
        action_mask = np.zeros((n_agents, n_tasks), dtype=bool)
        valid_masks = None
        try:
            valid_masks = np.array(target.compute_valid_action_masks(), dtype=bool)
        except Exception:
            valid_masks = None

        for agent_idx in range(n_agents):
            if agent_idx not in agv_agent_indices:
                continue
            for slot in range(n_tasks):
                loc_id = task_loc_ids[slot]
                if loc_id < 0:
                    action_mask[agent_idx, slot] = False
                    continue
                if valid_masks is None:
                    action_mask[agent_idx, slot] = True
                else:
                    action_mask[agent_idx, slot] = bool(valid_masks[agent_idx, int(loc_id)])

        metadata: Dict[str, Any] = {
            "graph_family": "assignment_v1",
            "num_agents": n_agents,
            "num_tasks": n_tasks,
            "grid_size_y": int(getattr(target, "grid_size", [0, 0])[0]),
            "grid_size_x": int(getattr(target, "grid_size", [0, 0])[1]),
            "agv_agent_indices": agv_agent_indices,
            "picker_agent_indices": picker_agent_indices,
            "request_slot_to_node_id": request_slot_to_node_id,
            "node_id_to_request_slot": node_id_to_request_slot,
            "request_slot_to_loc_id": {slot: loc for slot, loc in enumerate(task_loc_ids)},
            "task_item_ids": task_item_ids,
            "task_valid_loc_mask": task_valid_loc_mask,
            "agv_is_assigned_values": agv_is_assigned_values,
            "agv_mission_type_values": agv_mission_type_values,
            "edge_attr": edge_attr,
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
