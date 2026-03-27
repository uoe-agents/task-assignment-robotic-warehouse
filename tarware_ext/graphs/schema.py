"""Graph schema definitions.

This module defines the canonical GraphState dataclass used by builders and
policies. The goal is to keep a compact, serialisable description of the
constructed graph so policies (dummy or trained) can consume the same API.

Notes on conventions:
- Coordinates are represented internally as `(y, x)` (row, column). Builders
    and helpers must perform any needed swaps when interacting with the core
    env which historically mixes `(x,y)` and `(y,x)` in a few places.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class NodeType(str, Enum):
        AGV = "agv"
        PICKER = "picker"
        SHELF = "shelf"
        GOAL = "goal"


@dataclass
class GraphState:
        """Canonical graph representation returned by builders.

        Fields:
        - node_features: array (N_nodes, F) of node-level features (first two dims
            should contain y,x coordinates).
        - edge_index: array (2, N_edges) containing source and target node indices
            for each edge.
        - node_types: list of `NodeType` with length N_nodes.
        - metadata: arbitrary dict with environment-level info (grid size, seed).
        - agent_node_ids: indices (into node_features) corresponding to agents.
        - task_node_ids: indices corresponding to task nodes (requested shelves).
        - task_loc_ids: list mapping each task node -> `env` loc_id (int). This is
            required so a policy that selects a task index can be mapped to a valid
            macro-action for `env.step`.
        - action_mask: optional boolean array with shape (n_agents, n_tasks)
            indicating which (agent,task) assignments are valid. If None, all are
            considered valid.
        """

        node_features: np.ndarray
        edge_index: np.ndarray
        node_types: List[NodeType]
        metadata: Dict[str, Any]

        agent_node_ids: List[int]
        task_node_ids: List[int]
        task_loc_ids: List[int]
        action_mask: Optional[np.ndarray] = None
