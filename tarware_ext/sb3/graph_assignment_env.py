# ============================================================
"""GraphAssignmentEnv: A Gymnasium-compatible reinforcement learning environment for AGV task assignment.

This module provides a single-agent RL wrapper that controls explicit AGV-to-request assignments
in a robotic warehouse setting. The environment uses a heuristic controller to manage the full
mission lifecycle (picking, delivering, returning) while the RL agent focuses solely on making
assignment decisions.

Key Features:
- Single-agent Gymnasium API compatible with SB3 algorithms (e.g., PPO, MaskablePPO)
- Explicit AGV assignment actions: each AGV can be assigned to a request queue slot or pass
- Flexible observation backends: flat vector, graph-based, or dictionary-based with graph data
- Support for GNN encoders (SAGE, GCN, GAT) when using graph observations
- Action masking for valid assignment constraints
- Episode statistics tracking (deliveries, clashes, stucks, pick rate)

Configuration:
- obs_backend: Choose observation representation ("assignment", "graph", or "graph_dict")
- graph_encoder_mode: Manual feature extraction or GNN-based encoding
- max_request_slots: Maximum tasks in the assignment action space
- max_steps: Episode length limit
- distance_mode: Manhattan or Euclidean distance calculation

The environment bridges between low-level warehouse simulation and high-level RL training,
allowing agents to learn efficient task assignment strategies.

FILE: tarware_ext/sb3/graph_assignment_env.py
============================================================

SB3 env for explicit AGV assignment with heuristic mission parity.

Design:
- SB3 controls only new AGV->request assignments (explicit indices).
- A stateful heuristic controller executes full mission lifecycle for AGVs and
  pickers: PICKING -> DELIVERING -> RETURNING.
- Exposes a standard single-agent Gymnasium API for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings

import gymnasium as gym
import numpy as np

import tarware  # noqa: F401

from tarware_ext.controllers import HeuristicController
from tarware_ext.envs.tarware_adapter import TarwareAdapter, Transition
from tarware_ext.graphs import AssignmentGraphBuilder
from tarware_ext.sb3.assignment_obs_encoder import encode_assignment_obs
from tarware_ext.sb3.graph_assignment_obs_encoder import encode_graph_assignment_obs
from tarware.definitions import AgentType


@dataclass
class GraphAssignmentConfig:
    env_id: str
    max_request_slots: Optional[int] = None
    top_k: Optional[int] = None  # Deprecated alias for max_request_slots.
    max_steps: int = 200
    seed: Optional[int] = None
    distance_mode: str = "manhattan"
    obs_backend: str = "assignment"  # "assignment" (A), "graph" (B) or "graph_dict" (C)
    graph_encoder_mode: str = "manual"  # "manual" or "gnn" when obs_backend="graph"
    graph_gnn_arch: str = "sage"  # "sage", "gcn" or "gat" when graph_encoder_mode="gnn"
    verbose: bool = False
    debug_first_n_steps: int = 3


class GraphAssignmentEnv(gym.Env):
    """Single-agent wrapper where the action is explicit AGV->item assignment."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: GraphAssignmentConfig) -> None:
        super().__init__()
        self.cfg = config
        self._t = 0

        self._obs_backend = str(self.cfg.obs_backend).strip().lower()
        if self._obs_backend not in ("assignment", "graph", "graph_dict"):
            raise ValueError(
                "GraphAssignmentConfig.obs_backend must be 'assignment', 'graph' or 'graph_dict'."
            )

        self._graph_encoder_mode = str(self.cfg.graph_encoder_mode).strip().lower()
        if self._graph_encoder_mode not in ("manual", "gnn"):
            raise ValueError("GraphAssignmentConfig.graph_encoder_mode must be 'manual' or 'gnn'.")

        self._graph_gnn_arch = str(self.cfg.graph_gnn_arch).strip().lower()
        if self._graph_gnn_arch not in ("sage", "gcn", "gat"):
            raise ValueError("GraphAssignmentConfig.graph_gnn_arch must be 'sage', 'gcn' or 'gat'.")

        self._agv_feat_dim = 6
        self._slot_feat_dim = 7
        self._global_feat_dim = 4
        self._graph_node_feat_dim = 6
        self._graph_edge_feat_dim = 2

        base = gym.make(self.cfg.env_id, disable_env_checker=True)
        # Keep per-agent done semantics so we can build proper terminated/truncated.
        self.env = TarwareAdapter(base, done_all=False)
        self.controller = HeuristicController()
        self.graph_builder = AssignmentGraphBuilder()

        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_deliveries = 0
        self._ep_clashes = 0
        self._ep_stucks = 0

        _obs, _info = self.env.reset(seed=self.cfg.seed)
        unwrapped = self._unwrap_env()
        self.controller.reset(unwrapped, seed=self.cfg.seed)

        agents = list(getattr(unwrapped, "agents", []))
        self.num_agents = len(agents) or 1
        self.num_agvs = sum(1 for a in agents if getattr(a, "type", None) == AgentType.AGV)
        self.num_agvs = max(1, int(self.num_agvs))

        configured_slots = self.cfg.max_request_slots
        if configured_slots is None and self.cfg.top_k is not None:
            warnings.warn(
                "GraphAssignmentConfig.top_k is deprecated; use max_request_slots instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            configured_slots = int(self.cfg.top_k)

        if configured_slots is None:
            inferred_tasks = int(getattr(unwrapped, "request_queue_size", 0) or 0)
            if inferred_tasks <= 0:
                inferred_tasks = len(getattr(unwrapped, "request_queue", []))
            configured_slots = max(1, int(inferred_tasks))

        self.max_request_slots = max(1, int(configured_slots))

        if self._obs_backend == "graph_dict":
            graph0 = self.graph_builder.build(unwrapped, controller=self.controller)
            self._graph_node_feat_dim = int(graph0.node_features.shape[1])

            edge_attr0 = graph0.metadata.get("edge_attr") if isinstance(graph0.metadata, dict) else None
            if edge_attr0 is not None and getattr(edge_attr0, "ndim", 0) == 2 and edge_attr0.shape[1] > 0:
                self._graph_edge_feat_dim = int(edge_attr0.shape[1])

            self._graph_max_nodes = int(self.num_agents + self.max_request_slots)
            self._graph_max_edges = int(self.num_agvs * self.max_request_slots)

            self.observation_space = gym.spaces.Dict(
                {
                    "node_features": gym.spaces.Box(
                        low=-1e9,
                        high=1e9,
                        shape=(self._graph_max_nodes, self._graph_node_feat_dim),
                        dtype=np.float32,
                    ),
                    "edge_index": gym.spaces.Box(
                        low=0,
                        high=max(self._graph_max_nodes - 1, 0),
                        shape=(2, self._graph_max_edges),
                        dtype=np.int32,
                    ),
                    "edge_attr": gym.spaces.Box(
                        low=-1e9,
                        high=1e9,
                        shape=(self._graph_max_edges, self._graph_edge_feat_dim),
                        dtype=np.float32,
                    ),
                    "action_mask": gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.num_agvs, self.max_request_slots),
                        dtype=np.int8,
                    ),
                    "n_nodes": gym.spaces.Box(low=0, high=self._graph_max_nodes, shape=(1,), dtype=np.int32),
                    "n_edges": gym.spaces.Box(low=0, high=self._graph_max_edges, shape=(1,), dtype=np.int32),
                    "n_tasks": gym.spaces.Box(low=0, high=self.max_request_slots, shape=(1,), dtype=np.int32),
                }
            )
        else:
            obs_dim = (
                self.num_agvs * (self._agv_feat_dim + self.max_request_slots * self._slot_feat_dim)
                + self._global_feat_dim
            )
            self.observation_space = gym.spaces.Box(
                low=-1e9,
                high=1e9,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        # Explicit AGV assignment: 0=no assignment, 1..R=request_queue slot.
        self.action_space = gym.spaces.MultiDiscrete([self.max_request_slots + 1] * self.num_agvs)

        self._last_obs: Any = self._encode_obs(unwrapped)

    def _resolve_agv_action_mask_rows(self, graph: Any) -> List[int]:
        if getattr(graph, "action_mask", None) is None:
            return []

        agv_rows = []
        metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
        raw = metadata.get("agv_agent_indices")
        if isinstance(raw, list):
            for idx in raw:
                i = int(idx)
                if 0 <= i < graph.action_mask.shape[0]:
                    agv_rows.append(i)
        return agv_rows

    def _encode_graph_dict_obs(self, env: Any) -> Dict[str, np.ndarray]:
        """
        Encodes a graph dictionary observation from the environment.
        Converts graph data into a dictionary of normalized numpy arrays suitable for 
        neural network processing. Handles node features, edge indices, edge attributes, 
        and action masks, clipping them to predefined maximum dimensions.
        Args:
            env: The environment object used to build the graph.
        Returns:
            dict: A dictionary containing:
                - node_features: (max_nodes, node_feat_dim) array of node feature vectors
                - edge_index: (2, max_edges) array of edge connectivity indices
                - edge_attr: (max_edges, edge_feat_dim) array of edge attributes
                - action_mask: (num_agvs, max_request_slots) binary mask of valid actions
                - n_nodes: actual number of nodes in the graph
                - n_edges: actual number of edges in the graph
                - n_tasks: actual number of tasks in the graph
        """
        
        graph = self.graph_builder.build(env, controller=self.controller)

        node_features = np.zeros(
            (self._graph_max_nodes, self._graph_node_feat_dim),
            dtype=np.float32,
        )
        n_nodes = min(int(graph.node_features.shape[0]), self._graph_max_nodes)
        if n_nodes > 0:
            node_features[:n_nodes, :] = graph.node_features[:n_nodes, : self._graph_node_feat_dim].astype(np.float32)

        edge_index = np.zeros((2, self._graph_max_edges), dtype=np.int32)
        n_edges_raw = int(graph.edge_index.shape[1]) if graph.edge_index.ndim == 2 else 0
        n_edges = min(n_edges_raw, self._graph_max_edges)
        if n_edges > 0:
            clipped = np.clip(graph.edge_index[:, :n_edges], 0, max(n_nodes - 1, 0))
            edge_index[:, :n_edges] = clipped.astype(np.int32)

        edge_attr = np.zeros((self._graph_max_edges, self._graph_edge_feat_dim), dtype=np.float32)
        edge_attr_raw = graph.metadata.get("edge_attr") if isinstance(graph.metadata, dict) else None
        if edge_attr_raw is not None and getattr(edge_attr_raw, "ndim", 0) == 2:
            copy_edges = min(n_edges, int(edge_attr_raw.shape[0]))
            if copy_edges > 0:
                edge_attr[:copy_edges, :] = edge_attr_raw[:copy_edges, : self._graph_edge_feat_dim].astype(np.float32)

        action_mask = np.zeros((self.num_agvs, self.max_request_slots), dtype=np.int8)
        agv_rows = self._resolve_agv_action_mask_rows(graph)
        n_tasks = min(int(len(graph.task_node_ids)), self.max_request_slots)
        for i, row_idx in enumerate(agv_rows[: self.num_agvs]):
            if n_tasks <= 0:
                break
            action_mask[i, :n_tasks] = graph.action_mask[int(row_idx), :n_tasks].astype(np.int8)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "action_mask": action_mask,
            "n_nodes": np.array([n_nodes], dtype=np.int32),
            "n_edges": np.array([n_edges], dtype=np.int32),
            "n_tasks": np.array([n_tasks], dtype=np.int32),
        }

    def action_masks(self) -> np.ndarray:
        """Return valid-action masks for MaskablePPO.

        Shape: (num_agvs, max_request_slots + 1), where index 0 is no-op.
        """

        mask_tasks = None
        if isinstance(self._last_obs, dict) and "action_mask" in self._last_obs:
            mask_tasks = np.asarray(self._last_obs["action_mask"], dtype=bool)
        else:
            graph = self.graph_builder.build(self._unwrap_env(), controller=self.controller)
            mask_tasks = np.zeros((self.num_agvs, self.max_request_slots), dtype=bool)
            agv_rows = self._resolve_agv_action_mask_rows(graph)
            n_tasks = min(int(len(graph.task_node_ids)), self.max_request_slots)
            for i, row_idx in enumerate(agv_rows[: self.num_agvs]):
                if n_tasks <= 0:
                    break
                mask_tasks[i, :n_tasks] = graph.action_mask[int(row_idx), :n_tasks].astype(bool)

        mask_full = np.zeros((self.num_agvs, self.max_request_slots + 1), dtype=bool)
        # Keep no-op always valid so every AGV has at least one valid action.
        mask_full[:, 0] = True
        mask_full[:, 1:] = mask_tasks[:, : self.max_request_slots]
        return mask_full

    def _unwrap_env(self) -> Any:
        cand: Any = self.env
        for _ in range(6):
            if hasattr(cand, "unwrapped") and getattr(cand, "unwrapped") is not cand:
                cand = getattr(cand, "unwrapped")
                continue
            if hasattr(cand, "env") and getattr(cand, "env") is not cand:
                cand = getattr(cand, "env")
                continue
            break
        return cand

    def _update_episode_counters(self, reward: float, info: Dict[str, Any]) -> None:
        self._ep_len += 1
        self._ep_return += float(reward)
        if "shelf_deliveries" in info:
            self._ep_deliveries += int(info.get("shelf_deliveries", 0))
        if "clashes" in info:
            self._ep_clashes += int(info.get("clashes", 0))
        if "stucks" in info:
            self._ep_stucks += int(info.get("stucks", 0))

    def _encode_obs(self, env: Any) -> Any:
        if self._obs_backend == "graph_dict":
            return self._encode_graph_dict_obs(env)

        if self._obs_backend == "graph":
            graph = self.graph_builder.build(env, controller=self.controller)
            return encode_graph_assignment_obs(
                graph,
                max_request_slots=self.max_request_slots,
                num_agvs=self.num_agvs,
                obs_shape=self.observation_space.shape,
                agv_feat_dim=self._agv_feat_dim,
                slot_feat_dim=self._slot_feat_dim,
                global_feat_dim=self._global_feat_dim,
                encoder_mode=self._graph_encoder_mode,
                gnn_arch=self._graph_gnn_arch,
            )

        return encode_assignment_obs(
            env,
            controller=self.controller,
            max_request_slots=self.max_request_slots,
            num_agvs=self.num_agvs,
            obs_shape=self.observation_space.shape,
            agv_feat_dim=self._agv_feat_dim,
            slot_feat_dim=self._slot_feat_dim,
            global_feat_dim=self._global_feat_dim,
        )

    def _attach_episode_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        episode_length = max(self._ep_len, 1)
        pick_rate = float(self._ep_deliveries) * 3600.0 / (5.0 * float(episode_length))

        out = dict(info) if isinstance(info, dict) else {}
        out["episode"] = {
            "r": float(self._ep_return),
            "l": int(self._ep_len),
            "deliveries": int(self._ep_deliveries),
            "clashes": int(self._ep_clashes),
            "stucks": int(self._ep_stucks),
            "pick_rate": float(pick_rate),
        }
        return out

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state for a new episode.
        Initializes episode tracking variables (time, return, length, deliveries, clashes, 
        and stuck counts) and resets the underlying environment and controller. Encodes the 
        initial observation and returns it along with info.
        Args:
            seed: Optional seed for reproducibility.
            options: Optional dictionary of reset options.
        Returns:
            Tuple containing the initial observation (dict or array) and info dictionary.
        """
        
        
        self._t = 0
        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_deliveries = 0
        self._ep_clashes = 0
        self._ep_stucks = 0

        _obs, info = self.env.reset(seed=seed if seed is not None else self.cfg.seed, options=options)
        unwrapped = self._unwrap_env()
        self.controller.reset(unwrapped, seed=seed if seed is not None else self.cfg.seed)

        self._last_obs = self._encode_obs(unwrapped)

        if self.cfg.verbose:
            print(
                f"[reset] num_tasks={len(getattr(unwrapped, 'request_queue', []))} "
                f"num_agvs={self.num_agvs} request_slots={self.max_request_slots}"
            )

        if isinstance(self._last_obs, dict):
            out_obs = {k: v.copy() for k, v in self._last_obs.items()}
        else:
            out_obs = self._last_obs.copy()
        return out_obs, info if isinstance(info, dict) else {}

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        Converts the RL agent's action into environment-specific actions through the controller,
        executes the step in the underlying environment, and processes the results.
        Args:
            action (np.ndarray): The action from the RL agent to be converted into environment actions.
        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - observation (Any): The encoded observation from the environment.
                - reward (float): The aggregated reward for this step.
                - terminated (bool): Whether the episode has terminated naturally.
                - truncated (bool): Whether the episode was truncated (max steps reached or done condition).
                - info (dict): Additional information about the step, including episode statistics if terminal.
        Notes:
            - Handles both Transition and tuple-based step outputs.
            - Aggregates multi-agent rewards and termination conditions.
            - Automatically truncates episodes when max_steps is reached.
            - Logs debug information if verbose mode is enabled.
            - Updates internal episode counters and observation cache.
        """
        
        
        self._t += 1

        assignments = [int(x) for x in np.asarray(action).reshape(-1).tolist()]
        unwrapped = self._unwrap_env()
        loc_actions = self.controller.step(unwrapped, rl_agv_assignments=assignments)

        if self.cfg.verbose and self._t <= self.cfg.debug_first_n_steps:
            print(f"[assignment] t={self._t} rl={assignments} env_actions={loc_actions}")

        step_out = self.env.step(loc_actions)

        if isinstance(step_out, Transition):
            reward = float(step_out.reward_team)
            terminated = bool(all(step_out.terminated_by_agent))
            truncated = bool(step_out.done_all and not terminated)
            info = step_out.info if isinstance(step_out.info, dict) else {}
        else:
            _obs, reward_raw, terminated_raw, truncated_raw, info_raw = step_out
            reward = float(np.sum(reward_raw)) if isinstance(reward_raw, (list, tuple, np.ndarray)) else float(reward_raw)
            if isinstance(terminated_raw, (list, tuple, np.ndarray)):
                terminated = bool(all(bool(x) for x in terminated_raw))
            else:
                terminated = bool(terminated_raw)
            if isinstance(truncated_raw, (list, tuple, np.ndarray)):
                done_all = bool(all(bool(t) or bool(tr) for t, tr in zip(terminated_raw, truncated_raw)))
                truncated = bool(done_all and not terminated)
            else:
                truncated = bool(truncated_raw)
            info = info_raw if isinstance(info_raw, dict) else {}

        self._update_episode_counters(reward=reward, info=info)

        obs_vec = self._encode_obs(self._unwrap_env())
        self._last_obs = obs_vec

        if self._t >= self.cfg.max_steps:
            truncated = True

        if terminated or truncated:
            info = self._attach_episode_info(info)

        if self.cfg.verbose and (self._t <= 3 or terminated or truncated):
            print(
                f"[step] t={self._t} reward={reward:.3f} term={terminated} trunc={truncated} "
                f"tasks={len(getattr(self._unwrap_env(), 'request_queue', []))}"
            )

        if isinstance(obs_vec, dict):
            out_obs = {k: v.copy() for k, v in obs_vec.items()}
        else:
            out_obs = obs_vec.copy()
        return out_obs, reward, terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
