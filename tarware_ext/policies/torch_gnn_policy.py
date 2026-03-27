"""Minimal PyTorch-based GNN policy scaffold.

This implementation avoids external graph libs: it encodes node features with
a small MLP, scores agent->task pairs via dot-product (or small scorer MLP),
and performs greedy assignment. It requires `torch` to run; tests using this
module should use `pytest.importorskip('torch')` to skip when unavailable.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch

from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.graphs.serializer import graphstate_to_torch
from tarware_ext.graphs.schema import GraphState, NodeType


class TorchGNNPolicy:
    uses_env = True

    def __init__(self, builder: Optional[GraphBuilderV0] = None, hidden_dim: int = 64, device: str = "cpu") -> None:
        self.builder = builder or GraphBuilderV0()
        self.hidden_dim = int(hidden_dim)
        self.device = device
        # lazy import torch modules to avoid import errors at package import time
        self.torch = None
        self.model = None
        self._init_model = False

    def _ensure_torch(self):
        if self.torch is None:
            import torch
            import torch.nn as nn
            self.torch = torch

            # Simple GraphSAGE-style message passing encoder (two layers).
            class GraphSAGE(nn.Module):
                def __init__(self, in_dim: int, hidden: int, num_layers: int = 2):
                    super().__init__()
                    self.num_layers = num_layers
                    self.linears_self = nn.ModuleList()
                    self.linears_neigh = nn.ModuleList()
                    for l in range(num_layers):
                        in_d = in_dim if l == 0 else hidden
                        self.linears_self.append(nn.Linear(in_d, hidden))
                        self.linears_neigh.append(nn.Linear(in_d, hidden))

                def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
                    # x: (N, F), edge_index: (2, E)
                    N = x.shape[0]
                    device = x.device
                    # make undirected edges by adding reversed edges
                    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                    src = ei[0]
                    dst = ei[1]

                    for l in range(self.num_layers):
                        # aggregate neighbor features by mean
                        neigh_sum = torch.zeros_like(x)
                        neigh_sum.index_add_(0, dst, x[src])
                        deg = torch.zeros((N,), dtype=x.dtype, device=device)
                        deg.index_add_(0, dst, torch.ones((src.shape[0],), dtype=x.dtype, device=device))
                        deg = deg.clamp_min(1.0).unsqueeze(1)
                        neigh_mean = neigh_sum / deg

                        h_self = self.linears_self[l](x)
                        h_neigh = self.linears_neigh[l](neigh_mean)
                        x = h_self + h_neigh
                        x = torch.relu(x)

                    return x

            self.model = GraphSAGE(4, self.hidden_dim, num_layers=2)
            self.scorer = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.model.to(self.device)
            self.scorer.to(self.device)
            self._init_model = True

    def reset(self, env: Any) -> None:
        self._ensure_torch()

    def _greedy_assign(self, scores: "torch.Tensor", valid_mask: Optional["torch.Tensor"]) -> List[int]:
        # scores: (n_agvs, n_tasks) tensor
        t = self.torch
        n_agvs, n_tasks = scores.shape
        assigned = [-1] * int(n_agvs)
        agent_free = [True] * int(n_agvs)
        task_free = [True] * int(n_tasks)

        pairs = []
        scores_np = scores.detach().cpu().numpy()
        valid_np = None if valid_mask is None else valid_mask.detach().cpu().numpy()
        for i in range(n_agvs):
            for j in range(n_tasks):
                if valid_np is not None and not bool(valid_np[i, j]):
                    continue
                pairs.append((float(scores_np[i, j]), int(i), int(j)))
        pairs.sort(key=lambda x: -x[0])
        for score, i, j in pairs:
            if agent_free[i] and task_free[j]:
                assigned[i] = j
                agent_free[i] = False
                task_free[j] = False
        return assigned

    def act(self, env: Any) -> List[int]:
        self._ensure_torch()
        torch = self.torch

        # Accept wrapped envs; builder expects unwrapped Warehouse
        target_env = env.unwrapped if hasattr(env, "unwrapped") else env
        g: GraphState = self.builder.build(target_env)

        n_agents = len(g.agent_node_ids)
        n_tasks = len(g.task_node_ids)
        if n_tasks == 0:
            return [0 for _ in range(len(g.agent_node_ids))]

        data = graphstate_to_torch(g, device=self.device)
        node_features = data["node_features"]  # Tensor (N, F)
        edge_index = data["edge_index"]
        agent_node_ids = data["agent_node_ids"]
        task_node_ids = data["task_node_ids"]
        action_mask = data.get("action_mask")
        metadata = data.get("metadata", {})

        # Encode nodes
        with torch.no_grad():
            embeds = self.model(node_features, edge_index)  # (N, hidden)

            agent_emb = embeds[agent_node_ids.long(), :]
            task_emb = embeds[task_node_ids.long(), :]

            # pairwise scoring via scorer MLP on cat([a,t])
            # compute dense pairwise by broadcasting
            na = agent_emb.shape[0]
            nt = task_emb.shape[0]
            a_exp = agent_emb.unsqueeze(1).expand(-1, nt, -1)
            t_exp = task_emb.unsqueeze(0).expand(na, -1, -1)
            pair = torch.cat([a_exp, t_exp], dim=-1)  # (na, nt, 2*hidden)
            pair_flat = pair.view(-1, pair.shape[-1])
            scores_flat = self.scorer(pair_flat).view(na, nt).squeeze(-1)

            # valid mask
            if action_mask is not None:
                # action_mask originally aligned with full agent list; select agent rows
                # if action_mask is torch tensor on device
                valid_mask = action_mask[agent_node_ids.long(), :]
            else:
                valid_mask = torch.ones_like(scores_flat, dtype=torch.bool)

            # apply metadata top_k_candidates if present
            top_k_candidates = None
            if metadata is not None:
                top_k_candidates = metadata.get("top_k_candidates")
            if top_k_candidates is not None:
                candidate_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                for i_idx, cand in enumerate(top_k_candidates):
                    # cand is list of task indices
                    for j in cand:
                        if j < nt:
                            candidate_mask[i_idx, j] = True
                valid_mask = valid_mask & candidate_mask

            # relax rows with no valid candidates
            vm_np = valid_mask.detach().cpu().numpy()
            for r in range(vm_np.shape[0]):
                if not vm_np[r].any():
                    valid_mask[r, :] = True

            assigned = self._greedy_assign(scores_flat, valid_mask)

        # Map assignments back to loc_id
        actions = [0 for _ in range(len(g.agent_node_ids))]
        for agv_pos, agv_node_idx in enumerate(g.agent_node_ids):
            if g.node_types[agv_node_idx] != NodeType.AGV:
                continue
            try:
                pos_in_agv_list = [i for i, nid in enumerate(g.agent_node_ids) if nid == agv_node_idx][0]
            except Exception:
                continue
            task_idx = assigned[pos_in_agv_list]
            if task_idx is None or task_idx < 0:
                continue
            loc_id = g.task_loc_ids[int(task_idx)]
            actions[agv_pos] = int(loc_id)

        return actions
