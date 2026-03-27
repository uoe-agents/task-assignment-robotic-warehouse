"""Minimal pluggable GNN blocks for assignment over GraphState.

Supports architecture selection through a shared interface:
- ``sage``: GraphSAGE (PyG-backed when available)
- ``gcn``: GCN (PyG-backed when available)
- ``gat``: GAT (PyG-backed when available)

When PyG is unavailable, the module falls back to a dependency-light
``SimpleGraphSAGE`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tarware_ext.graphs.schema import GraphState, NodeType

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv

    HAS_PYG = True
except Exception:
    GATConv = None
    GCNConv = None
    SAGEConv = None
    HAS_PYG = False


@dataclass(frozen=True)
class GraphBatch:
    """Torch-ready single-graph container derived from GraphState."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor]
    agent_node_ids: torch.Tensor
    task_node_ids: torch.Tensor
    agv_node_ids: torch.Tensor
    agv_action_mask: Optional[torch.Tensor]

    @staticmethod
    def from_graph_state(gs: GraphState, device: Optional[torch.device] = None) -> "GraphBatch":
        device = device or torch.device("cpu")

        x = torch.as_tensor(gs.node_features, dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(gs.edge_index, dtype=torch.int64, device=device)

        edge_attr = None
        if isinstance(gs.metadata, dict) and gs.metadata.get("edge_attr", None) is not None:
            edge_attr = torch.as_tensor(gs.metadata["edge_attr"], dtype=torch.float32, device=device)

        agent_node_ids = torch.as_tensor(gs.agent_node_ids, dtype=torch.int64, device=device)
        task_node_ids = torch.as_tensor(gs.task_node_ids, dtype=torch.int64, device=device)
        agv_node_ids = torch.as_tensor(_resolve_agv_node_ids(gs), dtype=torch.int64, device=device)

        agv_action_mask = None
        if gs.action_mask is not None and len(gs.task_node_ids) > 0:
            agv_rows = _resolve_agv_action_mask_rows(gs)
            if agv_rows:
                agv_action_mask = torch.as_tensor(gs.action_mask[agv_rows, :], dtype=torch.bool, device=device)

        return GraphBatch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            agent_node_ids=agent_node_ids,
            task_node_ids=task_node_ids,
            agv_node_ids=agv_node_ids,
            agv_action_mask=agv_action_mask,
        )


def _is_agv_type(node_type: object) -> bool:
    if isinstance(node_type, NodeType):
        return node_type == NodeType.AGV
    return str(node_type).lower() == "agv"


def _resolve_agv_node_ids(gs: GraphState) -> list[int]:
    if isinstance(gs.metadata, dict):
        raw = gs.metadata.get("agv_agent_indices")
        if isinstance(raw, Sequence):
            out = [int(i) for i in raw if int(i) in set(int(n) for n in gs.agent_node_ids)]
            if out:
                return out

    out: list[int] = []
    for node_id in gs.agent_node_ids:
        nid = int(node_id)
        if 0 <= nid < len(gs.node_types) and _is_agv_type(gs.node_types[nid]):
            out.append(nid)
    return out


def _resolve_agv_action_mask_rows(gs: GraphState) -> list[int]:
    if gs.action_mask is None:
        return []

    agv_node_ids = _resolve_agv_node_ids(gs)
    n_rows = int(gs.action_mask.shape[0])

    if not agv_node_ids:
        return []

    # Common case for assignment builder: rows are env-agent indices == node ids.
    if max(agv_node_ids) < n_rows:
        return agv_node_ids

    # Fallback: rows aligned with agent_node_ids order.
    if n_rows == len(gs.agent_node_ids):
        pos_map = {int(nid): idx for idx, nid in enumerate(gs.agent_node_ids)}
        return [pos_map[nid] for nid in agv_node_ids if nid in pos_map]

    return []


def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Mask-aware softmax with safe fallback for fully-masked rows."""
    if mask is None:
        return torch.softmax(logits, dim=dim)

    valid_mask = mask.to(dtype=torch.bool)

    if dim != -1:
        valid_mask = valid_mask.transpose(dim, -1)
        logits = logits.transpose(dim, -1)

    # Rows with no valid action are relaxed to avoid NaNs.
    row_has_valid = valid_mask.any(dim=-1, keepdim=True)
    safe_mask = torch.where(row_has_valid, valid_mask, torch.ones_like(valid_mask, dtype=torch.bool))

    masked_logits = logits.masked_fill(~safe_mask, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)

    if dim != -1:
        probs = probs.transpose(dim, -1)
    return probs


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleGraphSAGE(nn.Module):
    """Small GraphSAGE-style encoder using edge_index and optional edge_attr."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.edge_dim = int(edge_dim)

        dims = [in_dim] + [hidden_dim] * (self.num_layers - 1) + [out_dim]
        self.msg_mlps = nn.ModuleList()
        self.upd_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            d_in = dims[layer_idx]
            d_out = dims[layer_idx + 1]
            msg_in_dim = (2 * d_in) + (self.edge_dim if self.edge_dim > 0 else 0)
            self.msg_mlps.append(MLP(msg_in_dim, hidden_dim, hidden_dim, dropout=self.dropout))
            self.upd_linears.append(nn.Linear(d_in + hidden_dim, d_out))
            self.norms.append(nn.LayerNorm(d_out))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        h = x

        if edge_index.numel() > 0:
            rev = edge_index.flip(0)
            edge_index = torch.cat([edge_index, rev], dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        for layer_idx in range(self.num_layers):
            if edge_index.numel() == 0:
                agg = torch.zeros((h.shape[0], self.msg_mlps[layer_idx].net[-1].out_features), device=h.device, dtype=h.dtype)
            else:
                src = edge_index[0]
                dst = edge_index[1]
                h_src = h[src]
                h_dst = h[dst]
                if self.edge_dim > 0 and edge_attr is not None:
                    msg_in = torch.cat([h_src, h_dst, edge_attr], dim=-1)
                else:
                    msg_in = torch.cat([h_src, h_dst], dim=-1)
                m = self.msg_mlps[layer_idx](msg_in)

                agg = torch.zeros((h.shape[0], m.shape[-1]), device=h.device, dtype=h.dtype)
                agg.index_add_(0, dst, m)
                deg = torch.zeros((h.shape[0],), device=h.device, dtype=h.dtype)
                deg.index_add_(0, dst, torch.ones((m.shape[0],), device=h.device, dtype=h.dtype))
                agg = agg / deg.clamp_min(1.0).unsqueeze(-1)

            h = self.upd_linears[layer_idx](torch.cat([h, agg], dim=-1))
            h = self.norms[layer_idx](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h


class BaseAssignmentEncoder(nn.Module):
    """Shared interface for assignment encoders returning node embeddings."""

    architecture: str = "base"

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class GraphSageEncoder(BaseAssignmentEncoder):
    """GraphSAGE encoder (PyG) with fallback to SimpleGraphSAGE."""

    architecture = "sage"

    def __init__(self, in_dim: int, emb_dim: int, gnn_layers: int, dropout: float) -> None:
        super().__init__()
        self._dropout = float(dropout)
        self._use_pyg = HAS_PYG

        if self._use_pyg:
            self.convs = nn.ModuleList()
            for layer_idx in range(int(gnn_layers)):
                d_in = in_dim if layer_idx == 0 else emb_dim
                self.convs.append(SAGEConv(d_in, emb_dim))
        else:
            self.fallback = SimpleGraphSAGE(
                in_dim=in_dim,
                hidden_dim=emb_dim,
                out_dim=emb_dim,
                edge_dim=0,
                num_layers=gnn_layers,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        if not self._use_pyg:
            return self.fallback(x, edge_index, edge_attr)

        h = x
        ei = edge_index
        if edge_index.numel() > 0:
            ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        for conv in self.convs:
            h = conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=self._dropout, training=self.training)
        return h


class GcnEncoder(BaseAssignmentEncoder):
    """GCN encoder (PyG) with fallback to SimpleGraphSAGE."""

    architecture = "gcn"

    def __init__(self, in_dim: int, emb_dim: int, gnn_layers: int, dropout: float) -> None:
        super().__init__()
        self._dropout = float(dropout)
        self._use_pyg = HAS_PYG

        if self._use_pyg:
            self.convs = nn.ModuleList()
            for layer_idx in range(int(gnn_layers)):
                d_in = in_dim if layer_idx == 0 else emb_dim
                self.convs.append(GCNConv(d_in, emb_dim))
        else:
            self.fallback = SimpleGraphSAGE(
                in_dim=in_dim,
                hidden_dim=emb_dim,
                out_dim=emb_dim,
                edge_dim=0,
                num_layers=gnn_layers,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        if not self._use_pyg:
            return self.fallback(x, edge_index, edge_attr)

        h = x
        ei = edge_index
        if edge_index.numel() > 0:
            ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        for conv in self.convs:
            h = conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=self._dropout, training=self.training)
        return h


class GatEncoder(BaseAssignmentEncoder):
    """GAT encoder (PyG) with fallback to SimpleGraphSAGE."""

    architecture = "gat"

    def __init__(self, in_dim: int, emb_dim: int, gnn_layers: int, dropout: float, heads: int = 2) -> None:
        super().__init__()
        self._dropout = float(dropout)
        self._use_pyg = HAS_PYG

        if self._use_pyg:
            self.convs = nn.ModuleList()
            for layer_idx in range(int(gnn_layers)):
                d_in = in_dim if layer_idx == 0 else emb_dim
                # concat=False keeps output dim fixed to emb_dim for interface consistency.
                self.convs.append(GATConv(d_in, emb_dim, heads=int(heads), concat=False, dropout=dropout))
        else:
            self.fallback = SimpleGraphSAGE(
                in_dim=in_dim,
                hidden_dim=emb_dim,
                out_dim=emb_dim,
                edge_dim=0,
                num_layers=gnn_layers,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        if not self._use_pyg:
            return self.fallback(x, edge_index, edge_attr)

        h = x
        ei = edge_index
        if edge_index.numel() > 0:
            ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        for conv in self.convs:
            h = conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=self._dropout, training=self.training)
        return h


def build_assignment_encoder(
    architecture: str,
    *,
    in_dim: int,
    emb_dim: int,
    gnn_layers: int,
    dropout: float,
) -> BaseAssignmentEncoder:
    arch = str(architecture).strip().lower()
    if arch in ("sage", "graphsage"):
        return GraphSageEncoder(in_dim=in_dim, emb_dim=emb_dim, gnn_layers=gnn_layers, dropout=dropout)
    if arch == "gcn":
        return GcnEncoder(in_dim=in_dim, emb_dim=emb_dim, gnn_layers=gnn_layers, dropout=dropout)
    if arch == "gat":
        return GatEncoder(in_dim=in_dim, emb_dim=emb_dim, gnn_layers=gnn_layers, dropout=dropout)
    raise ValueError("architecture must be one of: 'sage', 'gcn', 'gat'.")


class AssignmentPolicyHead(nn.Module):
    """Decoder that returns logits with shape (n_agvs, n_tasks)."""

    def __init__(self, emb_dim: int, mode: str = "dot") -> None:
        super().__init__()
        if mode not in {"dot", "bilinear"}:
            raise ValueError("mode must be 'dot' or 'bilinear'")
        self.mode = mode
        self.bilinear = nn.Bilinear(emb_dim, emb_dim, 1, bias=False) if mode == "bilinear" else None

    def forward(self, agv_emb: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        if task_emb.numel() == 0:
            return torch.empty((agv_emb.shape[0], 0), device=agv_emb.device, dtype=agv_emb.dtype)
        if self.mode == "dot":
            return agv_emb @ task_emb.t()

        a, d = agv_emb.shape
        t = task_emb.shape[0]
        agv_ex = agv_emb.unsqueeze(1).expand(a, t, d).reshape(a * t, d)
        task_ex = task_emb.unsqueeze(0).expand(a, t, d).reshape(a * t, d)
        return self.bilinear(agv_ex, task_ex).reshape(a, t)


class GnnAssignmentModel(nn.Module):
    """GraphState model: embeddings + logits/probs over AGV-task pairs."""

    def __init__(
        self,
        node_in_dim: int,
        emb_dim: int = 64,
        edge_dim: int = 2,
        gnn_layers: int = 2,
        dropout: float = 0.0,
        decoder: str = "dot",
        architecture: str = "sage",
    ) -> None:
        super().__init__()
        _ = edge_dim
        self.architecture = str(architecture).strip().lower()
        self.encoder = build_assignment_encoder(
            self.architecture,
            in_dim=node_in_dim,
            emb_dim=emb_dim,
            gnn_layers=gnn_layers,
            dropout=dropout,
        )
        self.head = AssignmentPolicyHead(emb_dim=emb_dim, mode=decoder)

    def forward(self, batch: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        agv_emb = h[batch.agv_node_ids]
        task_emb = h[batch.task_node_ids]

        logits = self.head(agv_emb, task_emb)
        probs = masked_softmax(logits, batch.agv_action_mask, dim=-1)
        return agv_emb, task_emb, logits, probs


def build_default_gnn_for_assignment(node_feature_dim: int, architecture: str = "sage") -> GnnAssignmentModel:
    return GnnAssignmentModel(
        node_in_dim=node_feature_dim,
        emb_dim=64,
        edge_dim=2,
        gnn_layers=2,
        dropout=0.0,
        decoder="dot",
        architecture=architecture,
    )
