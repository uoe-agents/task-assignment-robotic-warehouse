from __future__ import annotations
"""
GNN Feature Extractor for Stable Baselines3

This module provides a custom feature extractor for reinforcement learning agents using
Stable Baselines3. It processes graph-structured observations (nodes, edges, and masks)
through a Graph Neural Network (GNN) to generate fixed-size embeddings for policy and
value networks.

The extractor handles batched graph data by:
- Extracting node embeddings from a GNN encoder
- Separating embeddings for AGVs (Autonomous Guided Vehicles) and tasks
- Computing mean pooling of subsets to create fixed-dimensional representations
- Combining global graph statistics with pooled node embeddings

Classes:
    GnnFeatureExtractor: Main feature extractor class that converts dictionary observations
                        containing graph data into dense tensor representations compatible
                        with SB3 algorithms.
"""

from typing import Dict

import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tarware_ext.graphs.gnn_minimal import GraphBatch, GnnAssignmentModel


class GnnFeatureExtractor(BaseFeaturesExtractor):
    """SB3 features extractor that runs GNN over padded graph dict observations."""

    def __init__(
        self,
        observation_space,
        emb_dim: int = 64,
        gnn_layers: int = 2,
        dropout: float = 0.0,
        architecture: str = "sage",
    ) -> None:
        features_dim = int(emb_dim * 3 + 3)
        super().__init__(observation_space, features_dim=features_dim)

        node_in_dim = int(observation_space.spaces["node_features"].shape[-1])
        edge_dim = int(observation_space.spaces["edge_attr"].shape[-1])

        self.emb_dim = int(emb_dim)
        self.model = GnnAssignmentModel(
            node_in_dim=node_in_dim,
            emb_dim=self.emb_dim,
            edge_dim=edge_dim,
            gnn_layers=int(gnn_layers),
            dropout=float(dropout),
            decoder="dot",
            architecture=str(architecture).strip().lower(),
        )

    def _safe_mean(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((self.emb_dim,), device=x.device, dtype=x.dtype)
        return x.mean(dim=0)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass that processes a batch of graph observations and extracts node embeddings.
        Extracts node features, edges, and graph structure from observations. For each graph in the batch:
        - Encodes node features using a GNN encoder
        - Separates embeddings for AGVs (Autonomous Guided Vehicles) and tasks
        - Aggregates embeddings to produce graph-level, AGV-level, and task-level representations
        - Concatenates these with graph statistics (node, edge, and task counts)
        Args:
            obs: Dictionary containing:
                - node_features: Node feature matrix
                - edge_index: Edge connectivity indices
                - edge_attr: Edge attributes
                - action_mask: Valid action mask for AGVs
                - n_nodes: Number of nodes per graph
                - n_edges: Number of edges per graph
                - n_tasks: Number of tasks per graph
        Returns:
            torch.Tensor: Stacked embeddings of shape (batch_size, embedding_dim) with NaN values replaced by zeros.
        """
        
        
        node_features = obs["node_features"]
        edge_index_all = obs["edge_index"]
        edge_attr_all = obs["edge_attr"]
        action_mask_all = obs["action_mask"]
        n_nodes_all = obs["n_nodes"]
        n_edges_all = obs["n_edges"]
        n_tasks_all = obs["n_tasks"]

        batch_size = int(node_features.shape[0])
        batch_embeddings = []

        count = 0
        for b in range(batch_size):
            n_nodes = int(n_nodes_all[b].reshape(-1)[0].item())
            n_edges = int(n_edges_all[b].reshape(-1)[0].item())
            n_tasks = int(n_tasks_all[b].reshape(-1)[0].item())

            x = node_features[b, :n_nodes, :]
            edge_index = edge_index_all[b, :, :n_edges].to(dtype=torch.int64)
            edge_attr = edge_attr_all[b, :n_edges, :]

            n_tasks = max(0, min(n_tasks, n_nodes))
            n_agent_nodes = max(0, n_nodes - n_tasks)
            agent_node_ids = torch.arange(n_agent_nodes, device=x.device, dtype=torch.int64)
            task_node_ids = torch.arange(n_agent_nodes, n_nodes, device=x.device, dtype=torch.int64)

            agv_node_ids = agent_node_ids
            if n_agent_nodes > 0 and x.shape[1] > 4:
                agv_mask = x[:n_agent_nodes, 4] > 0.5
                agv_node_ids = torch.nonzero(agv_mask, as_tuple=False).flatten().to(dtype=torch.int64)

            n_agvs = int(agv_node_ids.shape[0])
            agv_action_mask = action_mask_all[b, :n_agvs, :n_tasks].to(dtype=torch.bool)

            graph = GraphBatch(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                agent_node_ids=agent_node_ids,
                task_node_ids=task_node_ids,
                agv_node_ids=agv_node_ids,
                agv_action_mask=agv_action_mask,
            )

            h = self.model.encoder(graph.x, graph.edge_index, graph.edge_attr)
            agv_emb = h[graph.agv_node_ids] if graph.agv_node_ids.numel() > 0 else h.new_zeros((0, self.emb_dim))
            task_emb = h[graph.task_node_ids] if graph.task_node_ids.numel() > 0 else h.new_zeros((0, self.emb_dim))

            graph_emb = self._safe_mean(h)
            agv_pool = self._safe_mean(agv_emb)
            task_pool = self._safe_mean(task_emb)

            counts = torch.tensor(
                [float(n_nodes), float(n_edges), float(n_tasks)],
                device=h.device,
                dtype=h.dtype,
            )
            z = torch.cat([graph_emb, agv_pool, task_pool, counts], dim=0)
            if count == 5:
                print(f"first 5 dimensions of graph_emb: {z[:5]}")
                count = 0
            count += 1
            batch_embeddings.append(z)

        out = torch.stack(batch_embeddings, dim=0)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
