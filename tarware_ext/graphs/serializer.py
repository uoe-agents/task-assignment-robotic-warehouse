"""GraphState serialization helpers.

Provide simple converters to turn a `GraphState` into plain numpy-backed
structures (and optionally into torch tensors when `torch` is available).

This module is intentionally lightweight to avoid forcing `torch` at
development time; the `graphstate_to_torch` helper will raise a clear
ImportError if `torch` is not installed.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .schema import GraphState


def graphstate_to_dict(g: GraphState) -> Dict[str, Any]:
    """Return a JSON/NP-friendly dict representation of *g*.

    Arrays are returned as numpy arrays; lists are returned as lists. The
    resulting dict is safe to pickle or save with `numpy.savez_compressed`.
    """
    return {
        "node_features": np.array(g.node_features),
        "edge_index": np.array(g.edge_index),
        "node_types": [nt.value if hasattr(nt, "value") else str(nt) for nt in g.node_types],
        "agent_node_ids": list(g.agent_node_ids),
        "task_node_ids": list(g.task_node_ids),
        "task_loc_ids": list(g.task_loc_ids),
        "action_mask": None if g.action_mask is None else np.array(g.action_mask),
        "metadata": g.metadata if g.metadata is not None else {},
    }


def graphstate_to_torch(g: GraphState, device: str = "cpu") -> Dict[str, Any]:
    """Convert a `GraphState` to a dict of PyTorch tensors.

    Raises ImportError if `torch` is not available.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for graphstate_to_torch") from exc

    d = graphstate_to_dict(g)
    out: Dict[str, Any] = {}
    out["node_features"] = torch.as_tensor(d["node_features"], dtype=torch.float32, device=device)
    out["edge_index"] = torch.as_tensor(d["edge_index"], dtype=torch.long, device=device)
    out["agent_node_ids"] = torch.as_tensor(d["agent_node_ids"], dtype=torch.long, device=device)
    out["task_node_ids"] = torch.as_tensor(d["task_node_ids"], dtype=torch.long, device=device)
    out["task_loc_ids"] = d["task_loc_ids"]
    out["action_mask"] = None if d["action_mask"] is None else torch.as_tensor(d["action_mask"], dtype=torch.bool, device=device)
    out["metadata"] = d["metadata"]
    out["node_types"] = d["node_types"]
    return out
