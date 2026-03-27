"""Graph builder placeholder."""

from __future__ import annotations

from typing import Any

from .schema import GraphState


class GraphBuilder:
    def build(self, env: Any) -> GraphState:
        raise NotImplementedError("GraphBuilder.build is not implemented yet.")
