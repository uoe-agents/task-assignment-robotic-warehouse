"""Simple rollout buffer."""

from __future__ import annotations

from typing import Any, List


class RolloutBuffer:
    def __init__(self) -> None:
        self._items: List[Any] = []

    def add(self, item: Any) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[Any]:
        return list(self._items)
