"""Policy protocol."""

from __future__ import annotations

from typing import Any, Protocol


class Policy(Protocol):
    def reset(self) -> None:
        ...

    def act(self, obs: Any) -> Any:
        ...
