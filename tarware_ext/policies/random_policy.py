"""Random policy for sanity checks."""

from __future__ import annotations

from typing import Any


class RandomPolicy:
    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self) -> None:
        return None

    def act(self, obs: Any) -> Any:
        return self.env.action_space.sample()
