"""Graph helpers and small utilities.

Helpers to convert between `loc_id` and `(y,x)` coords and a tiny per-step
pathfinding cache. These are intentionally lightweight to avoid pulling heavy
dependencies into the core builder/policy code.

Conventions:
- Coordinates returned by `loc_id_to_yx` are in `(y, x)` format.
"""

from typing import Any, Dict, Optional, Tuple


def loc_id_to_yx(env: Any, loc_id: int) -> Tuple[int, int]:
    """Return (y, x) coordinates for a given `loc_id`.

    Raises `KeyError` if the loc_id is unknown in `env.action_id_to_coords_map`.
    """
    coords = env.action_id_to_coords_map.get(int(loc_id))
    if coords is None:
        raise KeyError(f"loc_id {loc_id} not found in env.action_id_to_coords_map")
    return coords


def yx_to_loc_id(env: Any, yx: Tuple[int, int]) -> Optional[int]:
    """Try to map `(y,x)` -> loc_id.

    The environment historically mixes `(x,y)` and `(y,x)` in different
    places; this helper therefore tries both orders to be robust.
    Returns `None` when no mapping is found.
    """
    coords_map: Dict[Tuple[int, int], int] = {coords: lid for lid, coords in env.action_id_to_coords_map.items()}
    # Try direct
    lid = coords_map.get(yx)
    if lid is not None:
        return int(lid)
    # Try swapped
    swapped = (yx[1], yx[0])
    lid = coords_map.get(swapped)
    if lid is not None:
        return int(lid)
    return None


class PathCache:
    """Tiny dict-like cache for pathfinding results inside a single step.

    Usage: create one instance per builder `build()` call and clear it at the end
    of the step (or let it be garbage-collected). Not intended as a global
    persistent cache. Keys include agent type to avoid incorrectly sharing
    paths between agent classes (pickers vs agvs).
    """

    def __init__(self):
        self._cache = {}

    def get(self, start: Tuple[int, int], goal: Tuple[int, int], care_for_agents: bool, agent_type: Optional[str] = None):
        return self._cache.get((start, goal, bool(care_for_agents), agent_type))

    def set(self, start: Tuple[int, int], goal: Tuple[int, int], care_for_agents: bool, value, agent_type: Optional[str] = None):
        self._cache[(start, goal, bool(care_for_agents), agent_type)] = value

    def clear(self):
        self._cache.clear()
