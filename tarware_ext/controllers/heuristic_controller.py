"""Step-wise mission controller with heuristic parity.
Heuristic-based mission controller for a robotic warehouse environment.

This module implements a stateful step-wise mission controller that manages
AGV (Automated Guided Vehicle) and picker agent assignments in a warehouse
setting. It mirrors the mission lifecycle from the `tarware.heuristic` module
while supporting optional RL-based overrides for AGV-to-item assignments.

Key Features:
- Maintains separate mission queues for AGVs and pickers
- Supports three mission types: PICKING, DELIVERING, and RETURNING
- Implements nearest-neighbor heuristic for AGV assignment
- Allows RL agents to override default assignments
- Tracks agent state and location mappings
- Provides assignment snapshots for external encoders

The controller operates on a step-by-step basis, updating missions based on
agent positions, availability, and completion of previous tasks.

This controller mirrors the mission semantics in `tarware.heuristic` while
allowing an optional RL override for new AGV->item assignments.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tarware.heuristic import Mission, MissionType
from tarware.utils.utils import flatten_list, split_list
from tarware.warehouse import Agent, AgentType


class HeuristicController:
    """Stateful mission controller matching the original heuristic lifecycle."""

    def __init__(self) -> None:
        """Initialize empty state for the mission controller."""
        self._initialized = False
        self._timestep = 0

        self._agents: List[Agent] = []
        self._agvs: List[Agent] = []
        self._pickers: List[Agent] = []

        self._location_map: Dict[int, Tuple[int, int]] = {}
        self._coords_to_loc_id: Dict[Tuple[int, int], int] = {}
        self._non_goal_location_ids = np.array([], dtype=int)
        self._picker_sections: List[List[Tuple[int, int]]] = []

        self._assigned_agvs: "OrderedDict[Agent, Mission]" = OrderedDict()
        self._assigned_pickers: "OrderedDict[Agent, Mission]" = OrderedDict()
        self._assigned_items: "OrderedDict[Agent, int]" = OrderedDict()

    def _unwrap_env(self, env: Any) -> Any:
        """Unwrap environment wrappers to get the base environment."""
        cand = env
        for _ in range(6):
            if hasattr(cand, "unwrapped") and getattr(cand, "unwrapped") is not cand:
                cand = getattr(cand, "unwrapped")
                continue
            if hasattr(cand, "env") and getattr(cand, "env") is not cand:
                cand = getattr(cand, "env")
                continue
            break
        return cand

    def reset(self, env: Any, seed: int | None = None) -> None:
        """Reset missions and extract agents, locations, and zones from environment."""
        _ = seed
        self._timestep = 0

        target = self._unwrap_env(env)

        self._agents = list(target.agents)
        self._agvs = [a for a in self._agents if a.type == AgentType.AGV]
        self._pickers = [a for a in self._agents if a.type == AgentType.PICKER]

        self._location_map = dict(target.action_id_to_coords_map)
        self._coords_to_loc_id = {coords: loc_id for loc_id, coords in self._location_map.items()}

        non_goal_ids = []
        for loc_id, coords in self._location_map.items():
            if (coords[1], coords[0]) not in target.goals:
                non_goal_ids.append(loc_id)
        self._non_goal_location_ids = np.array(non_goal_ids, dtype=int)

        sections = list(getattr(target, "rack_groups", []))
        if self._pickers:
            picker_sections = split_list(sections, len(self._pickers))
            self._picker_sections = [flatten_list(section) for section in picker_sections]
        else:
            self._picker_sections = []

        self._assigned_agvs = OrderedDict()
        self._assigned_pickers = OrderedDict()
        self._assigned_items = OrderedDict()

        self._initialized = True

    def _dist(self, env: Any, start_yx: Tuple[int, int], goal_yx: Tuple[int, int], agv: Agent) -> int:
        """Calculate shortest path distance between two positions."""
        try:
            path = env.find_path(start_yx, goal_yx, agv, care_for_agents=False)
        except Exception:
            path = None
        if path is None:
            return 10**9
        return len(path)

    def _available_agvs(self) -> List[Agent]:
        """Return list of idle AGVs not carrying items or already assigned."""
        return [
            agv
            for agv in self._agvs
            if (not agv.busy) and (not agv.carrying_shelf) and (agv not in self._assigned_agvs)
        ]

    def _assign_agv_to_item(self, item: Any, agv: Agent) -> bool:
        """Assign an AGV to pick up a specific item."""
        item_yx = (int(item.y), int(item.x))
        loc_id = self._coords_to_loc_id.get(item_yx)
        if loc_id is None:
            return False

        self._assigned_items[agv] = int(item.id)
        self._assigned_agvs[agv] = Mission(
            mission_type=MissionType.PICKING,
            location_id=int(loc_id),
            location_x=int(item.x),
            location_y=int(item.y),
            assigned_time=self._timestep,
        )
        return True

    def _assign_new_agv_missions(self, env: Any, rl_agv_assignments: Optional[Sequence[int]]) -> None:
        """Create new AGV picking missions, optionally using RL overrides."""
        request_queue = list(env.request_queue)

        # 1) Optional RL override: explicit AGV -> request_queue slot assignment.
        if rl_agv_assignments is not None:
            for agv_idx, choice in enumerate(rl_agv_assignments):
                if agv_idx >= len(self._agvs):
                    break

                agv = self._agvs[agv_idx]
                if int(choice) <= 0:
                    continue
                if agv not in self._available_agvs():
                    continue

                item_pos = int(choice) - 1
                if item_pos < 0 or item_pos >= len(request_queue):
                    continue

                item = request_queue[item_pos]
                if int(item.id) in self._assigned_items.values():
                    continue
                self._assign_agv_to_item(item, agv)

        # 2) Fill remaining assignments with the original nearest-AGV heuristic.
        for item in request_queue:
            item_id = int(item.id)
            if item_id in self._assigned_items.values():
                continue

            available_agvs = self._available_agvs()
            if not available_agvs:
                continue

            item_yx = (int(item.y), int(item.x))
            dists = [self._dist(env, (int(a.y), int(a.x)), item_yx, a) for a in available_agvs]
            closest_agv = available_agvs[int(np.argmin(dists))]
            self._assign_agv_to_item(item, closest_agv)

    def _update_agv_missions(self, env: Any) -> None:
        """Progress AGV missions through picking, delivering, and returning states."""
        goal_locations = list(env.goals)

        for agv in list(self._assigned_agvs.keys()):
            mission = self._assigned_agvs[agv]

            if (int(agv.x) == int(mission.location_x)) and (int(agv.y) == int(mission.location_y)):
                mission.at_location = True

            # Keep parity with the heuristic loop: busy AGVs do not get new macro-actions.
            if agv.busy:
                continue

            if mission.mission_type == MissionType.PICKING and mission.at_location and agv.carrying_shelf:
                goal_distances = []
                for (goal_x, goal_y) in goal_locations:
                    d = self._dist(env, (int(agv.y), int(agv.x)), (int(goal_y), int(goal_x)), agv)
                    goal_distances.append(d)

                if goal_distances:
                    closest_goal = goal_locations[int(np.argmin(goal_distances))]
                    goal_yx = (int(closest_goal[1]), int(closest_goal[0]))
                    goal_location_id = self._coords_to_loc_id.get(goal_yx)
                    if goal_location_id is not None:
                        self._assigned_agvs.pop(agv, None)
                        self._assigned_agvs[agv] = Mission(
                            mission_type=MissionType.DELIVERING,
                            location_id=int(goal_location_id),
                            location_x=int(closest_goal[0]),
                            location_y=int(closest_goal[1]),
                            assigned_time=self._timestep,
                        )
                        mission = self._assigned_agvs[agv]

            if mission.mission_type == MissionType.DELIVERING and mission.at_location and agv.carrying_shelf:
                empty_shelves = env.get_empty_shelf_information()
                empty_location_ids = list(self._non_goal_location_ids[empty_shelves > 0])

                assigned_item_loc_ids = [m.location_id for m in self._assigned_agvs.values()]
                empty_location_ids = [loc_id for loc_id in empty_location_ids if loc_id not in assigned_item_loc_ids]
                if not empty_location_ids:
                    continue

                empty_location_yx = [self._location_map[int(loc_id)] for loc_id in empty_location_ids]
                dists = [
                    self._dist(env, (int(agv.y), int(agv.x)), (int(y), int(x)), agv)
                    for (y, x) in empty_location_yx
                ]
                closest_idx = int(np.argmin(dists))
                closest_location_id = int(empty_location_ids[closest_idx])
                closest_yx = self._location_map[closest_location_id]

                self._assigned_agvs.pop(agv, None)
                self._assigned_agvs[agv] = Mission(
                    mission_type=MissionType.RETURNING,
                    location_id=closest_location_id,
                    location_x=int(closest_yx[1]),
                    location_y=int(closest_yx[0]),
                    assigned_time=self._timestep,
                )
                mission = self._assigned_agvs[agv]

            if mission.mission_type == MissionType.RETURNING and mission.at_location and (not agv.carrying_shelf):
                self._assigned_agvs.pop(agv, None)
                self._assigned_items.pop(agv, None)

    def _update_picker_missions(self) -> None:
        """Assign pickers to locations where AGVs are picking up items."""
        for _agv, mission in self._assigned_agvs.items():
            if mission.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
                continue
            if not self._pickers or not self._picker_sections:
                break

            mission_yx = (int(mission.location_y), int(mission.location_x))
            in_zone = [mission_yx in section for section in self._picker_sections]
            if True not in in_zone:
                continue

            picker = self._pickers[in_zone.index(True)]
            if picker in self._assigned_pickers:
                continue

            self._assigned_pickers[picker] = Mission(
                mission_type=MissionType.PICKING,
                location_id=int(mission.location_id),
                location_x=int(mission.location_x),
                location_y=int(mission.location_y),
                assigned_time=self._timestep,
            )

        for picker in list(self._pickers):
            if picker not in self._assigned_pickers:
                continue

            pm = self._assigned_pickers[picker]
            if (int(picker.x) == int(pm.location_x)) and (int(picker.y) == int(pm.location_y)):
                self._assigned_pickers.pop(picker, None)

    def _missions_to_actions(self) -> List[int]:
        """Convert mission assignments into action IDs for all agents."""
        actions: Dict[Agent, int] = {agent: 0 for agent in self._agents}

        for agv, mission in self._assigned_agvs.items():
            actions[agv] = int(mission.location_id) if not agv.busy else 0

        for picker, mission in self._assigned_pickers.items():
            actions[picker] = int(mission.location_id)

        return [int(actions[agent]) for agent in self._agents]

    def get_assignment_snapshot(self) -> Dict[str, Any]:
        """Return current assignment state for external observation encoders."""
        mission_codes = {
            MissionType.PICKING: 1.0,
            MissionType.DELIVERING: 2.0,
            MissionType.RETURNING: 3.0,
        }

        agv_is_assigned: Dict[int, float] = {}
        agv_mission_type: Dict[int, float] = {}
        for agv in self._agvs:
            mission = self._assigned_agvs.get(agv)
            agv_key = id(agv)
            agv_is_assigned[agv_key] = 1.0 if mission is not None else 0.0
            if mission is None:
                agv_mission_type[agv_key] = 0.0
            else:
                agv_mission_type[agv_key] = float(mission_codes.get(mission.mission_type, 0.0))

        assigned_item_ids = sorted({int(item_id) for item_id in self._assigned_items.values()})

        return {
            "timestep": int(self._timestep),
            "assigned_item_ids": assigned_item_ids,
            "agv_is_assigned": agv_is_assigned,
            "agv_mission_type": agv_mission_type,
        }

    def step(self, env: Any, rl_agv_assignments: Optional[Sequence[int]] = None) -> List[int]:
        """Execute one control step: update missions and return actions for all agents."""
        target = self._unwrap_env(env)

        if not self._initialized:
            self.reset(target)

        self._assign_new_agv_missions(target, rl_agv_assignments)
        self._update_agv_missions(target)
        self._update_picker_missions()
        actions = self._missions_to_actions()

        self._timestep += 1
        return actions
