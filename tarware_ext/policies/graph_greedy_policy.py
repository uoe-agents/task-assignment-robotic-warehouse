"""Greedy graph-based policy MVP."""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

from tarware.heuristic import Mission, MissionType
from tarware.utils.utils import flatten_list, split_list
from tarware.warehouse import Agent, AgentType


def _manhattan(a_yx: Tuple[int, int], b_yx: Tuple[int, int]) -> int:
    return abs(a_yx[0] - b_yx[0]) + abs(a_yx[1] - b_yx[1])


class DistanceMode(str, Enum):
    MANHATTAN = "manhattan"
    FIND_PATH = "find_path"


class GraphGreedyPolicy:
    """
    MVP baseline between random and heuristic.

    Behavior:
      - Assign AGVs to request_queue items greedily by distance.
      - Send pickers to the same mission locations (same as heuristic).
      - Keep mission state across steps (step-wise policy).
    """

    uses_env = True

    def __init__(
        self,
        distance_mode: DistanceMode = DistanceMode.MANHATTAN,
        active_alpha: int = 3,
        max_active_agvs: int | None = None,
    ) -> None:
        self.distance_mode = distance_mode
        self.active_alpha = active_alpha
        self.max_active_agvs = max_active_agvs
        self._initialized = False
        self._timestep = 0
        self._agents: List[Agent] = []
        self._agvs: List[Agent] = []
        self._pickers: List[Agent] = []
        self._coords_to_loc_id: Dict[Tuple[int, int], int] = {}
        self._location_map: Dict[int, Tuple[int, int]] = {}
        self._non_goal_location_ids = np.array([], dtype=int)
        self._picker_sections: List[List[Tuple[int, int]]] = []
        self._assigned_agvs: "OrderedDict[Agent, Mission]" = OrderedDict()
        self._assigned_pickers: "OrderedDict[Agent, Mission]" = OrderedDict()
        self._assigned_items: "OrderedDict[int, Agent]" = OrderedDict()

    def reset(self, env) -> None:
        self._timestep = 0
        self._agents = list(env.agents)
        self._agvs = [a for a in self._agents if a.type == AgentType.AGV]
        self._pickers = [a for a in self._agents if a.type == AgentType.PICKER]

        self._location_map = dict(env.action_id_to_coords_map)
        self._coords_to_loc_id = {coords: loc_id for loc_id, coords in self._location_map.items()}

        non_goal_ids = []
        for loc_id, coords in self._location_map.items():
            if (coords[1], coords[0]) not in env.goals:
                non_goal_ids.append(loc_id)
        self._non_goal_location_ids = np.array(non_goal_ids, dtype=int)

        sections = env.rack_groups
        picker_sections = split_list(sections, max(1, len(self._pickers)))
        picker_sections = [flatten_list(l) for l in picker_sections]
        self._picker_sections = picker_sections

        self._assigned_agvs = OrderedDict()
        self._assigned_pickers = OrderedDict()
        self._assigned_items = OrderedDict()

        if self.max_active_agvs is None:
            default_limit = max(1, self.active_alpha * max(1, len(self._pickers)))
            self.max_active_agvs = min(len(self._agvs), default_limit)

        self._initialized = True

    def _goal_loc_id(self, goal_yx: Tuple[int, int]) -> int | None:
        direct = self._coords_to_loc_id.get(goal_yx)
        if direct is not None:
            return int(direct)
        swapped = self._coords_to_loc_id.get((goal_yx[1], goal_yx[0]))
        if swapped is not None:
            return int(swapped)
        return None

    def _dist(self, env, start_yx: Tuple[int, int], goal_yx: Tuple[int, int], agent: Agent) -> int:
        if self.distance_mode == DistanceMode.FIND_PATH:
            path = env.find_path(start_yx, goal_yx, agent, care_for_agents=False)
            return len(path)
        return _manhattan(start_yx, goal_yx)

    def act(self, env) -> List[int]:
        if not self._initialized:
            self.reset(env)

        request_queue = list(env.request_queue)
        actions: Dict[Agent, int] = {a: 0 for a in self._agents}

        active_count = sum(
            1
            for a in self._agvs
            if (a in self._assigned_agvs) or a.busy or a.carrying_shelf
        )

        for item in request_queue:
            item_id = int(item.id)
            if item_id in self._assigned_items:
                continue

            if active_count >= (self.max_active_agvs or 0):
                break

            available_agvs = [
                a
                for a in self._agvs
                if (not a.busy) and (not a.carrying_shelf) and (a not in self._assigned_agvs)
            ]
            if not available_agvs:
                continue

            item_yx = (int(item.y), int(item.x))
            dists = [self._dist(env, (a.y, a.x), item_yx, a) for a in available_agvs]
            chosen_agv = available_agvs[int(np.argmin(dists))]

            loc_id = self._coords_to_loc_id.get(item_yx)
            if loc_id is None:
                continue

            self._assigned_items[item_id] = chosen_agv
            self._assigned_agvs[chosen_agv] = Mission(
                MissionType.PICKING,
                int(loc_id),
                int(item.x),
                int(item.y),
                self._timestep,
            )
            active_count += 1

        for agv in list(self._assigned_agvs.keys()):
            mission = self._assigned_agvs[agv]

            if (agv.x == mission.location_x) and (agv.y == mission.location_y):
                mission.at_location = True

            if mission.mission_type == MissionType.PICKING and mission.at_location and agv.carrying_shelf:
                goal_locations = list(env.goals)
                goal_yx_list = [(y, x) for (x, y) in goal_locations]
                dists = [self._dist(env, (agv.y, agv.x), goal_yx, agv) for goal_yx in goal_yx_list]
                closest_goal_yx = goal_yx_list[int(np.argmin(dists))]
                closest_goal_loc_id = self._goal_loc_id(closest_goal_yx)
                if closest_goal_loc_id is None:
                    continue
                mission.mission_type = MissionType.DELIVERING
                mission.location_id = int(closest_goal_loc_id)
                mission.location_y = int(closest_goal_yx[0])
                mission.location_x = int(closest_goal_yx[1])
                mission.at_location = False

            if mission.mission_type == MissionType.DELIVERING and mission.at_location and agv.carrying_shelf:
                empty_shelves = env.get_empty_shelf_information()
                empty_location_ids = list(self._non_goal_location_ids[empty_shelves > 0])

                assigned_item_loc_ids = [m.location_id for m in self._assigned_agvs.values()]
                empty_location_ids = [loc for loc in empty_location_ids if loc not in assigned_item_loc_ids]
                if not empty_location_ids:
                    continue

                empty_yx = [self._location_map[int(loc)] for loc in empty_location_ids]
                dists = [self._dist(env, (agv.y, agv.x), (y, x), agv) for (y, x) in empty_yx]
                idx = int(np.argmin(dists))
                chosen_loc_id = int(empty_location_ids[idx])
                chosen_yx = empty_yx[idx]

                mission.mission_type = MissionType.RETURNING
                mission.location_id = chosen_loc_id
                mission.location_y = int(chosen_yx[0])
                mission.location_x = int(chosen_yx[1])
                mission.at_location = False

            if mission.mission_type == MissionType.RETURNING and mission.at_location and (not agv.carrying_shelf):
                self._assigned_agvs.pop(agv, None)
                for item_id, assigned_agv in list(self._assigned_items.items()):
                    if assigned_agv == agv:
                        self._assigned_items.pop(item_id, None)

        for agv, mission in list(self._assigned_agvs.items()):
            if mission.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
                continue
            if not self._pickers:
                break

            mission_yx = (mission.location_y, mission.location_x)
            in_zone = [(mission_yx[0], mission_yx[1]) in section for section in self._picker_sections]
            if True not in in_zone:
                continue
            picker = self._pickers[in_zone.index(True)]
            if picker not in self._assigned_pickers:
                self._assigned_pickers[picker] = Mission(
                    MissionType.PICKING,
                    mission.location_id,
                    mission.location_x,
                    mission.location_y,
                    self._timestep,
                )

        for picker in list(self._pickers):
            if picker in self._assigned_pickers:
                pm = self._assigned_pickers[picker]
                if (picker.x == pm.location_x) and (picker.y == pm.location_y):
                    self._assigned_pickers.pop(picker, None)

        for agv, mission in self._assigned_agvs.items():
            actions[agv] = int(mission.location_id) if not agv.busy else 0
        for picker, mission in self._assigned_pickers.items():
            actions[picker] = int(mission.location_id)

        self._timestep += 1
        return [int(actions[a]) for a in self._agents]
