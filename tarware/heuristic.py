
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

import numpy as np

from tarware.utils.utils import flatten_list, split_list
from tarware.warehouse import Agent, AgentType


class MissionType(Enum):
    PICKING = 1
    RETURNING = 2
    DELIVERING = 3

@dataclass
class Mission:
    mission_type: MissionType
    location_id: int
    location_x: int
    location_y: int
    assigned_time: int
    at_location: bool = False

def heuristic_episode(env, render=False, seed=None):
    # non_goal_location_ids corresponds to the item ordering in `get_empty_shelf_information`
    non_goal_location_ids = []
    for id_, coords in env.action_id_to_coords_map.items():
        if (coords[1], coords[0]) not in env.goals:
            non_goal_location_ids.append(id_)
    non_goal_location_ids = np.array(non_goal_location_ids)
    location_map =  env.action_id_to_coords_map
    _ = env.reset(seed=seed)
    done = False
    all_infos = []
    timestep = 0

    agents = env.agents
    agvs = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    coords_original_loc_map = {v:k for k, v in env.action_id_to_coords_map.items()}
    # split the pickers evenly into sections throughout the warehouse
    sections = env.rack_groups
    picker_sections = split_list(sections, len(pickers))
    picker_sections = [flatten_list(l) for l in picker_sections]

    assigned_agvs: dict[Agent, Mission] = OrderedDict({}) # keep track of what jobs AGVs have been assigned
    assigned_pickers: dict[Agent, Mission] = OrderedDict({}) # keep track of what jobs Pickers have been assigned
    assigned_items: dict[int, Agent] = OrderedDict({}) # keep track of which items have been picked up by an AGV (key is item.id)
    global_episode_return = 0
    episode_returns = np.zeros(env.num_agents)
    while not done:

        request_queue = env.request_queue # this is a list of locations that need to be picked up next.
        goal_locations = env.goals # (y, x) format
        actions = {k: 0 for k in agents} # default to no-op

        # [AGV None -> AGV PICKING] find closest non-busy agv agent to each item in request queue, send them there, and put the AGV in a mission queue
        for item in request_queue:

            if item.id in assigned_items.values():
                continue

            available_agvs = [a for a in agvs if not a.busy and not a.carrying_shelf]
            available_agvs = [a for a in available_agvs if not a in assigned_agvs]

            if not available_agvs:
                continue

            agv_shortest_paths = [env.find_path((a.y, a.x), (item.y, item.x), a, care_for_agents=False) for a in available_agvs]
            agv_distances = [len(p) for p in agv_shortest_paths]
            closest_agv = available_agvs[np.argmin(agv_distances)]
            item_location_id = coords_original_loc_map[(item.y, item.x)]
            if closest_agv:
                assigned_agvs[closest_agv] = Mission(MissionType.PICKING, item_location_id, item.x, item.y, timestep)
                assigned_items[closest_agv] = item.id

        for agv in agvs:

            if agv in assigned_agvs and (agv.x == assigned_agvs[agv].location_x) and (agv.y == assigned_agvs[agv].location_y):
                assigned_agvs[agv].at_location = True

            if agv not in assigned_agvs or agv.busy:
                continue

            # [AGV PICKING -> AGV DELIVERING] The shelf has been picked onto the AGV. Go to the closest goal location.
            if assigned_agvs[agv].mission_type == MissionType.PICKING and assigned_agvs[agv].at_location and agv.carrying_shelf:
                goal_shortest_paths = [env.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (x,y) in goal_locations]
                goal_distances = [len(p) for p in goal_shortest_paths] 
                closest_goal = goal_locations[np.argmin(goal_distances)] # goal locations are in (y, x) format
                goal_location_id = coords_original_loc_map[(closest_goal[1], closest_goal[0])]
                mission = assigned_agvs.pop(agv)
                assigned_agvs[agv] = Mission(MissionType.DELIVERING, goal_location_id, closest_goal[0], closest_goal[1], timestep)

            # [AGV DELIVERING -> AGV RETURNING] The shelf has been delivered to the pick station. Return to closest empty shelf.
            if assigned_agvs[agv].mission_type == MissionType.DELIVERING and assigned_agvs[agv].at_location and agv.carrying_shelf:
                empty_shelves = env.get_empty_shelf_information()
                empty_location_ids = list(non_goal_location_ids[empty_shelves > 0])
                assigned_item_loc_agvs = [mission.location_id for mission in assigned_agvs.values()]
                empty_location_ids = [loc_id for loc_id in empty_location_ids if loc_id not in assigned_item_loc_agvs]
                empty_location_yx = [location_map[i] for i in empty_location_ids]
                closest_empty_location_paths = [env.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (y,x) in empty_location_yx]
                closest_empty_location_distances = [len(p) for p in closest_empty_location_paths]
                closest_location_id = empty_location_ids[np.argmin(closest_empty_location_distances)]
                closest_location_yx = location_map[closest_location_id]
                assigned_agvs.pop(agv)
                assigned_agvs[agv] = Mission(MissionType.RETURNING, closest_location_id, closest_location_yx[1], closest_location_yx[0], timestep)

            # [AGV RETURNING -> AGV None] The item is returned to the rack. 
            if assigned_agvs[agv].mission_type == MissionType.RETURNING and assigned_agvs[agv].at_location and not agv.carrying_shelf:
                assigned_agvs.pop(agv)
                assigned_items.pop(agv)

        # Send pickers to where AGVs are going. Since assigned_agvs is ordered, the picker will prioritize the first agv
        for agv, mission in assigned_agvs.items():
            if mission.mission_type in [MissionType.PICKING, MissionType.RETURNING]:
                in_pickers_zone = [(mission.location_y, mission.location_x) in p for p in picker_sections]
                relevant_picker = pickers[in_pickers_zone.index(True)]
                if relevant_picker not in assigned_pickers.keys():
                    assigned_pickers[relevant_picker] = Mission(MissionType.PICKING, mission.location_id, mission.location_x, mission.location_y, timestep)

        # Picker has reached destination, remove its mission.
        for picker in pickers:
            if picker in assigned_pickers and (picker.x == assigned_pickers[picker].location_x) and (picker.y == assigned_pickers[picker].location_y):
                assigned_pickers[picker].at_location = True
                assigned_pickers.pop(picker)

        # Map the missions to actions
        for agv, mission in assigned_agvs.items():
            actions[agv] = mission.location_id if not agv.busy else 0
        for picker, mission in assigned_pickers.items():
            actions[picker] = mission.location_id
        # macro_action should be the index of self.action_id_to_coords_map
        if render:
            env.render(mode="human")

        _, reward, terminated, truncated, info = env.step(list(actions.values()))
        done = terminated or truncated
        episode_returns += np.array(reward, dtype=np.float64)
        global_episode_return += np.sum(reward)
        done = all(done)
        all_infos.append(info)
        timestep += 1

    return all_infos, global_episode_return, episode_returns
