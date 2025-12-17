from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

import numpy as np
import time

from tarware.warehouse import Agent


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


def single_robot_heuristic_episode(env, render: bool = False, seed=None):
    """
    A variant of `tarware.heuristic.heuristic_episode` that assumes a *single robot type*
    which can pick up and deliver shelves (i.e. no Picker coordination).

    High-level behavior (FIFO over request queue):
    - Assign the closest available robot to each requested shelf.
    - After loading, deliver to the closest goal location.
    - After delivery, return the shelf to the closest *empty* shelf location.
    """

    # non_goal_location_ids corresponds to the item ordering in `get_empty_shelf_information`
    non_goal_location_ids = []
    for id_, coords in env.action_id_to_coords_map.items():
        if (coords[1], coords[0]) not in env.goals:
            non_goal_location_ids.append(id_)
    non_goal_location_ids = np.array(non_goal_location_ids)

    location_map = env.action_id_to_coords_map
    _ = env.reset(seed=seed)
    done = False
    all_infos = []
    timestep = 0

    robots = env.agents
    coords_original_loc_map = {v: k for k, v in env.action_id_to_coords_map.items()}

    assigned_robots: dict[Agent, Mission] = OrderedDict({})
    assigned_items: dict[Agent, int] = OrderedDict({})

    global_episode_return = 0
    episode_returns = np.zeros(env.num_agents)

    while not done:
        time.sleep(0.1)
        request_queue = env.request_queue  # FIFO list of shelf entities to be picked next
        goal_locations = env.goals  # (x, y) format in the env; we swap to (y, x) for A*
        actions = {k: 0 for k in robots}  # default to no-op

        # [None -> PICKING] assign closest available robot to each shelf in request queue.
        for item in request_queue:
            if item.id in assigned_items.values():
                continue

            available = [a for a in robots if not a.busy and not a.carrying_shelf]
            available = [a for a in available if a not in assigned_robots]

            if not available:
                continue

            shortest_paths = [
                env.find_path((a.y, a.x), (item.y, item.x), a, care_for_agents=False) for a in available
            ]
            distances = [len(p) for p in shortest_paths]
            closest_robot = available[int(np.argmin(distances))]

            item_location_id = coords_original_loc_map[(item.y, item.x)]
            assigned_robots[closest_robot] = Mission(
                MissionType.PICKING, item_location_id, item.x, item.y, timestep
            )
            assigned_items[closest_robot] = item.id

        # Mission progression per robot
        for robot in robots:
            if (
                robot in assigned_robots
                and (robot.x == assigned_robots[robot].location_x)
                and (robot.y == assigned_robots[robot].location_y)
            ):
                assigned_robots[robot].at_location = True

            if robot not in assigned_robots or robot.busy:
                continue

            # [PICKING -> DELIVERING] shelf has been loaded; go to closest goal.
            if (
                assigned_robots[robot].mission_type == MissionType.PICKING
                and assigned_robots[robot].at_location
                and robot.carrying_shelf
            ):
                goal_paths = [
                    env.find_path((robot.y, robot.x), (y, x), robot, care_for_agents=False)
                    for (x, y) in goal_locations
                ]
                goal_distances = [len(p) for p in goal_paths]
                closest_goal = goal_locations[int(np.argmin(goal_distances))]  # (x, y)
                goal_location_id = coords_original_loc_map[(closest_goal[1], closest_goal[0])]
                assigned_robots.pop(robot)
                assigned_robots[robot] = Mission(
                    MissionType.DELIVERING,
                    goal_location_id,
                    closest_goal[0],
                    closest_goal[1],
                    timestep,
                )

            # [DELIVERING -> RETURNING] shelf delivered; return it to closest empty shelf location.
            if (
                assigned_robots[robot].mission_type == MissionType.DELIVERING
                and assigned_robots[robot].at_location
                and robot.carrying_shelf
            ):
                empty_shelves = env.get_empty_shelf_information()
                empty_location_ids = list(non_goal_location_ids[empty_shelves > 0])

                # avoid multiple robots being sent to the same return target
                already_assigned_locs = [mission.location_id for mission in assigned_robots.values()]
                empty_location_ids = [loc_id for loc_id in empty_location_ids if loc_id not in already_assigned_locs]

                if empty_location_ids:
                    empty_location_yx = [location_map[i] for i in empty_location_ids]
                    empty_paths = [
                        env.find_path((robot.y, robot.x), (y, x), robot, care_for_agents=False)
                        for (y, x) in empty_location_yx
                    ]
                    empty_distances = [len(p) for p in empty_paths]
                    closest_location_id = empty_location_ids[int(np.argmin(empty_distances))]
                    closest_location_yx = location_map[closest_location_id]
                    assigned_robots.pop(robot)
                    assigned_robots[robot] = Mission(
                        MissionType.RETURNING,
                        closest_location_id,
                        closest_location_yx[1],
                        closest_location_yx[0],
                        timestep,
                    )

            # [RETURNING -> None] shelf returned to rack; clear job state.
            if (
                assigned_robots[robot].mission_type == MissionType.RETURNING
                and assigned_robots[robot].at_location
                and not robot.carrying_shelf
            ):
                assigned_robots.pop(robot)
                assigned_items.pop(robot, None)

        # Map missions to actions (macro actions are location IDs)
        for robot, mission in assigned_robots.items():
            actions[robot] = mission.location_id if not robot.busy else 0

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


