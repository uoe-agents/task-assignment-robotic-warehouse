import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
import pyastar2d
from gymnasium import spaces
from tarware.definitions import (Action, AgentType, Direction,
                                 RewardType, CollisionLayers)
from tarware.spaces import observation_map
from tarware.utils import find_sections, get_next_micro_action

_FIXING_CLASH_TIME = 4
_STUCK_THRESHOLD = 5

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y

class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type
        self.target = 0

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir

class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)

class StuckCounter:
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.count = 0

    def update(self, new_position: Tuple[int, int]):
        if new_position == self.position:
            self.count += 1
        else:
            self.count = 0
            self.position = new_position

    def reset(self, position=None):
        self.count = 0
        if position:
            self.position = position

class Warehouse(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        num_agvs: int,
        num_pickers: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        normalised_coordinates: bool=False,
        observation_type: str = "global",
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param num_agvs: Number of spawned and controlled agv
        :type num_agvs: int
        :param num_pickers: Number of spawned and controlled pickers
        :type num_pickers: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param observation_type: Specifies type of observations
        :type normalised_coordinates: str
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers

        self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        # If no Pickers are generated, AGVs can perform picks independently
        if num_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(num_agvs)] + [AgentType.PICKER for _ in range(num_pickers)]
        else:
            self._agent_types = [AgentType.AGENT for _ in range(self.num_agents)]

        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.action_size = len(self.action_id_to_coords_map) + 1
        self.action_space = spaces.Tuple(tuple(self.num_agents * [spaces.Discrete(self.action_size)]))

        self.observation_space_mapper = observation_map[observation_type](
            self.num_agvs,
            self.num_pickers,
            self.grid_size,
            len(self.action_id_to_coords_map)-len(self.goals),
            normalised_coordinates,
        )
        self.observation_space = spaces.Tuple(tuple(self.observation_space_mapper.ma_spaces))

        self.request_queue_size = request_queue_size
        self.request_queue = []
        self.rack_groups = find_sections(list([loc for loc in self.action_id_to_coords_map.values() if (loc[1], loc[0]) not in self.goals]))
        self.agents: List[Agent] = []
        self.stuck_counters = []
        self.renderer = None

    @property
    def targets_agvs(self):
        return [agent.target for agent in self.agents[:self.num_agvs]]

    @property
    def targets_pickers(self):
        return [agent.target for agent in self.agents[self.num_agvs:]]

    def _make_layout_from_params(self, shelf_columns: int, shelf_rows: int, column_height: int) -> None:
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        self._bottom_rows = 2
        self._highway_lanes = 2
        self.column_width = 2
        self.column_height = column_height
        self.grid_size = (
            self._highway_lanes + (self.column_height + self._highway_lanes) * shelf_rows  + self._bottom_rows + 1,
            self._highway_lanes + (self.column_width  + self._highway_lanes) * shelf_columns,
        )
        self.grid = np.zeros((len(CollisionLayers), *self.grid_size), dtype=np.int32)

        def get_highway_lanes_indices(axis_size, step):
            return [
                i + j
                for i in range(
                    0, axis_size, step + self._highway_lanes
                )
                for j in range(self._highway_lanes)
            ]

        highway_ys = get_highway_lanes_indices(self.grid_size[0], self.column_height)
        highway_xs = get_highway_lanes_indices(self.grid_size[1], self.column_width)

        def highway_func(x, y):
            return x in highway_xs or y in highway_ys or y >= self.grid_size[0] - 1 - self._bottom_rows

        self.goals = [
            (i, self.grid_size[0] - 1)
            for i in range(self.grid_size[1]) if not i in highway_xs
        ]
        self.num_goals = len(self.goals)

        self.highways = np.zeros(self.grid_size, dtype=np.int32)
        self.action_id_to_coords_map = {i+1: (x, y) for i, (y, x) in enumerate(self.goals)}
        item_loc_index=len(self.action_id_to_coords_map)+1
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)
                if not highway_func(x, y) and (x, y) not in self.goals:
                    self.action_id_to_coords_map[item_loc_index] = (y, x)
                    item_loc_index+=1

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def find_path(self, start, goal: Tuple[int], agent: Tuple[int], care_for_agents: bool = True) -> List[Tuple[int]]:
        """
        Constructs a path from start to goal using the A* algorithm contidioned on a grid that integrates both the
        environment specific (shelf) and the presence of other agents as obstacles.

        If `care_for_agents` is True, the grid is adjusted to consider other agents as obtacles. However, we avoid
        situatiosns where the paths is invalidated by agents of other types waiting to cooperate with the current.
        For Pickers, the grid is further modified to ensure they can only travel through designated highways and
        access goal locations.

        Parameters:
        - care_for_agents (bool): Whether to consider other agents in the grid.
        - agent (Agent): The agent for which the path is being calculated.
        - start (tuple): The starting coordinates (x, y) of the agent.
        - goal (tuple): The goal coordinates (x, y) for the agent.

        Returns:
        - List of tuples representing the path from start to goal, or an empty list if no path is found.
        """
        grid = np.zeros(self.grid_size)
        if care_for_agents:
            grid += self.grid[CollisionLayers.AGVS]
            grid += self.grid[CollisionLayers.PICKERS]
        # Agents should start a path regardless if some others are waiting around the target location
        grid[goal[0], goal[1]] = 0

        if agent.type == AgentType.PICKER:
            # Pickers can only travel through the highway, but can access goal locations
            grid += (1-self.highways)
            grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])
            for i in range(self.grid_size[1]):
                grid[self.grid_size[0] - 1, i] = 1

        # Ban Pickers crossing through racks if adjacent target location is chosen and force them thake the long way around.
        start_fix = (0, 0)
        if agent.type == AgentType.PICKER and  ((not self._is_highway(start[1], start[0])) and goal[0] == start[0] and abs(goal[1] - start[1]) == 1):
            if self._is_highway(start[1] - 1, start[0]):
                start_fix = (0, - 1)
            if self._is_highway(start[1] + 1, start[0]):
                start_fix = (0, 1)
            grid[start[0], start[1]] = 1

        grid[start[0]+start_fix[0], start[1]+start_fix[1]] = 0
        grid = [list(map(int, l)) for l in (grid!=0)]
        grid = np.array(grid, dtype=np.float32)
        grid[np.where(grid == 1)] = np.inf
        grid[np.where(grid == 0)] = 1
        astar_path = pyastar2d.astar_path(grid, np.add(start, start_fix), goal, allow_diagonal=False) # returns None if cant find path
        # astar_path = astar_path[int(np.where(astar_path == np.add(start, start_fix))[0][0]):]
        if astar_path is not None:
            astar_path = [tuple(x) for x in list(astar_path)] # convert back to other format
            astar_path = astar_path[1 - int(grid[start[0], start[1]] > 1):]

        if astar_path:
            return [(x, y) for y, x in astar_path]
        else:
            return []

    def _recalc_grid(self) -> None:
        self.grid.fill(0)

        carried_shelf_ids = {agent.carrying_shelf.id for agent in self.agents if agent.carrying_shelf}
        for shelf in self.shelfs:
            if shelf.id not in carried_shelf_ids:
                self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] = shelf.id
        for agent in self.agents:
            layer = CollisionLayers.PICKERS if agent.type == AgentType.PICKER else CollisionLayers.AGVS
            self.grid[layer, agent.y, agent.x] = agent.id
            if agent.carrying_shelf:
                 self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = agent.carrying_shelf.id

    def get_carrying_shelf_information(self):
        return [agent.carrying_shelf != None for agent in self.agents[:self.num_agvs]]

    def get_shelf_request_information(self) -> np.ndarray[int]:
        request_item_map = np.zeros(len(self.shelfs))
        requested_shelf_ids = [shelf.id for shelf in self.request_queue]
        for id_, coords in self.action_id_to_coords_map.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] in requested_shelf_ids:
                    request_item_map[id_ - len(self.goals) - 1] = 1
        return request_item_map

    def get_empty_shelf_information(self) -> np.ndarray[int]:
        empty_item_map = np.zeros(len(self.shelfs))
        for id_, coords in self.action_id_to_coords_map.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] == 0 and (
                    self.grid[CollisionLayers.CARRIED_SHELVES, coords[0], coords[1]] == 0
                    or self.agents[
                        self.grid[CollisionLayers.AGVS, coords[0], coords[1]] - 1
                    ].req_action
                    not in [Action.NOOP, Action.TOGGLE_LOAD]
                ):
                    empty_item_map[id_ - len(self.goals) - 1] = 1
        return empty_item_map

    def attribute_macro_actions(self, macro_actions: List[int]) -> Tuple[int, int]:
        agvs_distance_travelled = 0
        pickrs_distance_travelled = 0
        # Logic for Macro Actions
        for agent, macro_action in zip(self.agents, macro_actions):
            # Initialize action for step
            agent.req_action = Action.NOOP
            # Collision avoidance logic
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
            if not agent.busy:
                agent.target = 0
                if macro_action != 0:
                    agent.path = self.find_path((agent.y, agent.x), self.action_id_to_coords_map[macro_action], agent, care_for_agents=False)
                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
            else:
                # Check if agent finished the given path, if not continue the path
                if agent.path == []:
                    if agent.type in [AgentType.AGV, AgentType.AGENT]:
                        agent.req_action = Action.TOGGLE_LOAD
                    if agent.type == AgentType.PICKER:
                        agent.busy = False
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    agvs_distance_travelled += int(agent.type == AgentType.AGV)
                    pickrs_distance_travelled += int(agent.type == AgentType.PICKER)
                if len(agent.path) == 1:
                    # If agent is at the end of a path and carrying a shelf and the target location is already occupied, restart agent
                    if agent.carrying_shelf and self.grid[CollisionLayers.SHELVES, agent.path[-1][1], agent.path[-1][0]]:
                        agent.req_action = Action.NOOP
                        agent.busy = False
                    # Logic for Pickers to load shelves if AGV is present at location or wait otherwise
                    if agent.type == AgentType.PICKER:
                        if (
                            self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]] == 0
                            or self.agents[self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]]- 1].req_action != Action.TOGGLE_LOAD
                        ):
                            agent.req_action = Action.NOOP
                        elif (
                            self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]] != 0
                            and self.agents[self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]] - 1].req_action == Action.TOGGLE_LOAD
                            ):
                            self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
        return agvs_distance_travelled, pickrs_distance_travelled

    def resolve_move_conflict(self, agent_list):
        commited_agents = set()
        G = nx.DiGraph()
        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)
            G.add_edge(start, target)
        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[CollisionLayers.AGVS, start_node[1], start_node[0]]
                    # action = self.agents[agent_id - 1].req_action
                    # print(f"{agent_id}: C {cycle} {action}")
                    if agent_id > 0:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[CollisionLayers.PICKERS, start_node[1], start_node[0]]
                    if picker_id > 0:
                        commited_agents.add(picker_id)
                        continue
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[CollisionLayers.AGVS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[CollisionLayers.PICKERS, y, x]
                    if picker_id:
                        commited_agents.add(picker_id)
        clashes = 0
        for agent in agent_list:
            for other in agent_list:
                if agent.id != other.id:
                    agent_new_x, agent_new_y = agent.req_location(self.grid_size)
                    other_new_x, other_new_y = other.req_location(self.grid_size)
                    # Clash fixing logic
                    if agent.path and ((agent_new_x, agent_new_y) in [(other.x, other.y), (other_new_x, other_new_y)]):
                        # If we are in a rack and one of the agents is a picker we ignore clashses, assumed behaviour is Picker is loading
                        if not self._is_highway(agent_new_x, agent_new_y) and (agent.type == AgentType.PICKER or other.type == AgentType.PICKER) and agent.type != other.type:
                            # Allow Pickers to step over AGVs (if no other Picker at that shelf location) or AGVs to step over Pickers (if no other AGV at that shelf location)
                            if ((agent.type == AgentType.PICKER and self.grid[CollisionLayers.PICKERS, agent_new_y, agent_new_x] in [0, agent.id])
                                or (agent.type == AgentType.AGV and self.grid[CollisionLayers.AGVS, agent_new_y, agent_new_x] in [0, agent.id])):
                                commited_agents.add(agent.id)
                                continue
                        # If the agent's next action bumps it into another agent
                        if (agent_new_x, agent_new_y) == (other.x, other.y):
                            agent.req_action = Action.NOOP # Stop the action
                            # Check if the clash is not solved naturaly by the other agent moving away
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and not other.req_action in (Action.LEFT, Action.RIGHT):
                                if other.fixing_clash == 0:# If the others are not already fixing the clash
                                    clashes+=1
                                    agent.fixing_clash = _FIXING_CLASH_TIME # Agent start time for clash fixing
                                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                    if new_path != []: # If the agent can find an alternative path, assign it if not let the other solve the clash
                                        agent.path = new_path
                                    else:
                                        agent.fixing_clash = 0
                        elif (agent_new_x, agent_new_y) == (other_new_x, other_new_y) and (agent_new_x, agent_new_y) != (agent.x, agent.y):
                            # If the agent's next action bumps it into another agent position after they take actions simultaneously
                            if agent.fixing_clash == 0 and other.fixing_clash == 0:
                                agent.req_action = Action.NOOP # If the agent's actions leads them in the position of another STOP
                                agent.fixing_clash = _FIXING_CLASH_TIME  # Agent wait one step while the other moves into place

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents
        for agent in failed_agents:
            agent.req_action = Action.NOOP
        return clashes

    def resolve_stuck_agents(self) -> None:
        # This can happen when their goal is occupied after reaching their last step/re-calculating a path
        overall_stucks = 0
        moving_agents = [
            agent
            for agent in self.agents
            if agent.busy
            and agent.req_action not in (Action.LEFT, Action.RIGHT) # Don't count changing directions
            and (agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals) # Don't count loading or changing directions / if at goal
        ]
        for agent in moving_agents:
            agent_stuck_count = self.stuck_counters[agent.id - 1]
            agent_stuck_count.update((agent.x, agent.y))
            if _STUCK_THRESHOLD < agent_stuck_count.count < _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                agent.req_action = Action.NOOP
                if agent.path:
                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1], agent.path[-1][0]), agent)
                    # Picker should wait for AGV to arrive at destination regardless of stuck count
                    if new_path:
                        agent.path = new_path
                        if len(agent.path) == 1:
                            continue
                        agent_stuck_count.reset((agent.x, agent.y))
                        continue
                else:
                    overall_stucks += 1
                    agent.busy = False
                    agent_stuck_count.reset()
            if agent_stuck_count.count > _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                overall_stucks += 1
                agent_stuck_count.reset((agent.x, agent.y))
                agent.req_action = Action.NOOP
                agent.busy = False
        return overall_stucks

    def _execute_forward(self, agent: Agent) -> None:
        agent.x, agent.y = agent.req_location(self.grid_size)
        agent.path = agent.path[1:]
        if agent.carrying_shelf:
            agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y

    def _execute_rotation(self, agent: Agent) -> None:
        agent.dir = agent.req_direction()

    def _execute_load(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
        picker_id = self.grid[CollisionLayers.PICKERS, agent.y, agent.x]
        if shelf_id:
            if (
                (agent.type == AgentType.AGV and picker_id)
                or agent.type == AgentType.AGENT
            ):
                agent.carrying_shelf = self.shelfs[shelf_id - 1]
                self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = 0
                self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = shelf_id
                agent.busy = False
                # Reward picker for loading
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 0.5
                elif self.reward_type == RewardType.INDIVIDUAL:
                    if agent.type == AgentType.AGENT:
                        rewards[agent.id - 1] += 0.1
                    else:
                        rewards[picker_id - 1] += 0.1
        else:
            agent.busy = False
        return rewards

    def _execute_unload(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        if (agent.x, agent.y) in self.goals or self.grid[CollisionLayers.SHELVES, agent.y, agent.x] != 0:
            agent.busy = False
            return rewards
        picker_id = self.grid[CollisionLayers.PICKERS, agent.y, agent.x]
        if not self._is_highway(agent.x, agent.y):
            if (
                (agent.type == AgentType.AGV and picker_id)
                or agent.type == AgentType.AGENT
            ):
                self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = agent.carrying_shelf.id
                self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
                agent.carrying_shelf = None
                agent.busy = False
                agent.has_delivered = False
                # Reward picker for unloading
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 0.5
                elif self.reward_type == RewardType.INDIVIDUAL:
                    if agent.type == AgentType.AGENT:
                        rewards[agent.id - 1] += 0.1
                    else:
                        rewards[picker_id - 1] += 0.1
        return rewards

    def execute_micro_actions(self, rewards: np.ndarray[int]) -> np.ndarray[int]:
        for agent in self.agents:
            if agent.req_action == Action.FORWARD:
                self._execute_forward(agent)
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                self._execute_rotation(agent)
            elif agent.req_action == Action.TOGGLE_LOAD:
                if not agent.carrying_shelf:
                    rewards = self._execute_load(agent, rewards)
                else:
                    rewards = self._execute_unload(agent, rewards)
        return rewards

    def process_shelf_deliveries(self, rewards: np.ndarray[int]) -> np.ndarray[int]:
        shelf_deliveries = 0
        for y, x in self.goals:
            shelf_id = self.grid[CollisionLayers.CARRIED_SHELVES, x, y]
            if not shelf_id  or self.shelfs[shelf_id - 1] not in self.request_queue:
                continue
            # Remove shelf from request queue and add replacement
            carried_shels = [agent.carrying_shelf for agent in self.agents if agent.carrying_shelf]
            new_shelf_candidates = list(set(self.shelfs) - set(self.request_queue) - set(carried_shels)) # sort so np.random with seed is repeatable
            new_shelf_candidates.sort(key = lambda x: x.id)
            new_request = np.random.choice(new_shelf_candidates)
            self.request_queue[self.request_queue.index(self.shelfs[shelf_id - 1])] = new_request

            agent = self.agents[self.grid[CollisionLayers.AGVS, x, y] - 1]
            if not agent.has_delivered:
                agent.has_delivered = True
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 1
                elif self.reward_type == RewardType.INDIVIDUAL:
                    rewards[agent.id - 1] += 1
            shelf_deliveries += 1

        if shelf_deliveries:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1

        return rewards, shelf_deliveries

    def reset(self, seed=None, options=None)-> Tuple:
        # Reset counters
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # Set seed
        self.seed(seed)

        # Make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]
        self._higway_locs = np.array([(y, x) for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            ) if self._is_highway(x, y)])

        # Spawn agents on higwahy locations
        agent_loc_ids = np.random.choice(
            np.arange(len(self._higway_locs)),
            size=self.num_agents,
            replace=False,
        )
        agent_locs = [self._higway_locs[agent_loc_ids, 0], self._higway_locs[agent_loc_ids, 1]]
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.num_agents)
        self.agents = [
            Agent(x, y, dir_, agent_type = agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]

        self.stuck_counters = [StuckCounter((agent.x, agent.y)) for agent in self.agents]
        self._recalc_grid()

        self.request_queue = list(
            np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )
        self.observation_space_mapper.extract_environment_info(self)
        return tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])

    def step(
        self, macro_actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], Dict]:
        # Attribute macro actions to agents and resolve conflicts
        agvs_distance_travelled, pickers_distance_travelled = self.attribute_macro_actions(macro_actions)
        clashes_count = self.resolve_move_conflict(self.agents)
        # Restart agents if they are stuck at the same position
        stucks_count = self.resolve_stuck_agents()

        rewards = np.zeros(self.num_agents)
        # Apply penalty for inactivity
        rewards -= 0.001
        # Execute micro actions
        rewards = self.execute_micro_actions(rewards)
        # Process shelf deliveries
        rewards, shelf_deliveries = self.process_shelf_deliveries(rewards)

        self._recalc_grid()
        self._cur_steps += 1
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            terminateds = truncateds = self.num_agents * [True]
        else:
            terminateds = truncateds =  self.num_agents * [False]

        self.observation_space_mapper.extract_environment_info(self)
        new_obs = tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])
        info = self._build_info(
            agvs_distance_travelled,
            pickers_distance_travelled,
            clashes_count,
            stucks_count,
            shelf_deliveries,
        )
        return new_obs, list(rewards), terminateds, terminateds, info

    def _build_info(
        self,
        agvs_distance_travelled: int,
        pickers_distance_travelled: int,
        clashes_count: int,
        stucks_count:  int,
        shelf_deliveries: int,
    ) -> Dict[str, np.ndarray]:
        info = {}
        agvs_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[:self.num_agvs]])
        pickers_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[self.num_agvs:]])
        info["vehicles_busy"] = [agent.busy for agent in self.agents]
        info["shelf_deliveries"] = shelf_deliveries
        info["clashes"] = clashes_count
        info["stucks"] = stucks_count
        info["agvs_distance_travelled"] = agvs_distance_travelled
        info["pickers_distance_travelled"] = pickers_distance_travelled
        info["agvs_idle_time"] = agvs_idle_time
        info["pickers_idle_time"] = pickers_idle_time
        return info

    def compute_valid_action_masks(self, pickers_to_agvs=True, block_conflicting_actions=True):
        requested_items = self.get_shelf_request_information()
        empty_items = self.get_empty_shelf_information()
        carrying_shelf_info = self.get_carrying_shelf_information()
        targets_agvs = [target - len(self.goals) - 1 for target in self.targets_agvs if target > len(self.goals)]
        targets_pickers = [target - len(self.goals) - 1 for target in self.targets_pickers if target > len(self.goals)]
        # Compute valid location list for AGVs
        valid_location_list_agvs = np.array([
            empty_items if carrying_shelf else requested_items for carrying_shelf in carrying_shelf_info
        ])
        # Compute valid location list for Pickers
        if pickers_to_agvs:
            valid_location_list_pickers = np.zeros(len(self.action_id_to_coords_map) - len(self.goals))
            valid_location_list_pickers[targets_agvs] = 1
        else:
            valid_location_list_pickers = requested_items
        # Mask out conflicting actions for agents of the same type
        if block_conflicting_actions:
            valid_location_list_agvs[:, targets_agvs] = 0
            valid_location_list_pickers[targets_pickers] = 0
        valid_action_masks = np.ones((self.num_agents, self.action_size))
        valid_action_masks[:self.num_agvs,  1 + len(self.goals):] = valid_location_list_agvs
        valid_action_masks[:self.num_agvs,  1 : 1 + len(self.goals)] = np.repeat(np.expand_dims(np.array(carrying_shelf_info), 1), len(self.goals), axis=1)
        valid_action_masks[self.num_agvs:,  1 + len(self.goals):] = valid_location_list_pickers
        valid_action_masks[self.num_agvs:,  1 : 1 + len(self.goals)] = 0
        return valid_action_masks

    def render(self, mode="human"):
        if not self.renderer:
            from tarware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
