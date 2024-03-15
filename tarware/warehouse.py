import logging

from collections import defaultdict, OrderedDict
import gym
from gym import spaces
import pyastar2d
from tarware.utils import find_sections
from enum import Enum
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import networkx as nx
import random

_COLLISION_LAYERS = 4

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
_LAYER_CARRIED_SHELFS = 2
_LAYER_PICKERS = 3

class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits

class AgentType(Enum):
    AGV = 0
    PICKER = 1
    AGENT = 2

class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    TOGGLE_LOAD = 4

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

DIR_TO_ENUM = {
    (0, -1): Direction.UP,
    (0, 1): Direction.DOWN,
    (-1, 0): Direction.LEFT,
    (1, 0): Direction.RIGHT,
    }

def get_next_micro_action(agent_x, agent_y, agent_direction, target):
    target_x, target_y = target
    target_direction =  DIR_TO_ENUM[(target_x - agent_x, target_y - agent_y)]

    turn_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    # Find the indices of the source and target directions in the turn order
    source_index = turn_order.index(agent_direction)
    target_index = turn_order.index(target_direction)

    # Calculate the difference in indices to determine the number of turns needed
    turn_difference = (source_index - target_index) % len(turn_order)

    # Determine the direction of the best next turn
    if turn_difference == 0:
        return Action.FORWARD
    elif turn_difference == 1:
        return Action.LEFT
    elif turn_difference == 2:
        return Action.RIGHT
    elif turn_difference == 3:
        return Action.RIGHT

class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObserationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2

class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """
    SHELVES = 0 # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1 # binary layer indicating requested shelves
    AGENTS = 2 # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3 # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4 # binary layer indicating agents with load
    GOALS = 5 # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6 # binary layer indicating accessible cells (all but occupied cells/ out of map)
    PICKERS = 7 # binary layer indicating agents in the environment which only can_load
    PICKERS_DIRECTION = 8 # layer indicating agent directions as int (see Direction enum + 1 for values)

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS)
        else:
            return (_LAYER_AGENTS)

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

    @property
    def collision_layers(self):
        return ()

class Warehouse(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agvs: int,
        n_pickers: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        layout: str = None,
        observation_type: ObserationType=ObserationType.FLATTENED,
        normalised_coordinates: bool=False,
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
        :param n_agvs: Number of spawned and controlled agv
        :type n_agvs: int
        :param n_pickers: Number of spawned and controlled pickers
        :type n_pickers: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        self.n_agvs = n_agvs
        self.n_pickers = n_pickers
        self.n_agents_ = n_agvs + n_pickers

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)
        if n_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(n_agvs)] + [AgentType.PICKER for _ in range(n_pickers)]
        else:
            self._agent_types = [AgentType.AGENT for _ in range(self.n_agents_)]
        assert msg_bits == False
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)
        self.no_need_return_item = False
        self.fixing_clash_time = 4
        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps
        
        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(self.item_loc_dict) + 1, *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(sa_action_space)

        self.action_space_ = spaces.Tuple(tuple(self.n_agents_ * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []
        
        self.targets = np.zeros(len(self.item_loc_dict), dtype=int)
        self.stuck_count = []
        self._stuck_threshold = 5
        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.observation_space_ = None
        self.rack_groups = find_sections(list([loc for loc in self.item_loc_dict.values() if (loc[1], loc[0]) not in self.goals]))
        self._use_slow_obs()

        # for performance reasons we
        # can flatten the obs vector
        if observation_type == ObserationType.FLATTENED:
            self._use_fast_obs()

        self.renderer = None

    @property
    def observation_space(self):
        return self.observation_space_

    @property
    def action_space(self):
        return self.action_space_
    
    @property
    def n_agents(self):
        return self.n_agents_

    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        self.extra_rows = 2
        self._extra_rows_columns = 1
        self.grid_size = (
            (column_height + 1) * shelf_rows + 2 + self.extra_rows,
            (2 + 1) * shelf_columns + 1,
        )
        
        self.grid_size = (
            1 + (column_height + 1 + self._extra_rows_columns) * shelf_rows + 1 + self._extra_rows_columns + self.extra_rows,
            (2 + 1 + self._extra_rows_columns) * shelf_columns + 1  + self._extra_rows_columns,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        # Goals under racks
        accepted_x = []
        for i in range(0, self.grid_size[1],  3 + self._extra_rows_columns):
            accepted_x.append(i)
            for j in range(self._extra_rows_columns):
                accepted_x.append(i+j+1)

        accepted_y = []
        for i in range(0, self.grid_size[0],  1 + self._extra_rows_columns + column_height):
            accepted_y.append(i)
            for j in range(self._extra_rows_columns):
                accepted_y.append(i+j+1)

        self.goals = [
            (i, self.grid_size[0] - 1)
            for i in range(self.grid_size[1]) if not i in accepted_x
        ]
        self.num_goals = len(self.goals)

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
            ((x < 1 + self._extra_rows_columns or x >= self.grid_size[1] - 1 - self._extra_rows_columns) 
             or (y < 1 + self._extra_rows_columns or y >= self.grid_size[0] - 1 - self._extra_rows_columns))
            or x in accepted_x  # vertical highways
            or y in accepted_y  # vertical highways
            or (y >= self.grid_size[0] - 1 - self.extra_rows)  # delivery row
            or y in [self.goals[0][1] - i - 1 for i in range(self._extra_rows_columns)]
        )
        item_loc_index = 1
        self.item_loc_dict = {}
        for y, x in self.goals:
            self.item_loc_dict[item_loc_index] = (x, y)
            item_loc_index+=1
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)
                if not highway_func(x, y) and (x, y) not in self.goals:
                    self.item_loc_dict[item_loc_index] = (y, x)
                    item_loc_index+=1
    
    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

        assert len(self.goals) >= 1, "At least one goal is required"

    def _use_slow_obs(self):
        self.fast_obs = False

        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)
        agent_id_space = spaces.Box(low=0.0, high=self.n_agents_, shape=(1,), dtype=np.float32)

        self._obs_bits_for_agvs = spaces.flatdim(agent_id_space) + 3  + spaces.flatdim(location_space)  + spaces.flatdim(location_space)
        self._obs_bits_for_pickers = spaces.flatdim(agent_id_space) + spaces.flatdim(location_space)  + spaces.flatdim(location_space)
        self._obs_bits_per_shelf = 1 #+ 1
        self._obs_bits_for_requests = 1


        self._obs_sensor_locations = len(self.item_loc_dict) - len(self.goals) #self.grid_size[0] * self.grid_size[1]
        
        self._obs_length = (
            self._obs_bits_for_agvs * self.n_agvs
            + self._obs_bits_for_pickers * self.n_pickers
            + self._obs_sensor_locations * (self._obs_bits_per_shelf
            + self._obs_bits_for_requests)
        )

        obs = {}
        for agent_id in range(self.n_agvs):
            obs[f"agent{agent_id+1}"] = spaces.Dict(OrderedDict(
                {
                    "agent_id": agent_id_space,
                    "carrying_shelf": spaces.MultiBinary(1),
                    "shelf_requested": spaces.MultiBinary(1),
                    "loading_state": spaces.MultiBinary(1),
                    "location": location_space,
                    "target_location": location_space,
                }
            ))
        for agent_id in range(self.n_pickers):
            obs[f"agent{self.n_agvs + agent_id+1}"] = spaces.Dict(OrderedDict(
                {
                    "agent_id": agent_id_space,
                    "location": location_space,
                    "target_location": location_space,
                }
            ))
                
        individual_location_obs = spaces.Dict(OrderedDict(
            {
                "has_shelf": spaces.MultiBinary(1),
                "shelf_requested": spaces.MultiBinary(1),
                # "has_carried_shelf": spaces.MultiBinary(1),
            }
        ))
        obs["sensors"] = spaces.Tuple(self._obs_sensor_locations * (individual_location_obs,))
        self.observation_space_ = spaces.Tuple(tuple([spaces.Dict(OrderedDict(obs)) for _ in range(self.n_agents_)]))

    def _use_fast_obs(self):
        if self.fast_obs:
            return

        self.fast_obs = True
        ma_spaces = []
        for _ in self.observation_space_:
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self._obs_length,),
                    dtype=np.float32,
                )
            ]

        self.observation_space_ = spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _make_obs(self, agent):
        if self.image_obs:
            # write image observations
            if agent.id == 1:
                layers = []
                # first agent's observation --> update global observation layers
                for layer_type in self.image_observation_layers:
                    if layer_type == ImageLayer.SHELVES:
                        layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0
                        # print("SHELVES LAYER")
                    elif layer_type == ImageLayer.REQUESTS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for requested_shelf in self.request_queue:
                            layer[requested_shelf.y, requested_shelf.x] = 1.0
                        # print("REQUESTS LAYER")
                    elif layer_type == ImageLayer.AGENTS:
                        layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                        # print("AGENTS LAYER")
                    elif layer_type == ImageLayer.AGENT_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.can_carry:
                                agent_direction = ag.dir.value + 1
                                layer[ag.x, ag.y] = float(agent_direction)
                    elif layer_type == ImageLayer.PICKERS:
                        layer = self.grid[_LAYER_PICKERS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                        # print("AGENTS LAYER")
                    elif layer_type == ImageLayer.PICKERS_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.can_load and not ag.can_carry:
                                agent_direction = ag.dir.value + 1
                                layer[ag.x, ag.y] = float(agent_direction)
                        # print("AGENT DIRECTIONS LAYER")
                    elif layer_type == ImageLayer.AGENT_LOAD:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.carrying_shelf is not None:
                                layer[ag.x, ag.y] = 1.0
                        # print("AGENT LOAD LAYER")
                    elif layer_type == ImageLayer.GOALS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for goal_y, goal_x in self.goals:
                            layer[goal_x, goal_y] = 1.0
                        # print("GOALS LAYER")
                    elif layer_type == ImageLayer.ACCESSIBLE:
                        layer = np.ones(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            layer[ag.y, ag.x] = 0.0
                        # print("ACCESSIBLE LAYER")
                    # print(layer)
                    # print()
                    # pad with 0s for out-of-map cells
                    layer = np.pad(layer, self.sensor_range, mode="constant")
                    layers.append(layer)
                self.global_layers = np.stack(layers)

            # global information was generated --> get information for agent
            start_x = agent.y
            end_x = agent.y + 2 * self.sensor_range + 1
            start_y = agent.x
            end_y = agent.x + 2 * self.sensor_range + 1
            obs = self.global_layers[:, start_x:end_x, start_y:end_y]

            if self.image_observation_directional:
                # rotate image to be in direction of agent
                if agent.dir == Direction.DOWN:
                    # rotate by 180 degrees (clockwise)
                    obs = np.rot90(obs, k=2, axes=(1,2))
                elif agent.dir == Direction.LEFT:
                    # rotate by 90 degrees (clockwise)
                    obs = np.rot90(obs, k=3, axes=(1,2))
                elif agent.dir == Direction.RIGHT:
                    # rotate by 270 degrees (clockwise)
                    obs = np.rot90(obs, k=1, axes=(1,2))
                # no rotation needed for UP direction
            return obs
        min_x = 0
        max_x = self.grid_size[1]
        min_y = 0
        max_y = self.grid_size[0]

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_pickers = np.pad(
                self.grid[_LAYER_PICKERS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]
            padded_pickers = self.grid[_LAYER_PICKERS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        pickers = padded_pickers[min_y:max_y, min_x:max_x].reshape(-1)
        if self.fast_obs:
            # write flattened observations
            obs = _VectorWriter(self.observation_space_[agent.id - 1].shape[0])

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y
            # Agent self observation
            obs.write([agent.id])
            if agent.type == AgentType.AGV:
                if agent.carrying_shelf:
                    obs.write([1, int(agent.carrying_shelf in self.request_queue)])
                else:
                    obs.skip(2)
                obs.write([agent.req_action == Action.TOGGLE_LOAD])
            obs.write([agent.y, agent.x])
            if self.targets[agent.id - 1] != 0:
                obs.write(self.item_loc_dict[self.targets[agent.id - 1]])
            else:
                obs.skip(2)
            # Others observation
            for i in range(self.n_agents_):
                agent_ = self.agents[i]
                if agent_.id != agent.id:
                    obs.write([agent_.id])
                    if agent_.type == AgentType.AGV:
                        if agent_.carrying_shelf:
                            obs.write([1, int(agent_.carrying_shelf in self.request_queue)])
                        else:
                            obs.skip(2)
                        obs.write([agent_.req_action == Action.TOGGLE_LOAD])
                    obs.write([agent_.y, agent_.x])
                    if self.targets[agent_.id - 1] != 0:
                        obs.write(self.item_loc_dict[self.targets[agent_.id - 1]])
                    else:
                        obs.skip(2)
            # Shelves observation
            for group in self.rack_groups:
                for (x, y) in group:
                    id_shelf = self.grid[_LAYER_SHELFS, x, y]
                    if id_shelf == 0:
                        obs.skip(2)
                    else:
                        obs.write(
                            [1.0 , int(self.shelfs[id_shelf - 1] in self.request_queue)]
                        )
                    # id_carried_shelf = self.grid[_LAYER_CARRIED_SHELFS, x, y]
                    # if id_carried_shelf == 0:
                    #     obs.skip(1)
                    # else:
                    #     obs.write([1.0])
            return obs.vector
 
        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y]),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = self.agents[id_ - 1].message
        # find neighboring pickers
        for i, id_ in enumerate(pickers):
            if id_ == 0:
                obs["sensors"][i]["has_picker"] = [0]
                obs["sensors"][i]["direction_picker"] = 0
                obs["sensors"][i]["local_message_picker"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_picker"] = [1]
                obs["sensors"][i]["direction_picker"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message_picker"] = self.agents[id_ - 1].message
        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        return obs
    
    def find_path(self, start, goal, agent, care_for_agents = True):
        grid = np.zeros(self.grid_size)
        # if agent.type in [AgentType.AGV, AgentType.AGENT]:
        if care_for_agents:
            grid += self.grid[_LAYER_AGENTS]
            grid += self.grid[_LAYER_PICKERS]
            if agent.type == AgentType.PICKER:
                grid[goal[0], goal[1]] -= self.grid[_LAYER_AGENTS, goal[0], goal[1]]
            else:
                grid[goal[0], goal[1]] -= self.grid[_LAYER_PICKERS, goal[0], goal[1]]

        special_case_jump = False
        if agent.type == AgentType.PICKER:
            # Agents can only travel through highways if carrying a shelf
            for x in range(self.grid_size[1]):
                for y in range(self.grid_size[0]):
                    grid[y, x] += not self._is_highway(x, y)
            grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])
            if   agent.type == AgentType.PICKER and  ((not self._is_highway(start[1], start[0])) and goal[0] == start[0] and abs(goal[1] - start[1]) == 1): # Ban "jumps" from on shelf to another
                special_case_jump = True
            for i in range(self.grid_size[1]):
                grid[self.grid_size[0] - 1, i] = 1

        grid[start[0], start[1]] = 0
        if not special_case_jump:
            grid = [list(map(int, l)) for l in (grid!=0)]
            grid = np.array(grid, dtype=np.float32)
            grid[np.where(grid == 1)] = np.inf
            grid[np.where(grid == 0)] = 1
            astar_path = pyastar2d.astar_path(grid, start, goal, allow_diagonal=False) # returns None if cant find path
            if astar_path is not None:
                astar_path = [tuple(x) for x in list(astar_path)] # convert back to other format
                astar_path = astar_path[1:]
        else:
            special_start = None
            if self._is_highway(start[1] - 1, start[0]):
                special_start = (start[0], start[1] - 1)
            if self._is_highway(start[1] + 1, start[0]):
                special_start = (start[0], start[1] + 1)
            grid[start[0], start[1]] = 1
            grid[special_start[0], special_start[1]] = 0
            grid = [list(map(int, l)) for l in (grid!=0)]
            grid = np.array(grid, dtype=np.float32)
            grid[np.where(grid == 1)] = np.inf
            grid[np.where(grid == 0)] = 1
            astar_path = pyastar2d.astar_path(grid, special_start, goal, allow_diagonal=False)
            if astar_path is not None:
                astar_path = [tuple(x) for x in list(astar_path)] # convert back to other format
            astar_path = astar_path
        if astar_path:
            return [(x, y) for y, x in astar_path]
        else:
            return []

    def _recalc_grid(self):
        self.grid[:] = 0
        carried_shelf_ids = [agent.carrying_shelf.id for agent in self.agents if agent.carrying_shelf]
        for s in self.shelfs:
            if s.id not in carried_shelf_ids:
                self.grid[_LAYER_SHELFS, s.y, s.x] = s.id
        for agent in self.agents:
            if agent.type == AgentType.PICKER:
                self.grid[_LAYER_PICKERS, agent.y, agent.x] = agent.id
            else:
                self.grid[_LAYER_AGENTS, agent.y, agent.x] = agent.id
            if agent.carrying_shelf:
                self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = agent.carrying_shelf.id
    
    def get_carrying_shelf_information(self):
        return [agent.carrying_shelf != None for agent in self.agents[:self.n_agvs]]

    def get_shelf_request_information(self):
        request_item_map = np.zeros(len(self.item_loc_dict) - len(self.goals))
        requested_shelf_ids = [shelf.id for shelf in self.request_queue]
        for id_, coords in self.item_loc_dict.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[_LAYER_SHELFS, coords[0], coords[1]] in requested_shelf_ids:
                    request_item_map[id_ - len(self.goals) - 1] = 1
        return request_item_map
    
    def get_empty_shelf_information(self):
        empty_item_map = np.zeros(len(self.item_loc_dict) - len(self.goals))
        for id_, coords in self.item_loc_dict.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[_LAYER_SHELFS, coords[0], coords[1]] == 0 and (self.grid[_LAYER_CARRIED_SHELFS, coords[0], coords[1]] == 0 or self.agents[self.grid[_LAYER_AGENTS, coords[0], coords[1]] - 1].req_action not in [Action.NOOP, Action.TOGGLE_LOAD]):
                    empty_item_map[id_ - len(self.goals) - 1] = 1
        return empty_item_map
    
    def get_shelf_dispatch_information(self):
        dispatch_item_map = np.zeros(len(self.item_loc_dict) - len(self.goals))
        for id_, coords in self.item_loc_dict.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[_LAYER_AGENTS, coords[0], coords[1]]!=0 and self.agents[self.grid[_LAYER_AGENTS, coords[0], coords[1]] - 1].req_action == Action.TOGGLE_LOAD:
                        dispatch_item_map[id_ - len(self.goals) - 1] = 1
        return dispatch_item_map
    
    def reset(self):
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
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
            size=self.n_agents_,
            replace=False,
        )
        agent_locs = [self._higway_locs[agent_loc_ids, 0], self._higway_locs[agent_loc_ids, 1]]
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.n_agents_)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits, agent_type = agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]
        self._recalc_grid()

        self.request_queue = list(
            np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )
        self.targets = np.zeros(len(self.agents), dtype=int)
        self.stuck_count = [[0, (agent.x, agent.y)] for agent in self.agents]
        return tuple([self._make_obs(agent) for agent in self.agents])

    def resolve_move_conflict(self, agent_list):
        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents if agent.action != Action.FORWARD]

        # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents if agent.action == Action.FORWARD]
        
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
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    # action = self.agents[agent_id - 1].req_action
                    # print(f"{agent_id}: C {cycle} {action}")
                    if agent_id > 0:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[_LAYER_PICKERS, start_node[1], start_node[0]]
                    if picker_id > 0:
                        commited_agents.add(picker_id)
                        continue
            except nx.NetworkXNoCycle:

                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[_LAYER_PICKERS, y, x]
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
                            if ((agent.type == AgentType.PICKER and self.grid[_LAYER_PICKERS, agent_new_y, agent_new_x] in [0, agent.id]) 
                                or (agent.type == AgentType.AGV and self.grid[_LAYER_AGENTS, agent_new_y, agent_new_x] in [0, agent.id])):
                                commited_agents.add(agent.id)
                                continue
                        # If the agent's next action bumps it into another agent
                        if (agent_new_x, agent_new_y) == (other.x, other.y):
                            agent.req_action = Action.NOOP # Stop the action
                            # Check if the clash is not solved naturaly by the other agent moving away
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and not other.req_action in (Action.LEFT, Action.RIGHT):
                                if other.fixing_clash == 0:# If the others are not already fixing the clash
                                    clashes+=1
                                    agent.fixing_clash = self.fixing_clash_time # Agent start time for clash fixing
                                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                    if new_path != []: # If the agent can find an alternative path, assign it if not let the other solve the clash
                                        agent.path = new_path
                                    else:
                                        agent.fixing_clash = 0
                        elif (agent_new_x, agent_new_y) == (other_new_x, other_new_y) and (agent_new_x, agent_new_y) != (agent.x, agent.y): 
                            # If the agent's next action bumps it into another agent position after they take actions simultaneously
                            if agent.fixing_clash == 0 and other.fixing_clash == 0:
                                agent.req_action = Action.NOOP # If the agent's actions leads them in the position of another STOP
                                agent.fixing_clash = self.fixing_clash_time  # Agent wait one step while the other moves into place

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents
        for agent in failed_agents:
            agent.req_action = Action.NOOP
        return clashes
    def step(
        self, macro_actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        # Logic for Macro Actions
        for agent, macro_action in zip(self.agents, macro_actions):
            # Initialize action for step
            agent.req_action = Action.NOOP
            # Collision avoidance logic
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
                # continue
            if not agent.busy:
                if macro_action != 0:
                    agent.path = self.find_path((agent.y, agent.x), self.item_loc_dict[macro_action], agent, care_for_agents=False)
                    # If not path was found refuse location
                    if agent.path == []:
                        agent.busy = False
                    else:
                        agent.busy = True
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self.targets[agent.id-1] = macro_action
                        self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
            else:
                # Check agent finished the give path if not continue the path
                if agent.path == []:
                    #self.targets[agent.id-1] = 0
                    if agent.type != AgentType.PICKER:
                        agent.req_action = Action.TOGGLE_LOAD
                    if agent.type != AgentType.AGV:
                        agent.busy = False
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    # If agent is at the end of a path and carrying a shelf and the target location is already occupied restart agent
                if agent.path and len(agent.path) == 1:
                    if agent.carrying_shelf and self.grid[_LAYER_SHELFS, agent.path[-1][1], agent.path[-1][0]]:
                        agent.req_action = Action.NOOP
                        agent.busy = False
                    if agent.type == AgentType.PICKER:
                        if (self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] == 0 or self.agents[self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] - 1].req_action != Action.TOGGLE_LOAD):
                            agent.req_action = Action.NOOP
                            # agent.busy = False
                        elif self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] != 0 and self.agents[self.grid[_LAYER_AGENTS, agent.path[-1][1], agent.path[-1][0]] - 1].req_action == Action.TOGGLE_LOAD:
                            self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]

        #  agents that can_carry should not collide
        clashes_count = self.resolve_move_conflict(self.agents)

        # Restart agents if they are stuck at the same position
        # This can happen when their goal is occupied after reaching their last step/re-calculating a path
        stucks_count = 0
        agvs_distance_travelled = 0
        pickers_distance_travelled = 0
        for agent in self.agents:
            if agent.busy: # Don't count path calculation/fixing steps
                if agent.req_action not in (Action.LEFT, Action.RIGHT): # Don't count changing directions
                    if agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals: # Don't count loading or changing directions
                        pos = self.stuck_count[agent.id - 1][1]
                        if agent.x == pos[0] and agent.y == pos[1]:
                            self.stuck_count[agent.id - 1][0] += 1
                        else:
                            self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
                            if agent.type == AgentType.PICKER:
                                pickers_distance_travelled += 1
                            else:
                                agvs_distance_travelled += 1
                        if self.stuck_count[agent.id - 1][0] > self._stuck_threshold and self.stuck_count[agent.id - 1][0] < self._stuck_threshold + self.column_height + 2: # Time to get out of aisle 
                            agent.req_action = Action.NOOP
                            if agent.path:
                                new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                if new_path:
                                    agent.path = new_path
                                    if agent.type == AgentType.PICKER and len(agent.path) == 1:
                                        continue
                                    self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
                                    continue
                            else:
                                stucks_count += 1
                                agent.busy = False
                                self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]

                        if self.stuck_count[agent.id - 1][0] > self._stuck_threshold + self.column_height + 2: # Time to get out of aisle 
                            stucks_count += 1
                            self.stuck_count[agent.id - 1] = [0, (agent.x, agent.y)]
                            agent.req_action = Action.NOOP
                            agent.busy = False

        rewards = np.zeros(self.n_agents_)
        # Add step penalty
        rewards -= 0.001
        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y
            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                agent.path = agent.path[1:]
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf and agent.type != AgentType.PICKER:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    if agent.type == AgentType.AGV:
                        picker_id = self.grid[_LAYER_PICKERS, agent.y, agent.x]
                        if picker_id:
                            agent.carrying_shelf = self.shelfs[shelf_id - 1]
                            self.grid[_LAYER_SHELFS, agent.y, agent.x] = 0
                            self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = shelf_id
                            agent.busy = False
                            # Reward Pickers for loading shelf
                            if self.reward_type == RewardType.GLOBAL:
                                rewards += 0.5
                            elif self.reward_type == RewardType.INDIVIDUAL:
                                rewards[picker_id - 1] += 0.1
                    elif agent.type == AgentType.AGENT:
                        agent.carrying_shelf = self.shelfs[shelf_id - 1]
                        agent.busy = False
                else:
                    agent.busy = False
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf and agent.type != AgentType.PICKER:
                picker_id = self.grid[_LAYER_PICKERS, agent.y, agent.x]
                if (agent.x, agent.y) in self.goals:
                    agent.busy = False
                    continue
                if self.grid[_LAYER_SHELFS, agent.y, agent.x] != 0:
                    agent.busy = False
                    continue
                if not self._is_highway(agent.x, agent.y):
                    if agent.type == AgentType.AGENT:
                        self.grid[_LAYER_SHELFS, agent.y, agent.x] =  agent.carrying_shelf.id
                        self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = 0
                        agent.carrying_shelf = None
                        agent.busy = False
                    if agent.type == AgentType.AGV and picker_id:
                        self.grid[_LAYER_SHELFS, agent.y, agent.x] =  agent.carrying_shelf.id
                        self.grid[_LAYER_CARRIED_SHELFS, agent.y, agent.x] = 0
                        agent.carrying_shelf = None
                        agent.busy = False
                        # Reward Pickers for un-loading shelf
                        if self.reward_type == RewardType.GLOBAL:
                            rewards += 0.5
                        elif self.reward_type == RewardType.INDIVIDUAL:
                            rewards[picker_id - 1] += 0.1
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        # rewards[agent.id - 1] += 0.5
                        raise NotImplementedError('TWO_STAGE reward not implemenred for diverse rware')
                    agent.has_delivered = False
        shelf_delivered = False
        shelf_deliveries = 0
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_CARRIED_SHELFS, x, y]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            shelf_deliveries += 1
            # remove from queue and replace it
            carried_shels = [agent.carrying_shelf for agent in self.agents if agent.carrying_shelf]
            new_shelf_candidates = list(set(self.shelfs) - set(self.request_queue) - set(carried_shels)) # sort so np.random with seed is repeatable
            new_shelf_candidates.sort(key = lambda x: x.id)
            new_request = np.random.choice(new_shelf_candidates)
            self.request_queue[self.request_queue.index(shelf)] = new_request

            if self.no_need_return_item:
                agent.carrying_shelf = None
                for sx, sy in zip(
                    np.indices(self.grid_size)[0].reshape(-1),
                    np.indices(self.grid_size)[1].reshape(-1),
                ): 
                    if not self._is_highway(sy, sx) and not self.grid[_LAYER_SHELFS, sy, sx]:
                        print(f"{sx}-{sy}")
                        self.shelfs[shelf_id - 1].x = sx
                        self.shelfs[shelf_id - 1].y = sy
                        self.grid[_LAYER_SHELFS, sy, sx] = shelf_id
                        break
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 1
        self._recalc_grid()

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            dones = self.n_agents_ * [True]
        else:
            dones = self.n_agents_ * [False]

        
        agvs_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[:self.n_agvs]])
        pickers_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[self.n_agvs:]])
        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = {}
        info["vehicles_busy"] = [agent.busy for agent in self.agents]
        info["shelf_deliveries"] = shelf_deliveries
        info["clashes"] = clashes_count
        info["stucks"] = stucks_count
        info["agvs_distance_travelled"] = agvs_distance_travelled
        info["pickers_distance_travelled"] = pickers_distance_travelled
        info["agvs_idle_time"] = agvs_idle_time
        info["pickers_idle_time"] = pickers_idle_time
        return new_obs, list(rewards), dones, info

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
    
