import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tarware.definitions import Action, AgentType, CollisionLayers
from tarware.spaces.MultiAgentBaseObservationSpace import (
    MultiAgentBaseObservationSpace, _VectorWriter)


class MultiAgentPartialObservationSpace(MultiAgentBaseObservationSpace):
    def __init__(self, num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates=False):
        super(MultiAgentPartialObservationSpace, self).__init__(num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates)

        self._define_obs_length_agvs()
        self._define_obs_length_pickers()
        self.agv_obs_lengths = [self._obs_length_agvs for _ in range(self.num_agvs)]
        self.picker_obs_lengths = [self._obs_length_pickers for _ in range(self.num_pickers)]
        self._current_agvs_agents_info = []
        self._current_pickers_agents_info = []
        self._current_shelves_info = []
        ma_spaces = []
        for obs_length in self.agv_obs_lengths + self.picker_obs_lengths:
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(obs_length,),
                    dtype=np.float32,
                )
            ]

        self.ma_spaces = spaces.Tuple(tuple(ma_spaces))

    def _define_obs_length_agvs(self):
        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)

        self.agvs_obs_bits_for_agvs = 3 + (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_agvs
        self.agvs_obs_bits_for_pickers = (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_pickers
        self.agvs_obs_bits_per_shelf = 1 * self.shelf_locations
        self.agvs_obs_bits_for_requests = 1 * self.shelf_locations

        self._obs_length_agvs = (
            self.agvs_obs_bits_for_agvs
            + self.agvs_obs_bits_for_pickers
            + self.agvs_obs_bits_per_shelf
            + self.agvs_obs_bits_for_requests
        )

    def _define_obs_length_pickers(self):
        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)

        self.pickers_obs_bits_for_agvs = (3 + spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_agvs
        self.pickers_obs_bits_for_pickers = (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_pickers

        self._obs_length_pickers = (
            self.pickers_obs_bits_for_agvs
            + self.pickers_obs_bits_for_pickers
        )

    def extract_environment_info(self, environment):
        self._current_agvs_agents_info = []
        self._current_pickers_agents_info = []
        self._current_shelves_info = []

        # Extract agents info
        for agent in environment.agents:
            agvs_agent_info = []
            pickers_agent_info = []
            if agent.type == AgentType.AGV:
                if agent.carrying_shelf:
                    pickers_agent_info.extend([1, int(agent.carrying_shelf in environment.request_queue)])
                else:
                    pickers_agent_info.extend([0, 0])
                pickers_agent_info.extend([agent.req_action == Action.TOGGLE_LOAD])
            agvs_agent_info.extend(self.process_coordinates((agent.y, agent.x), environment))
            pickers_agent_info.extend(self.process_coordinates((agent.y, agent.x), environment))
            if agent.target:
                agvs_agent_info.extend(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
                pickers_agent_info.extend(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
            else:
                agvs_agent_info.extend([0, 0])
                pickers_agent_info.extend([0, 0])
            self._current_agvs_agents_info.append(agvs_agent_info)
            self._current_pickers_agents_info.append(pickers_agent_info)

        # Extract shelves info
        for group in environment.rack_groups:
            for (x, y) in group:
                id_shelf = environment.grid[CollisionLayers.SHELVES, x, y]
                if id_shelf!=0:
                    self._current_shelves_info.extend([1.0 , int(environment.shelfs[id_shelf - 1] in environment.request_queue)])
                else:
                    self._current_shelves_info.extend([0, 0])

    def observation(self, agent):
        obs = _VectorWriter(self.ma_spaces[agent.id - 1].shape[0])
        if agent.type == AgentType.AGV:
            obs.write(self._current_pickers_agents_info[agent.id - 1])
            for agent_id, agent_info in enumerate(self._current_agvs_agents_info):
                if agent_id != agent.id - 1:
                    obs.write(agent_info)
            obs.write(self._current_shelves_info)
        else:
            obs.write(self._current_pickers_agents_info[agent.id - 1])
            for agent_id, agent_info in enumerate(self._current_pickers_agents_info):
                if agent_id != agent.id - 1:
                    obs.write(agent_info)
        return obs.vector