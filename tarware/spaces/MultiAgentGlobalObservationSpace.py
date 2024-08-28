import numpy as np
from gymnasium import spaces

from tarware.definitions import Action, AgentType, CollisionLayers
from tarware.spaces.MultiAgentBaseObservationSpace import (
    MultiAgentBaseObservationSpace, _VectorWriter)


class MultiAgentGlobalObservationSpace(MultiAgentBaseObservationSpace):
    def __init__(self, num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates=False):
        super(MultiAgentGlobalObservationSpace, self).__init__(num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates)

        self._define_obs_length()
        self.obs_lengths = [self.obs_length for _ in range(self.num_agents)]
        self._current_agents_info = []
        self._current_shelves_info = []

        ma_spaces = []
        for obs_length in self.obs_lengths:
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(obs_length,),
                    dtype=np.float32,
                )
            ]

        self.ma_spaces = spaces.Tuple(tuple(ma_spaces))

    def _define_obs_length(self):
        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)

        self.obs_bits_for_agvs = (3 + spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_agvs
        self.obs_bits_for_pickers = (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_pickers
        self.obs_bits_per_shelf = 1 * self.shelf_locations
        self.obs_bits_for_requests = 1 * self.shelf_locations
        self.obs_length = (
            self.obs_bits_for_agvs
            + self.obs_bits_for_pickers
            + self.obs_bits_per_shelf
            + self.obs_bits_for_requests
        )

    def extract_environment_info(self, environment):
        self._current_agents_info = []
        self._current_shelves_info = []

        # Extract agents info
        for agent in environment.agents:
            agent_info = []
            if agent.type == AgentType.AGV:
                if agent.carrying_shelf:
                    agent_info.extend([1, int(agent.carrying_shelf in environment.request_queue)])
                else:
                    agent_info.extend([0, 0])
                agent_info.extend([agent.req_action == Action.TOGGLE_LOAD])
            agent_info.extend(self.process_coordinates((agent.y, agent.x), environment))
            if agent.target:
                agent_info.extend(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
            else:
                agent_info.extend([0, 0])
            self._current_agents_info.append(agent_info)

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
        obs.write(self._current_agents_info[agent.id - 1])
        for agent_id, agent_info in enumerate(self._current_agents_info):
            if agent_id != agent.id - 1:
                obs.write(agent_info)
        obs.write(self._current_shelves_info)
        return obs.vector
