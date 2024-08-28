from abc import ABC

import numpy as np


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


class MultiAgentBaseObservationSpace(ABC):
    def __init__(self, num_agvs, num_pickers, grid_size, shelf_locations, msg_bits, normalised_coordinates=False):
        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers
        self.grid_size = grid_size
        self.shelf_locations = shelf_locations
        self.normalised_coordinates = normalised_coordinates
        self.ma_spaces = []
        super(MultiAgentBaseObservationSpace, self).__init__()

    def process_coordinates(self, coords, environment):
        if self.normalised_coordinates:
            return (coords[0] / (environment.grid_size[0] - 1), coords[1] / (environment.grid_size[1] - 1))
        else:
            return coords
    
    def observation(self, agent, environment):
        raise NotImplementedError("Please Implement this method")