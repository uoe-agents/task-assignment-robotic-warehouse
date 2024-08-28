from enum import Enum, IntEnum


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

class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2

class CollisionLayers(IntEnum):
    AGVS = 0
    PICKERS = 1
    SHELVES = 2
    CARRIED_SHELVES = 3