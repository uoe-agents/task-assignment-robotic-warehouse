import gym
from tarware.warehouse import Warehouse, RewardType, Action, ObserationType
import itertools

_sizes = {
    "tiny": (1, 3),
    "small": (2, 5),
    "medium": (3, 5),
    "large": (4, 7),
}

_difficulty = {"-easy": 2, "": 1, "-hard": 0.5}

_perms = itertools.product(_sizes.keys(), _difficulty, range(1,20), range(1, 10))

for size, diff, n_agvs, n_pickers in _perms:
    # normal tasks
    gym.register(
        id=f"tarware-{size}-{n_agvs}agvs-{n_pickers}pickers-ag{diff}-v1",
        entry_point="tarware.warehouse:Warehouse",
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "n_agvs":  n_agvs,
            "n_pickers": n_pickers,
            "msg_bits": 0,
            "sensor_range": 1,
            "request_queue_size": int(n_agvs * _difficulty[diff]),
            "max_inactivity_steps": None,
            "max_steps": 500,
            "reward_type": RewardType.INDIVIDUAL,
        },
    )

def full_registration():
    _observation_type = {"": ObserationType.FLATTENED}
    _sensor_ranges = {f"-{sight}s": sight for sight in range(2, 6)}
    _sensor_ranges[""] = 1
    _image_directional = {"": True, "-Nd": False}
    _perms = itertools.product(_sizes.keys(), _difficulty, _observation_type, _sensor_ranges, _image_directional, range(1,20), range(1, 10), range(1, 16),)
    for size, diff, obs_type, sensor_range, directional, n_agvs, n_pickers, column_height in _perms:
        # normal tasks with modified column height
        if directional != "" and obs_type == "":
            # directional should only be used with image observations 
            continue
        gym.register(
            id=f"rware{obs_type}{directional}{sensor_range}-{size}-{column_height}h-{n_agvs}agvs-{n_pickers}pickers-ag{diff}-v1",
            entry_point="tarware.warehouse:Warehouse",
            kwargs={
                "column_height": 8,
                "shelf_rows": _sizes[size][0],
                "shelf_columns": _sizes[size][1],
                "n_agvs":  n_agvs,
                "n_pickers": n_pickers,
                "msg_bits": 0,
                "sensor_range": 1,
                "request_queue_size": int(n_agvs * _difficulty[diff]),
                "max_inactivity_steps": None,
                "max_steps": 500,
                "reward_type": RewardType.INDIVIDUAL,
            },
        )
