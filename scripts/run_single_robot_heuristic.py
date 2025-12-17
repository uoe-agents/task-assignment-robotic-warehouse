import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym

from tarware.heuristic_single_robot import single_robot_heuristic_episode


parser = ArgumentParser(
    description="Run single-robot-type heuristic on WarehouseEnv",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--num_episodes",
    default=1000,
    type=int,
    help="Number of episodes to run",
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Seed to run with",
)
parser.add_argument(
    "--render",
    action="store_true",
)
parser.add_argument(
    "--env_id",
    default="tarware-extralarge-21agvs-0pickers-partialobs-v1",
    type=str,
    help="Gymnasium environment ID",
)

args = parser.parse_args()


def info_statistics(infos, global_episode_return, episode_returns):
    _total_deliveries = 0
    _total_clashes = 0
    _total_stuck = 0
    for info in infos:
        _total_deliveries += info["shelf_deliveries"]
        _total_clashes += info["clashes"]
        _total_stuck += info["stucks"]
        info["total_deliveries"] = _total_deliveries
        info["total_clashes"] = _total_clashes
        info["total_stuck"] = _total_stuck
    last_info = infos[-1]
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = global_episode_return
    last_info["episode_returns"] = episode_returns
    return last_info


if __name__ == "__main__":
    env = gym.make(args.env_id)
    seed = args.seed
    completed_episodes = 0
    for i in range(args.num_episodes):
        start = time.time()
        infos, global_episode_return, episode_returns = single_robot_heuristic_episode(
            env.unwrapped, args.render, seed + i
        )
        end = time.time()
        last_info = info_statistics(infos, global_episode_return, episode_returns)
        last_info["overall_pick_rate"] = (
            last_info.get("total_deliveries") * 3600 / (5 * last_info["episode_length"])
        )
        episode_length = len(infos)
        print(
            f"Completed Episode {completed_episodes}: "
            f"| [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]"
            f"| [Global return={last_info.get('global_episode_return'):.2f}]"
            f"| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]"
            f"| [Total clashes={last_info.get('total_clashes'):.2f}]"
            f"| [Total stuck={last_info.get('total_stuck'):.2f}] "
            f"| [FPS = {episode_length/(end-start):.2f}]"
        )
        completed_episodes += 1


