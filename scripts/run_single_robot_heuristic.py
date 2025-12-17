import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym
import numpy as np

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
parser.add_argument(
    "--allocation_strategy",
    default="greedy",
    choices=["greedy", "batch_opt", "bnb_opt"],
    help="Task allocation strategy for assigning robots to requested shelves",
)
parser.add_argument(
    "--batch_size",
    default=None,
    type=int,
    help="For batch_opt: number of FIFO requests to consider for min-cost matching (default: min(#available, #unassigned))",
)
parser.add_argument(
    "--max_tasks_per_robot",
    default=10000,
    type=int,
    help="For bnb_opt: maximum tasks per robot within the lookahead plan",
)
parser.add_argument(
    "--node_budget",
    default=5000,
    type=int,
    help="For bnb_opt: max branch-and-bound nodes to expand (safety cap)",
)
parser.add_argument(
    "--max_seconds",
    default=1000,
    type=float,
    help="For bnb_opt: wall-clock time limit for the branch-and-bound search (seconds)",
)
parser.add_argument(
    "--empty_candidates_k",
    default=1000000,
    type=int,
    help="For bnb_opt: number of candidate empty rack slots (closest to goal) to consider when estimating return cost",
)
parser.add_argument(
    "--vary_seed",
    action="store_true",
    help="If set, uses (seed + episode_idx). Otherwise uses the same seed for all episodes (deterministic).",
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
    episode_summaries = []
    for i in range(args.num_episodes):
        start = time.time()
        episode_seed = (seed + i) if args.vary_seed else seed
        infos, global_episode_return, episode_returns = single_robot_heuristic_episode(
            env.unwrapped,
            args.render,
            episode_seed,
            allocation_strategy=args.allocation_strategy,
            batch_size=args.batch_size,
            max_tasks_per_robot=args.max_tasks_per_robot,
            node_budget=args.node_budget,
            max_seconds=args.max_seconds,
            empty_candidates_k=args.empty_candidates_k,
        )
        end = time.time()
        last_info = info_statistics(infos, global_episode_return, episode_returns)
        last_info["overall_pick_rate"] = (
            last_info.get("total_deliveries") * 3600 / (5 * last_info["episode_length"])
        )
        last_info["fps"] = last_info["episode_length"] / (end - start)
        last_info["seed"] = episode_seed
        episode_summaries.append(last_info)
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

    # Summary overview across episodes
    if episode_summaries:
        def _arr(key):
            return np.array([e.get(key, 0.0) for e in episode_summaries], dtype=np.float64)

        keys = [
            "overall_pick_rate",
            "global_episode_return",
            "total_deliveries",
            "total_clashes",
            "total_stuck",
            "episode_length",
            "fps",
        ]
        print("\n=== Episode Summary (mean ± std | min .. max) ===")
        print(f"strategy={args.allocation_strategy} env_id={args.env_id} episodes={len(episode_summaries)} seed_mode={'vary' if args.vary_seed else 'fixed'}")
        for k in keys:
            a = _arr(k)
            print(f"{k}: {a.mean():.3f} ± {a.std(ddof=0):.3f} | {a.min():.3f} .. {a.max():.3f}")


