"""Evaluate policies on TA-RWARE envs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tarware  # noqa: F401
from tarware_ext.envs import TarwareAdapter
from tarware_ext.logs import CSVLogger
from tarware_ext.policies import (
    DistanceMode,
    GraphGreedyPolicy,
    HeuristicPolicy,
    RandomPolicy,
    GraphScorePolicy,
    GNNPolicy,
)
from tarware_ext.runners import evaluate
from tarware_ext.graphs.builder_v0 import GraphBuilderV0


def _make_env(env_id: str) -> Callable[[], TarwareAdapter]:
    def _factory() -> TarwareAdapter:
        env = gym.make(env_id)
        return TarwareAdapter(env)

    return _factory


def _build_policy(name: str, env: TarwareAdapter, distance: str | None = None, top_k: int | None = None):
    if name == "random":
        return RandomPolicy(env)
    if name == "heuristic":
        return HeuristicPolicy(env)
    if name == "graph_greedy":
        mode = DistanceMode(distance or DistanceMode.MANHATTAN.value)
        return GraphGreedyPolicy(distance_mode=mode)
    if name == "graph_score":
        # Lightweight graph-based scoring policy (non-learning). We forward the
        # distance mode and candidate `top_k` to the underlying builder; the
        # policy/builder will use that information to score candidate tasks.
        return GraphScorePolicy(distance_mode=(distance or DistanceMode.MANHATTAN.value), top_k=top_k)
    if name == "gnn":
        return GNNPolicy(builder=GraphBuilderV0(distance_mode=(distance or DistanceMode.MANHATTAN.value), top_k=top_k))
    if name == "torch_gnn":
        # Local import to avoid requiring torch unless the user requests it
        from tarware_ext.policies.torch_gnn_policy import TorchGNNPolicy

        return TorchGNNPolicy(builder=GraphBuilderV0(distance_mode=(distance or DistanceMode.MANHATTAN.value), top_k=top_k))
    raise ValueError(f"Unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--policy", choices=["random", "heuristic", "graph_greedy", "graph_score", "gnn", "torch_gnn"], default="random")
    parser.add_argument("--distance", choices=["manhattan", "find_path"], default="manhattan")
    parser.add_argument("--top-k", type=int, default=2, help="Top-K candidate tasks per agent used by graph builder/policies")
    parser.add_argument("--active-alpha", type=int, default=3)
    parser.add_argument("--max-active-agvs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv", default="eval.csv")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--debug-graph", action="store_true", help="Build and print a debug summary of the GraphState before evaluation")
    args = parser.parse_args()

    env = TarwareAdapter(gym.make(args.env_id))
    if args.policy == "graph_greedy":
        policy = _build_policy(
            args.policy,
            env,
            distance=args.distance,
            top_k=args.top_k,
        )
        policy.active_alpha = args.active_alpha
        if args.max_active_agvs is not None:
            policy.max_active_agvs = args.max_active_agvs
    else:
        policy = _build_policy(args.policy, env, distance=args.distance, top_k=args.top_k)
    # Optionally build and print a debug graph snapshot using the builder.
    if args.debug_graph:
        try:
            builder = getattr(policy, "builder", None) or GraphBuilderV0(
                distance_mode=(args.distance or DistanceMode.MANHATTAN.value),
                top_k=args.top_k,
            )
            # Safely unwrap common Gym wrappers to reach the underlying
            # `Warehouse` object expected by builders. We attempt a few
            # unwrap strategies (env.unwrapped, env.env, nested unwrapping)
            # and fall back to the original `env` when no underlying object
            # exposes `agents`.
            def _unwrap_env(e):
                cand = e
                for _ in range(6):
                    try:
                        if hasattr(cand, "unwrapped") and getattr(cand, "unwrapped") is not cand:
                            cand = getattr(cand, "unwrapped")
                            continue
                    except Exception:
                        pass
                    try:
                        if hasattr(cand, "env") and getattr(cand, "env") is not cand:
                            cand = getattr(cand, "env")
                            continue
                    except Exception:
                        pass
                    break
                return cand

            target_env = _unwrap_env(env)
            # Ensure the env has been initialised (agents, request_queue)
            # by performing a single reset on the adapter if possible.
            try:
                # Use the adapter reset when available to preserve wrappers
                if hasattr(env, "reset"):
                    env.reset(seed=args.seed)
            except Exception:
                pass

            # If we still don't have the attributes the builder expects, try
            # an extra step via `env.env.unwrapped` as a last resort.
            if not hasattr(target_env, "agents"):
                try:
                    inner = getattr(env, "env", None)
                    if inner is not None and hasattr(inner, "unwrapped"):
                        candidate = getattr(inner, "unwrapped")
                        if hasattr(candidate, "agents"):
                            target_env = candidate
                except Exception:
                    pass

            g = builder.build(target_env)
            nf_shape = getattr(g.node_features, "shape", None)
            ei_shape = getattr(g.edge_index, "shape", None)
            print("---- Graph Debug Summary ----")
            print(f"env_id: {args.env_id}")
            print(f"nodes: {nf_shape[0] if nf_shape else 'N/A'}, node_feature_dim: {nf_shape[1] if nf_shape else 'N/A'}")
            print(f"edge_index shape: {ei_shape}")
            print(f"num_agents (reported): {g.metadata.get('num_agents')}")
            print(f"num_tasks (reported): {g.metadata.get('num_tasks')}")
            print(f"agent_node_ids: {g.agent_node_ids}")
            print(f"task_node_ids (count): {len(g.task_node_ids)}")
            print(f"sample task_loc_ids (first 10): {g.task_loc_ids[:10]}")
            if g.action_mask is not None:
                print(f"action_mask shape: {g.action_mask.shape}")
            else:
                print("action_mask: None")
            print("---- End Graph Debug ----")
        except Exception as exc:  # pragma: no cover - debug convenience
            print("Graph debug build failed:", exc)
    env.close()

    eval_fn = _make_env(args.env_id)
    results = evaluate(
        eval_fn,
        policy,
        episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
    )
    if not results:
        return

    summary = results["summary"]
    episodes = results["episodes"]

    print(
        " | ".join(
            [
                f"episodes={int(summary['episodes'])}",
                f"mean_return={summary['mean_return']:.2f}",
                f"mean_deliveries={summary['mean_deliveries']:.2f}",
                f"mean_clashes={summary['mean_clashes']:.2f}",
                f"mean_stuck={summary['mean_stuck']:.2f}",
                f"mean_pick_rate={summary['mean_pick_rate']:.2f}",
                f"overall_pick_rate={summary['overall_pick_rate']:.2f}",
                f"mean_episode_length={summary['mean_episode_length']:.2f}",
                f"mean_fps={summary['mean_fps']:.2f}",
            ]
        )
    )

    if not args.no_csv:
        fieldnames = [
            "episode",
            "seed",
            "env_id",
            "distance_mode",
            "active_alpha",
            "max_active_agvs",
            "top_k",
            "episode_length",
            "shelf_deliveries",
            "clashes",
            "stucks",
            "global_episode_return",
            "pick_rate",
            "fps",
        ]
        logger = CSVLogger(args.csv, fieldnames=fieldnames)
        for row in episodes:
            enriched = dict(row)
            enriched["env_id"] = args.env_id
            enriched["distance_mode"] = args.distance
            enriched["active_alpha"] = args.active_alpha
            enriched["max_active_agvs"] = args.max_active_agvs
            enriched["top_k"] = args.top_k
            logger.log({key: enriched.get(key) for key in fieldnames})
        logger.close()


if __name__ == "__main__":
    main()
