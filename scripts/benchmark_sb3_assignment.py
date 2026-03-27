# ============================================================
# FILE: scripts/benchmark_sb3_assignment.py
# ============================================================
"""Benchmark PPO-assignment vs heuristic baseline on TA-RWARE envs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import gymnasium as gym
import numpy as np

# Allow running as `python scripts/...` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tarware  # noqa: F401
from tarware.heuristic import heuristic_episode

from tarware_ext.runners.metrics import summarize_episode
from tarware_ext.sb3 import GraphAssignmentConfig, GraphAssignmentEnv
from tarware_ext.controllers import HeuristicController


def _mean_of(rows: Iterable[Dict[str, float]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return float(np.mean(vals)) if vals else 0.0


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "episodes": float(len(rows)),
        "mean_return": _mean_of(rows, "global_episode_return"),
        "mean_deliveries": _mean_of(rows, "shelf_deliveries"),
        "mean_deliveries_per_step": _mean_of(rows, "deliveries_per_step"),
        "mean_return_per_step": _mean_of(rows, "return_per_step"),
        "mean_pick_rate": _mean_of(rows, "pick_rate"),
        "mean_clashes": _mean_of(rows, "clashes"),
        "mean_stucks": _mean_of(rows, "stucks"),
        "mean_episode_length": _mean_of(rows, "episode_length"),
    }


def _to_row(m: Dict[str, float]) -> Dict[str, float]:
    length = max(float(m["episode_length"]), 1.0)
    deliveries = float(m["shelf_deliveries"])
    ret = float(m["global_episode_return"])
    return {
        "global_episode_return": ret,
        "shelf_deliveries": deliveries,
        "deliveries_per_step": deliveries / length,
        "return_per_step": ret / length,
        "pick_rate": float(m["pick_rate"]),
        "clashes": float(m["clashes"]),
        "stucks": float(m["stucks"]),
        "episode_length": length,
    }


def _run_heuristic(env_id: str, seed: int, episodes: int, steps: int) -> List[Dict[str, float]]:
    """Evaluate heuristic policy with the same horizon used for PPO evaluation."""
    out: List[Dict[str, float]] = []

    for ep in range(episodes):
        env = gym.make(env_id, disable_env_checker=True)
        try:
            env.reset(seed=seed + ep)
            controller = HeuristicController()
            controller.reset(env, seed=seed + ep)

            infos: List[Dict[str, Any]] = []
            global_return = 0.0
            episode_returns = np.zeros(env.unwrapped.num_agents, dtype=np.float64)

            t = 0
            done = False
            while not done and t < steps:
                actions = controller.step(env, rl_agv_assignments=None)
                _obs, reward, terminated, truncated, info = env.step(actions)

                reward_vec = np.array(reward, dtype=np.float64)
                episode_returns += reward_vec
                global_return += float(np.sum(reward_vec))
                infos.append(info if isinstance(info, dict) else {})

                done_flags = np.array(terminated, dtype=bool) | np.array(truncated, dtype=bool)
                done = bool(np.all(done_flags))
                t += 1

            m = summarize_episode(infos, global_return, episode_returns)
            out.append(_to_row(m))
        finally:
            env.close()

    return out


def _train_ppo(
    env_id: str,
    seed: int,
    timesteps: int,
    steps: int,
    max_request_slots: int | None,
    obs_backend: str,
    graph_encoder_mode: str,
    graph_gnn_arch: str,
    gnn_emb_dim: int,
    gnn_layers: int,
    gnn_dropout: float,
    train_verbose: int,
) -> Any:
    from stable_baselines3 import PPO

    train_env = GraphAssignmentEnv(
        GraphAssignmentConfig(
            env_id=env_id,
            obs_backend=obs_backend,
            graph_encoder_mode=graph_encoder_mode,
            graph_gnn_arch=graph_gnn_arch,
            max_request_slots=max_request_slots,
            max_steps=steps,
            seed=seed,
            verbose=False,
        )
    )
    try:
        if obs_backend == "graph_dict":
            from tarware_ext.sb3.gnn_feature_extractor import GnnFeatureExtractor

            model = PPO(
                "MultiInputPolicy",
                train_env,
                policy_kwargs={
                    "features_extractor_class": GnnFeatureExtractor,
                    "features_extractor_kwargs": {
                        "emb_dim": int(gnn_emb_dim),
                        "gnn_layers": int(gnn_layers),
                        "dropout": float(gnn_dropout),
                        "architecture": str(graph_gnn_arch).strip().lower(),
                    },
                },
                verbose=train_verbose,
                seed=seed,
            )
        else:
            model = PPO("MlpPolicy", train_env, verbose=train_verbose, seed=seed)

        model.learn(total_timesteps=timesteps)
        return model
    finally:
        train_env.close()


def _eval_ppo(
    model: Any,
    env_id: str,
    seed: int,
    episodes: int,
    steps: int,
    max_request_slots: int | None,
    obs_backend: str,
    graph_encoder_mode: str,
    graph_gnn_arch: str,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    env = GraphAssignmentEnv(
        GraphAssignmentConfig(
            env_id=env_id,
            obs_backend=obs_backend,
            graph_encoder_mode=graph_encoder_mode,
            graph_gnn_arch=graph_gnn_arch,
            max_request_slots=max_request_slots,
            max_steps=steps,
            seed=seed,
            verbose=False,
        )
    )
    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=seed + ep)
            done = False
            infos: List[Dict[str, Any]] = []
            global_return = 0.0

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                global_return += float(reward)
                infos.append(info if isinstance(info, dict) else {})
                done = bool(terminated or truncated)

            m = summarize_episode(infos, global_return, [global_return])
            out.append(_to_row(m))
    finally:
        env.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env-ids",
        nargs="+",
        default=[
            "tarware-small-2agvs-1pickers-globalobs-v1",
            "tarware-medium-4agvs-2pickers-globalobs-v1",
            "tarware-large-8agvs-4pickers-globalobs-v1",
        ],
    )
    p.add_argument("--seed", type=int, default=21, help="Deprecated: use --seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Training/eval seeds")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=10_000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--obs-backend", choices=["assignment", "graph", "graph_dict"], default="assignment")
    p.add_argument("--graph-encoder-mode", choices=["manual", "gnn"], default="manual")
    p.add_argument("--graph-gnn-arch", choices=["sage", "gcn", "gat"], default="sage")
    p.add_argument("--gnn-emb-dim", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--gnn-dropout", type=float, default=0.0)
    p.add_argument("--max-request-slots", type=int, default=None)
    p.add_argument("--train-verbose", type=int, default=0)
    p.add_argument("--csv", type=str, default="")
    args = p.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]

    all_rows: List[Dict[str, Any]] = []

    for env_id in args.env_ids:
        print(f"\n=== {env_id} ===")
        heuristic_rows: List[Dict[str, float]] = []
        ppo_rows: List[Dict[str, float]] = []

        for seed in seeds:
            heuristic_rows.extend(
                _run_heuristic(
                    env_id=env_id,
                    seed=seed,
                    episodes=args.eval_episodes,
                    steps=args.steps,
                )
            )

            model = _train_ppo(
                env_id=env_id,
                seed=seed,
                timesteps=args.timesteps,
                steps=args.steps,
                max_request_slots=args.max_request_slots,
                obs_backend=args.obs_backend,
                graph_encoder_mode=args.graph_encoder_mode,
                graph_gnn_arch=args.graph_gnn_arch,
                gnn_emb_dim=args.gnn_emb_dim,
                gnn_layers=args.gnn_layers,
                gnn_dropout=args.gnn_dropout,
                train_verbose=args.train_verbose,
            )
            ppo_rows.extend(
                _eval_ppo(
                    model=model,
                    env_id=env_id,
                    seed=seed,
                    episodes=args.eval_episodes,
                    steps=args.steps,
                    max_request_slots=args.max_request_slots,
                    obs_backend=args.obs_backend,
                    graph_encoder_mode=args.graph_encoder_mode,
                    graph_gnn_arch=args.graph_gnn_arch,
                )
            )

        heuristic_summary = _aggregate(heuristic_rows)
        ppo_summary = _aggregate(ppo_rows)

        print(
            "heuristic "
            f"return={heuristic_summary['mean_return']:.3f} "
            f"deliveries={heuristic_summary['mean_deliveries']:.2f} "
            f"deliv/step={heuristic_summary['mean_deliveries_per_step']:.4f} "
            f"ret/step={heuristic_summary['mean_return_per_step']:.4f} "
            f"pick_rate={heuristic_summary['mean_pick_rate']:.2f} "
            f"clashes={heuristic_summary['mean_clashes']:.2f} "
            f"stucks={heuristic_summary['mean_stucks']:.2f} "
            f"episodes={int(heuristic_summary['episodes'])}"
        )
        print(
            "ppo       "
            f"return={ppo_summary['mean_return']:.3f} "
            f"deliveries={ppo_summary['mean_deliveries']:.2f} "
            f"deliv/step={ppo_summary['mean_deliveries_per_step']:.4f} "
            f"ret/step={ppo_summary['mean_return_per_step']:.4f} "
            f"pick_rate={ppo_summary['mean_pick_rate']:.2f} "
            f"clashes={ppo_summary['mean_clashes']:.2f} "
            f"stucks={ppo_summary['mean_stucks']:.2f} "
            f"episodes={int(ppo_summary['episodes'])}"
        )

        all_rows.append(
            {
                "env_id": env_id,
                "mode": "heuristic",
                **heuristic_summary,
            }
        )
        all_rows.append(
            {
                "env_id": env_id,
                "mode": "ppo_assignment",
                **ppo_summary,
            }
        )

    if args.csv:
        fieldnames = [
            "env_id",
            "mode",
            "episodes",
            "mean_return",
            "mean_deliveries",
            "mean_deliveries_per_step",
            "mean_return_per_step",
            "mean_pick_rate",
            "mean_clashes",
            "mean_stucks",
            "mean_episode_length",
        ]
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"\nCSV guardado en: {out_path}")


if __name__ == "__main__":
    main()
