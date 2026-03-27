# ============================================================
# FILE: scripts/train_sb3_assignment_ppo.py
# ============================================================
"""Train SB3 PPO on GraphAssignmentEnv (explicit AGV assignment)."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Allow running as `python scripts/...` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tarware_ext.sb3 import GraphAssignmentConfig, GraphAssignmentEnv


def main() -> None:
    """
    Train a Maskable PPO reinforcement learning model for task assignment in a robotic warehouse.
    
    This function:
    - Parses command-line arguments for environment configuration, training parameters, and output settings
    - Dynamically imports stable-baselines3 contrib modules for masked action spaces
    - Creates a GraphAssignmentEnv with specified observation backend and graph encoder configuration
    - Wraps the environment with ActionMasker to enforce valid action constraints
    - Optionally creates an evaluation environment and callback for monitoring training progress
    - Initializes and trains a MaskablePPO model with either MultiInputPolicy (for graph observations) 
      or MlpPolicy (for standard observations)
    - Performs final evaluation if episodes are configured
    - Saves the trained model to the specified output path
    - Cleans up environment resources
    
    Raises:
        RuntimeError: If required stable-baselines3 or sb3-contrib dependencies are not installed.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--eval-env-id", default=None, help="Optional env id for evaluation; defaults to --env-id")
    p.add_argument("--obs-backend", choices=["assignment", "graph", "graph_dict"], default="assignment")
    p.add_argument("--graph-encoder-mode", choices=["manual", "gnn"], default="manual")
    p.add_argument("--graph-gnn-arch", choices=["sage", "gcn", "gat"], default="sage")
    p.add_argument("--gnn-emb-dim", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--gnn-dropout", type=float, default=0.0)
    p.add_argument("--max-request-slots", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None, help="Deprecated alias of --max-request-slots")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--eval-freq", type=int, default=0, help="Eval callback frequency in env steps (0 disables callback)")
    p.add_argument("--eval-episodes", type=int, default=5, help="Episodes for callback/final evaluation")
    p.add_argument("--best-model-dir", type=str, default=None, help="Optional directory to save best model from eval")
    p.add_argument("--out", type=str, default="data/sb3_assignment_ppo_model")
    args = p.parse_args()

    max_request_slots = args.max_request_slots if args.max_request_slots is not None else args.top_k

    try:
        maskable_module = importlib.import_module("sb3_contrib")
        wrappers_module = importlib.import_module("sb3_contrib.common.wrappers")
        callbacks_module = importlib.import_module("sb3_contrib.common.maskable.callbacks")
        evaluation_module = importlib.import_module("sb3_contrib.common.maskable.evaluation")
        MaskablePPO = getattr(maskable_module, "MaskablePPO")
        ActionMasker = getattr(wrappers_module, "ActionMasker")
        MaskableEvalCallback = getattr(callbacks_module, "MaskableEvalCallback")
        evaluate_policy = getattr(evaluation_module, "evaluate_policy")
    except Exception as exc:
        raise RuntimeError(
            "Instala dependencias con: pip install '.[sb3]' o pip install stable-baselines3 sb3-contrib"
        ) from exc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # create env function to ensure separate instances for training/eval with different seeds
    def make_env(*, env_id: str, seed: int, verbose: bool):
        return GraphAssignmentEnv(
            GraphAssignmentConfig(
                env_id=env_id,
                obs_backend=args.obs_backend,
                graph_encoder_mode=args.graph_encoder_mode,
                graph_gnn_arch=args.graph_gnn_arch,
                max_request_slots=max_request_slots,
                max_steps=args.steps,
                seed=seed,
                verbose=verbose,
            )
        )
    # Create training environment
    env = make_env(env_id=args.env_id, seed=args.seed, verbose=True)
    # Wrap with ActionMasker to enforce valid action constraints
    def mask_fn(current_env):
        return current_env.action_masks()
    # Note: GraphAssignmentEnv should be designed to return appropriate action masks for ActionMasker to function correctly.
    env = ActionMasker(env, mask_fn)

    eval_env = None
    eval_callback = None
    if int(args.eval_freq) > 0:
        eval_env_id = str(args.eval_env_id or args.env_id)
        eval_env = make_env(env_id=eval_env_id, seed=int(args.seed) + 1, verbose=False)
        eval_env = ActionMasker(eval_env, mask_fn)

        callback_kwargs = {
            "eval_env": eval_env,
            "eval_freq": max(1, int(args.eval_freq)),
            "n_eval_episodes": max(1, int(args.eval_episodes)),
            "deterministic": True,
            "verbose": 1,
            "log_path": str(out_path.parent / "maskable_eval_logs"),
        }
        if args.best_model_dir:
            best_dir = Path(args.best_model_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
            callback_kwargs["best_model_save_path"] = str(best_dir)
        eval_callback = MaskableEvalCallback(**callback_kwargs)

    if args.obs_backend == "graph_dict":
        from tarware_ext.sb3.gnn_feature_extractor import GnnFeatureExtractor
        # Note: MaskablePPO with MultiInputPolicy is used to handle graph observations, and GnnFeatureExtractor processes the graph data into features for the policy network.
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": GnnFeatureExtractor,
                "features_extractor_kwargs": {
                    "emb_dim": args.gnn_emb_dim,
                    "gnn_layers": args.gnn_layers,
                    "dropout": args.gnn_dropout,
                    "architecture": args.graph_gnn_arch,
                },
            },
            verbose=1,
            seed=args.seed,
        )
    else:
        # For non-graph observation backends, we can use the standard MlpPolicy.
        model = MaskablePPO("MlpPolicy", env, verbose=1, seed=args.seed)
    # Train the model with the optional evaluation callback to monitor progress and save best models.
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    final_eval_env = eval_env
    if final_eval_env is None and int(args.eval_episodes) > 0:
        eval_env_id = str(args.eval_env_id or args.env_id)
        final_eval_env = make_env(env_id=eval_env_id, seed=int(args.seed) + 2, verbose=False)
        final_eval_env = ActionMasker(final_eval_env, mask_fn)

    if final_eval_env is not None and int(args.eval_episodes) > 0:
        mean_reward, std_reward = evaluate_policy(
            model,
            final_eval_env,
            n_eval_episodes=max(1, int(args.eval_episodes)),
            deterministic=True,
            warn=False,
        )
        print(
            f"Eval final (maskable): mean_reward={float(mean_reward):.3f} std_reward={float(std_reward):.3f} "
            f"episodes={max(1, int(args.eval_episodes))}"
        )

    model.save(str(out_path))
    if final_eval_env is not None:
        final_eval_env.close()
    env.close()
    print("Modelo guardado en:", out_path)


if __name__ == "__main__":
    main()
