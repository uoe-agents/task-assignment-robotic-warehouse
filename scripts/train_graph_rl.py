"""Minimal training data collection & imitation scaffold for GraphRL.

This script collects (GraphState, chosen_task_index) pairs using the
`GraphScorePolicy` as teacher and saves them to a pickle file. The purpose is
to provide a minimal dataset and a place to plug-in a training loop later.

Usage (example):
    .venv/bin/python scripts/train_graph_rl.py --env-id tarware-small-2agvs-1pickers-globalobs-v1 --episodes 5 --steps 200 --out data/imit_data.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym

from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.policies import GraphScorePolicy
from tarware_ext.envs import TarwareAdapter


def _train_torch_from_samples(
    samples,
    out_model: str,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cpu",
    resume_model: str | None = None,
    checkpoint_interval: int = 1,
) -> None:
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        raise RuntimeError("PyTorch is required for training. Install torch to enable training.")

    from tarware_ext.policies.torch_gnn_policy import TorchGNNPolicy

    policy = TorchGNNPolicy()
    policy._ensure_torch()

    # Optionally resume from existing checkpoint
    if resume_model is not None:
        try:
            ckpt = torch.load(resume_model, map_location=device)
            if isinstance(ckpt, dict):
                model_sd = ckpt.get("model")
                scorer_sd = ckpt.get("scorer")
                if model_sd is not None:
                    policy.model.load_state_dict(model_sd)
                if scorer_sd is not None:
                    policy.scorer.load_state_dict(scorer_sd)
                print("Resumed model from", resume_model)
        except Exception as exc:
            print("Failed to resume model:", exc)
    model_params = list(policy.model.parameters()) + list(policy.scorer.parameters())
    optim = torch.optim.Adam(model_params, lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        n_updates = 0
        for sample in samples:
            node_features = torch.as_tensor(sample["node_features"], dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(sample.get("edge_index", []), dtype=torch.long, device=device)
            agent_node_ids = torch.as_tensor(sample["agent_node_ids"], dtype=torch.long, device=device)
            task_node_ids = torch.as_tensor(sample["task_node_ids"], dtype=torch.long, device=device)
            action_mask = None if sample["action_mask"] is None else torch.as_tensor(sample["action_mask"], dtype=torch.bool, device=device)
            teacher = sample["teacher_task_indices"]

            optim.zero_grad()
            # Forward through the node encoder (GraphSAGE) using edge_index
            embeds = policy.model(node_features, edge_index)
            agent_emb = embeds[agent_node_ids, :]
            task_emb = embeds[task_node_ids, :]
            na = agent_emb.shape[0]
            nt = task_emb.shape[0]
            if nt == 0:
                continue
            a_exp = agent_emb.unsqueeze(1).expand(-1, nt, -1)
            t_exp = task_emb.unsqueeze(0).expand(na, -1, -1)
            pair = torch.cat([a_exp, t_exp], dim=-1)
            pair_flat = pair.view(-1, pair.shape[-1])
            scores = policy.scorer(pair_flat).view(na, nt).squeeze(-1)

            # Collect per-agent supervised targets where teacher assigned a task
            logits_list = []
            targets_list = []
            for i in range(len(teacher)):
                targ = int(teacher[i])
                if targ is None or targ < 0:
                    continue
                if targ >= nt:
                    continue
                # check action_mask validity if present
                if action_mask is not None and not bool(action_mask[i, targ]):
                    continue
                logits_list.append(scores[i])
                targets_list.append(targ)

            if not logits_list:
                continue

            logits = torch.stack(logits_list, dim=0)
            targets = torch.as_tensor(targets_list, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optim.step()

            total_loss += float(loss.detach().cpu().numpy())
            n_updates += 1

        avg_loss = total_loss / max(1, n_updates)
        print(f"Epoch {ep+1}/{epochs} avg_loss={avg_loss:.4f} updates={n_updates}")

        # Save checkpoint per interval and final model
        try:
            if checkpoint_interval and checkpoint_interval > 0 and ((ep + 1) % checkpoint_interval == 0):
                ckpt_path = f"{out_model}.epoch{ep+1}"
                torch.save({"model": policy.model.state_dict(), "scorer": policy.scorer.state_dict()}, ckpt_path)
                print("Saved checkpoint to", ckpt_path)
        except Exception as exc:
            print("Failed to save checkpoint:", exc)

    # Final save
    try:
        torch.save({"model": policy.model.state_dict(), "scorer": policy.scorer.state_dict()}, out_model)
        print("Saved trained model to", out_model)
    except Exception as exc:
        print("Failed to save model:", exc)


def sample_dataset(env_id: str, episodes: int, steps: int, out_path: Path, top_k: int | None = 2) -> None:
    samples: List[Dict[str, Any]] = []
    for ep in range(episodes):
        env = TarwareAdapter(gym.make(env_id))
        builder = GraphBuilderV0(top_k=top_k)
        policy = GraphScorePolicy(distance_mode="manhattan", assigner="greedy", top_k=top_k)
        obs, _ = env.reset()
        policy.reset(env.unwrapped if hasattr(env, "unwrapped") else env)
        for _ in range(steps):
            # Build GraphState from the unwrapped env to access internals
            target_env = env.unwrapped if hasattr(env, "unwrapped") else env
            g = builder.build(target_env)
            # teacher actions (loc_id per agent)
            actions = policy.act(env)
            # convert actions loc_id -> task_idx (or -1)
            task_indices = []
            for a in actions:
                if int(a) == 0:
                    task_indices.append(-1)
                    continue
                try:
                    task_idx = g.task_loc_ids.index(int(a))
                except ValueError:
                    task_idx = -1
                task_indices.append(int(task_idx))

            sample = {
                "node_features": g.node_features.copy(),
                "edge_index": g.edge_index.copy(),
                "agent_node_ids": list(g.agent_node_ids),
                "task_node_ids": list(g.task_node_ids),
                "task_loc_ids": list(g.task_loc_ids),
                "action_mask": None if g.action_mask is None else g.action_mask.copy(),
                "metadata": dict(g.metadata) if g.metadata is not None else {},
                "teacher_task_indices": list(task_indices),
            }
            samples.append(sample)
            # step with teacher actions
            env.step(actions)
        env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", default="data/imit_data.pkl")
    parser.add_argument("--train", action="store_true", help="Train a TorchGNNPolicy on collected samples if torch is available")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-model", type=str, default="data/torch_gnn.pth")
    parser.add_argument("--top-k", type=int, default=2, help="Top-K candidate tasks per agent used when collecting samples")
    parser.add_argument("--resume-model", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    sample_dataset(args.env_id, args.episodes, args.steps, Path(args.out), top_k=args.top_k)

    if args.train:
        # load samples and train using torch (if installed)
        with open(args.out, "rb") as f:
            samples = pickle.load(f)
        try:
            _train_torch_from_samples(
                samples,
                args.out_model,
                epochs=args.epochs,
                lr=args.lr,
                device="cpu",
                resume_model=args.resume_model,
                checkpoint_interval=args.checkpoint_interval,
            )
        except RuntimeError as exc:
            print("Training skipped:", exc)


if __name__ == "__main__":
    main()
