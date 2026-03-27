"""Demo visualization for TA-RWARE assignments.

This script runs an episode with a chosen policy and displays the env using
the existing pyglet renderer. Optionally it saves annotated frames (GIF)
showing agent->task assignment lines. Pillow (`PIL`) and `imageio` are
optional dependencies used for frame annotation and GIF saving.

Usage examples:
  # Real-time window (no saved frames)
  .venv/bin/python scripts/demo_visualize.py --env-id tarware-small-2agvs-1pickers-globalobs-v1 --policy graph_score --steps 200

  # Save annotated GIF (requires pillow + imageio)
  .venv/bin/python scripts/demo_visualize.py --env-id tarware-small-2agvs-1pickers-globalobs-v1 --policy graph_score --steps 200 --save-gif out.gif

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tarware_ext.envs import TarwareAdapter
import subprocess
from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.graphs.schema import NodeType, GraphState
from tarware_ext.policies import (
    GraphScorePolicy,
    GraphGreedyPolicy,
    RandomPolicy,
    HeuristicPolicy,
    GNNPolicy,
    DistanceMode,
)

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    import imageio
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def _build_policy(name: str, env: Any, distance: str | None = None, top_k: int | None = 2):
    mode = DistanceMode(distance or DistanceMode.MANHATTAN.value)
    if name == "graph_score":
        return GraphScorePolicy(distance_mode=(distance or DistanceMode.MANHATTAN.value), top_k=top_k)
    if name == "graph_greedy":
        return GraphGreedyPolicy(distance_mode=mode)
    if name == "random":
        return RandomPolicy(env)
    if name == "heuristic":
        return HeuristicPolicy(env)
    if name == "gnn":
        return GNNPolicy()
    if name == "torch_gnn":
        # Import locally to avoid forcing a global torch import at module load
        from tarware_ext.policies.torch_gnn_policy import TorchGNNPolicy

        # Pass a builder that respects the CLI `--top-k` flag
        return TorchGNNPolicy(builder=GraphBuilderV0(top_k=top_k))
    raise ValueError(f"Unknown policy: {name}")


def _to_pixel(env: Any, x: int, y: int) -> tuple[int, int]:
    # Resolve inner env that exposes rendering/grid properties (unwrap wrappers)
    def _resolve_inner(adapter: Any):
        # Adapter has attribute `env` pointing to the wrapped env
        target = getattr(adapter, "env", adapter)
        for _ in range(10):
            # prefer objects that expose renderer/grid_size
            if hasattr(target, "renderer") or hasattr(target, "grid_size") or hasattr(target, "rows"):
                return target
            try:
                if hasattr(target, "unwrapped") and getattr(target, "unwrapped") is not target:
                    target = getattr(target, "unwrapped")
                    continue
            except Exception:
                pass
            try:
                if hasattr(target, "env") and getattr(target, "env") is not target:
                    target = getattr(target, "env")
                    continue
            except Exception:
                pass
            break
        return target

    target = _resolve_inner(env)
    # Prefer the Viewer instance if it's been created by a prior render
    viewer = None
    if hasattr(target, "renderer") and getattr(target, "renderer") is not None:
        viewer = getattr(target, "renderer")

    if viewer is not None:
        grid_pixel = int(getattr(viewer, "grid_size", 30))
        rows = int(getattr(viewer, "rows", getattr(target, "grid_size", (30,))[0]))
    else:
        gs = getattr(target, "grid_size", None)
        if isinstance(gs, (tuple, list)):
            rows = int(gs[0])
        else:
            rows = int(getattr(target, "rows", 30))
        grid_pixel = 30
    # center position used by Viewer: center_x = (grid_pixel+1)*col + grid_pixel//2 + 1
    center_x = (grid_pixel + 1) * int(x) + grid_pixel // 2 + 1
    row = rows - int(y) - 1
    center_y = (grid_pixel + 1) * row + grid_pixel // 2 + 1
    return int(center_x), int(center_y)


def _find_viewer(adapter: Any):
    # Return the Viewer instance if present (otherwise None)
    target = getattr(adapter, "env", adapter)
    for _ in range(10):
        if hasattr(target, "renderer") and getattr(target, "renderer") is not None:
            return getattr(target, "renderer")
        try:
            if hasattr(target, "unwrapped") and getattr(target, "unwrapped") is not target:
                target = getattr(target, "unwrapped")
                continue
        except Exception:
            pass
        try:
            if hasattr(target, "env") and getattr(target, "env") is not target:
                target = getattr(target, "env")
                continue
        except Exception:
            pass
        break
    return None


def demo(
    env_id: str,
    policy_name: str,
    steps: int = 200,
    top_k: int = 2,
    save_gif: str | None = None,
    sleep: float = 0.05,
    hold_window: bool = False,
    open_gif: bool = False,
) -> None:
    env = TarwareAdapter(gym.make(env_id))
    policy = _build_policy(policy_name, env, distance="manhattan", top_k=top_k)
    builder = getattr(policy, "builder", None) or GraphBuilderV0(top_k=top_k)

    obs, _ = env.reset()
    # initialise policy state
    if hasattr(policy, "reset"):
        try:
            policy.reset(env.unwrapped if hasattr(env, "unwrapped") else env)
        except TypeError:
            policy.reset()

    frames = []
    for t in range(steps):
        target_env = env.unwrapped if hasattr(env, "unwrapped") else env
        g: GraphState = builder.build(target_env)

        # Decide actions
        if getattr(policy, "uses_env", False):
            actions = policy.act(target_env)
        else:
            actions = policy.act(obs)

        # Map actions to task indices for annotation
        assigned_task_idx = []
        for a in actions:
            if int(a) == 0:
                assigned_task_idx.append(-1)
            else:
                try:
                    idx = g.task_loc_ids.index(int(a))
                except ValueError:
                    idx = -1
                assigned_task_idx.append(int(idx))

        # Print concise input/output for supervisor (Entrada/Salida)
        print(f"Step {t:03d}: actions={actions}")
        # Brief graph summary
        print(f"  Graph: num_agents={g.metadata.get('num_agents')} num_tasks={g.metadata.get('num_tasks')} top_k={g.metadata.get('top_k')}")

        # Render frame (human window) and optionally capture RGB array
        if save_gif is not None or PIL_AVAILABLE:
            frame = env.render(mode="rgb_array")
            # PIL annotation when available
            if PIL_AVAILABLE:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                # optionally load a default font (may be system dependent)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

                # draw assignment lines for AGVs
                for ai_pos, agv_node_idx in enumerate(g.agent_node_ids):
                    if g.node_types[agv_node_idx] != NodeType.AGV:
                        continue
                    ay = int(g.node_features[agv_node_idx, 0])
                    ax = int(g.node_features[agv_node_idx, 1])
                    if assigned_task_idx[ai_pos] is None or assigned_task_idx[ai_pos] < 0:
                        continue
                    task_idx = assigned_task_idx[ai_pos]
                    if task_idx < 0:
                        continue
                    task_node_idx = g.task_node_ids[task_idx]
                    ty = int(g.node_features[task_node_idx, 0])
                    tx = int(g.node_features[task_node_idx, 1])
                    s = _to_pixel(env, ax, ay)
                    e = _to_pixel(env, tx, ty)
                    draw.line([s, e], fill=(255, 0, 0), width=3)
                    draw.ellipse([s[0] - 6, s[1] - 6, s[0] + 6, s[1] + 6], outline=(0, 255, 0), width=2)
                    draw.ellipse([e[0] - 6, e[1] - 6, e[0] + 6, e[1] + 6], outline=(0, 0, 255), width=2)
                    if font is not None:
                        draw.text((s[0] + 8, s[1] + 2), str(task_idx), fill=(255, 0, 0), font=font)

                frame = np.array(img)
                frames.append(frame)
            else:
                frames.append(frame)
        else:
            env.render()

        # Step env and update obs
        step_out = env.step(actions)
        if isinstance(step_out, tuple) and len(step_out) in (5,):
            # gymnasium signature: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = step_out
            done = all(bool(x) for x in _as_seq(terminated)) or all(bool(x) for x in _as_seq(truncated))
        else:
            # Transition object path
            try:
                from tarware_ext.envs import Transition

                if isinstance(step_out, Transition):
                    obs = step_out.obs
                    done = step_out.done_all
                else:
                    obs = step_out
                    done = False
            except Exception:
                obs = step_out
                done = False

        if done:
            break

        time.sleep(max(0.0, sleep))

    # Save gif if requested
    if save_gif is not None and frames:
        if not PIL_AVAILABLE:
            print("Pillow or imageio not available; install 'pillow imageio' to save annotated GIFs")
        else:
            out = Path(save_gif)
            # Ensure destination directory exists to avoid FileNotFoundError
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                imageio.mimsave(str(out), frames, fps=10)
                print(f"Saved GIF to {out}")
                if open_gif:
                    try:
                        # Try to open with the system default image viewer (Linux)
                        subprocess.run(["xdg-open", str(out)], check=False)
                    except Exception as e:
                        print(f"Unable to open GIF automatically: {e}")
            except Exception as e:
                print(f"Unable to write GIF {out}: {e}")

    # If requested, hold the renderer window open until the user closes it.
    if hold_window:
        viewer = _find_viewer(env)
        if viewer is not None:
            print("Holding renderer window open. Close the window to exit the demo.")
            try:
                while getattr(viewer, "isopen", True):
                    try:
                        viewer.window.dispatch_events()
                    except Exception:
                        pass
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Interrupted by user.")
            # Note: Viewer.window.on_close currently calls exit(), so process may already have exited.
            return
        else:
            print("No renderer/viewer instance found to hold open.")

    try:
        env.close()
    except Exception:
        pass


def _as_seq(x: Any):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    return [x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument(
        "--policy",
        default="graph_score",
        choices=["graph_score", "graph_greedy", "random", "heuristic", "gnn", "torch_gnn"],
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--save-gif", type=str, default=None, help="Path to save annotated GIF (requires pillow+imageio)")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds between frames in real-time window")
    parser.add_argument("--hold-window", action="store_true", help="Keep the pyglet window open after the demo until you close it")
    parser.add_argument("--open-gif", action="store_true", help="Open the saved GIF with the system default viewer after saving (Linux: xdg-open)")
    args = parser.parse_args()
    demo(
        args.env_id,
        args.policy,
        steps=args.steps,
        top_k=args.top_k,
        save_gif=args.save_gif,
        sleep=args.sleep,
        hold_window=args.hold_window,
        open_gif=args.open_gif,
    )


if __name__ == "__main__":
    main()
