#!/usr/bin/env python3
"""main.py – Entry-point demo for the swarm LED visualisation system.

Running this script:
    python main.py [--steps N] [--nodes N] [--pattern NAME] [--save PATH]

It sets up a 3-D swarm arena, runs the simulation for the requested number
of steps cycling through all built-in patterns, and either displays a live
matplotlib window or saves a snapshot/animation.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Optional

from swarm import Environment, Simulation
from swarm.patterns import PatternEngine, PatternType


_PATTERN_MAP = {p.name.lower(): p for p in PatternType}
DEFAULT_FIGSIZE = (12, 9)
DEFAULT_INTERVAL_MS = 50
DEFAULT_FPS = 15
SNIPPET_FPS = 12
SNIPPET_STEPS = 30


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for *path* when one is specified."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _run_animation_default(viz, steps: int, save_path: Optional[str]) -> None:
    """Run animation with shared defaults for interval and fps."""
    viz.run_animation(steps, interval_ms=DEFAULT_INTERVAL_MS, save_path=save_path, fps=DEFAULT_FPS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Swarm LED visualisation – relative-position 3-D simulation"
    )
    parser.add_argument("--steps", type=int, default=60, help="Number of simulation steps (default: 60)")
    parser.add_argument("--nodes", type=int, default=80, help="Number of swarm nodes (default: 80)")
    parser.add_argument(
        "--pattern",
        choices=list(_PATTERN_MAP.keys()) + ["all"],
        default="all",
        help="Pattern to display, or 'all' to cycle through every pattern (default: all)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save a snapshot PNG to this path (optional)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--movement",
        action="store_true",
        default=False,
        help="Enable random node movement each step",
    )
    parser.add_argument(
        "--output-dir",
        default="swarm_output",
        help="Directory to write CSV, plots, and video (created if missing)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Display a live 3-D view while the simulation runs (requires GUI backend)",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional path to save an animation (gif/mp4). Defaults to <output-dir>/swarm_animation.gif",
    )
    parser.add_argument(
        "--image-pattern",
        default=None,
        help="Path to a black/white image to project as a brightness mask (enables IMAGE pattern)",
    )
    parser.add_argument(
        "--image-threshold",
        type=float,
        default=0.5,
        help="Brightness threshold (0–1) for IMAGE pattern (values below are treated as off)",
    )
    parser.add_argument(
        "--image-invert",
        action="store_true",
        help="Invert the brightness mask from the image (black ↔ white)",
    )
    return parser.parse_args(argv)


def build_pattern_sequence(pattern_arg: str, steps_per_pattern: int = 30, has_image_pattern: bool = False):
    if pattern_arg == "all":
        patterns = list(PatternType) if has_image_pattern else [p for p in PatternType if p is not PatternType.IMAGE]
        return [(p, steps_per_pattern * 0.1) for p in patterns]  # dt=0.1 per step
    return [(_PATTERN_MAP[pattern_arg], float("inf"))]


def main(argv=None):
    args = parse_args(argv)

    print(f"[swarm] Initialising environment: {args.nodes} nodes, seed={args.seed}")
    env = Environment(
        bounds=(10.0, 10.0, 5.0),
        num_nodes=args.nodes,
        detection_radius=4.0,
        noise_std=0.05,
        seed=args.seed,
    )
    env.update_all_neighbors()

    if args.live:
        os.environ.setdefault("SWARM_INTERACTIVE", "1")
    from swarm.visualization import SwarmVisualizer  # Delayed import to honour SWARM_INTERACTIVE

    has_image_pattern = args.image_pattern is not None
    if args.image_pattern and args.pattern not in {"image", "all"}:
        raise SystemExit("When using --image-pattern you must also set --pattern image or --pattern all.")

    pattern_sequence = build_pattern_sequence(args.pattern, has_image_pattern=has_image_pattern)
    print(f"[swarm] Pattern sequence: {[p.name for p, _ in pattern_sequence]}")

    sim = Simulation(
        env,
        pattern_sequence=pattern_sequence,
        dt=0.1,
        enable_movement=args.movement,
        movement_scale=0.05,
    )

    if args.image_pattern:
        image_arr = PatternEngine.load_bw_image(args.image_pattern)
        sim.pattern_engine.set_image_pattern(
            image_arr, threshold=args.image_threshold, invert=args.image_invert
        )

    viz = SwarmVisualizer(sim, figsize=DEFAULT_FIGSIZE)

    print(f"[swarm] Running {args.steps} steps ...")
    video_default = os.path.join(args.output_dir, "swarm_animation.gif")
    video_path = args.video_path or video_default
    if args.video_path:
        _ensure_parent_dir(video_path)
    elif args.live:
        _ensure_parent_dir(video_default)

    video_created_path: Optional[str] = None
    if args.live:
        target = args.video_path or video_default
        _run_animation_default(viz, args.steps, target)
        video_created_path = target
    elif args.video_path:
        _run_animation_default(viz, args.steps, video_path)
        video_created_path = video_path
    else:
        for step in range(args.steps):
            sim.step()
            if step % 10 == 0:
                print(f"  step {step:4d}/{args.steps}  t={sim.t:.2f}  pattern={sim.current_pattern.name}")

    export_paths = export_outputs(
        sim,
        args.output_dir,
        viz,
        video_created_path,  # None triggers default animation export (covers live without --video-path)
        args.save,
    )

    for label, path in export_paths.items():
        if path:
            print(f"[swarm] {label} → {path}")

    print("[swarm] Done.")


def export_outputs(sim: Simulation, output_dir: str, viz: "SwarmVisualizer", prerendered_video_path: Optional[str], snapshot_override: Optional[str]) -> Dict[str, str]:
    """Persist simulation artefacts (plots, CSV, optional animation) to *output_dir*.

    Parameters
    ----------
    sim:
        The active Simulation instance that has already run.
    output_dir:
        Directory where artefacts will be written (created if missing).
    viz:
        A visualiser tied to the simulation; used to render plots.
    prerendered_video_path:
        If provided, use this pre-rendered animation path and skip creating a new clip.
    snapshot_override:
        Optional custom snapshot path; defaults to <output_dir>/swarm_snapshot.png.

    Returns
    -------
    Dict[str, str]
        Mapping of human-readable artefact labels to output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    snapshot_path = snapshot_override or os.path.join(output_dir, "swarm_snapshot.png")
    topdown_path = os.path.join(output_dir, "swarm_topdown.png")
    csv_path = os.path.join(output_dir, "relative_positions.csv")
    from swarm.visualization import SwarmVisualizer  # Local import to avoid missing reference when called outside main

    if prerendered_video_path and not os.path.exists(prerendered_video_path):
        print(f"[swarm] Provided animation path {prerendered_video_path} does not exist; regenerating.")
        prerendered_video_path = None

    video_out = prerendered_video_path or os.path.join(output_dir, "swarm_animation.gif")

    # Re-create visualiser to avoid closed figure after animation
    fresh_viz = SwarmVisualizer(sim, figsize=DEFAULT_FIGSIZE)
    fresh_viz.save_snapshot(snapshot_path)
    fresh_viz.save_topdown(topdown_path)

    positions = []
    for node in sim.env.nodes:
        est = node.estimated_position if node.estimated_position is not None else node.position
        positions.append(
            {
                "node_id": node.node_id,
                "x": est[0],
                "y": est[1],
                "z": est[2],
            }
        )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(positions)

    # Save a short animation snippet if requested or by default for exports
    if prerendered_video_path is None:
        # Record a brief clip continuing from the current state
        temp_sim = _build_animation_sim(sim)
        temp_viz = SwarmVisualizer(temp_sim, figsize=DEFAULT_FIGSIZE)
        temp_viz.run_animation(SNIPPET_STEPS, interval_ms=DEFAULT_INTERVAL_MS, save_path=video_out, fps=SNIPPET_FPS)

    if not os.path.exists(video_out):
        raise RuntimeError(
            f"Animation was expected at {video_out} but was not created. This may be caused by missing codecs (ffmpeg/pillow) or insufficient file permissions."
        )

    return {
        "Snapshot": snapshot_path,
        "Top-down plot": topdown_path,
        "Positions CSV": csv_path,
        "Animation": video_out,
    }


def _build_animation_sim(sim: Simulation) -> Simulation:
    """Create a short-run simulation preserving the current pattern/image config."""
    cloned_env = _clone_environment(sim.env)
    temp_sim = Simulation(cloned_env, pattern_sequence=[(sim.current_pattern, float("inf"))], dt=sim.dt)
    config = sim.pattern_engine.get_image_pattern_config()
    if config:
        image_arr, threshold, invert = config
        temp_sim.pattern_engine.set_image_pattern(image_arr, threshold=threshold, invert=invert)
    return temp_sim


def _clone_environment(env: Environment) -> Environment:
    """Deep-copy environment geometry and LED state without re-randomising placement."""
    cloned = Environment(
        bounds=tuple(env.bounds),
        num_nodes=len(env.nodes),
        detection_radius=env.detection_radius,
        noise_std=env.noise_std,
        seed=None,
    )
    for src, dst in zip(env.nodes, cloned.nodes):
        dst.position = src.position.copy()
        dst.led_color = src.led_color.copy()
        dst.led_on = src.led_on
    cloned.update_all_neighbors()
    return cloned


if __name__ == "__main__":
    main()
