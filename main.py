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
import os
import sys

from swarm import Environment, Simulation
from swarm.patterns import PatternType
from swarm.visualization import SwarmVisualizer


_PATTERN_MAP = {p.name.lower(): p for p in PatternType}


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
    return parser.parse_args(argv)


def build_pattern_sequence(pattern_arg: str, steps_per_pattern: int = 30):
    if pattern_arg == "all":
        return [(p, steps_per_pattern * 0.1) for p in PatternType]  # dt=0.1 per step
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

    pattern_sequence = build_pattern_sequence(args.pattern)
    print(f"[swarm] Pattern sequence: {[p.name for p, _ in pattern_sequence]}")

    sim = Simulation(
        env,
        pattern_sequence=pattern_sequence,
        dt=0.1,
        enable_movement=args.movement,
        movement_scale=0.05,
    )

    viz = SwarmVisualizer(sim, figsize=(12, 9))

    print(f"[swarm] Running {args.steps} steps ...")
    for step in range(args.steps):
        sim.step()
        if step % 10 == 0:
            print(f"  step {step:4d}/{args.steps}  t={sim.t:.2f}  pattern={sim.current_pattern.name}")

    viz.render_frame()

    if args.save:
        viz.save_snapshot(args.save)
        print(f"[swarm] Snapshot saved → {args.save}")
    else:
        snapshot_path = "swarm_snapshot.png"
        viz.save_snapshot(snapshot_path)
        print(f"[swarm] Snapshot saved → {snapshot_path}")

    print("[swarm] Done.")


if __name__ == "__main__":
    main()
