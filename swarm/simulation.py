"""Simulation: orchestrates the swarm over time."""

from __future__ import annotations

import time as _time
from typing import List, Optional

from .environment import Environment
from .positioning import RelativePositioner
from .patterns import PatternEngine, PatternType


class Simulation:
    """Main simulation loop.

    Parameters
    ----------
    env:
        The :class:`~swarm.environment.Environment` containing the nodes.
    pattern_sequence:
        List of ``(PatternType, duration)`` tuples.  The simulation cycles
        through patterns in order, spending ``duration`` ticks on each.
        If ``None``, defaults to a single WAVE pattern that runs forever.
    dt:
        Simulation timestep (arbitrary units).
    enable_movement:
        Whether to apply random node movement each step.
    movement_scale:
        Maximum per-step displacement when movement is enabled.
    noise_std:
        Distance measurement noise injected during neighbour updates.
    """

    def __init__(
        self,
        env: Environment,
        pattern_sequence: Optional[List] = None,
        dt: float = 0.1,
        enable_movement: bool = False,
        movement_scale: float = 0.05,
    ) -> None:
        self.env = env
        self.dt = dt
        self.enable_movement = enable_movement
        self.movement_scale = movement_scale

        if pattern_sequence is None:
            pattern_sequence = [(PatternType.WAVE, float("inf"))]
        self.pattern_sequence = pattern_sequence

        self.positioner = RelativePositioner(env.nodes)
        self.pattern_engine = PatternEngine(env.nodes)

        self.t: float = 0.0
        self._step_count: int = 0

        # Track which pattern slot we are in
        self._pattern_idx: int = 0
        self._pattern_elapsed: float = 0.0

    # ------------------------------------------------------------------

    @property
    def current_pattern(self) -> PatternType:
        return self.pattern_sequence[self._pattern_idx][0]

    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one timestep ``dt``."""
        # 1. Optionally move nodes
        if self.enable_movement:
            self.env.move_nodes(max_step=self.movement_scale)

        # 2. Refresh neighbour distance tables
        self.env.update_all_neighbors()

        # 3. Update relative position estimates (MDS + Kalman)
        self.positioner.update(dt=self.dt)

        # 4. Apply current pattern
        self.pattern_engine.apply(self.current_pattern, self.t)

        # 5. Advance time
        self.t += self.dt
        self._step_count += 1
        self._pattern_elapsed += self.dt

        # 6. Switch pattern if duration exceeded
        _, duration = self.pattern_sequence[self._pattern_idx]
        if self._pattern_elapsed >= duration:
            self._pattern_idx = (self._pattern_idx + 1) % len(self.pattern_sequence)
            self._pattern_elapsed = 0.0

    def run(self, n_steps: int, callback=None) -> None:
        """Run the simulation for *n_steps* ticks.

        Parameters
        ----------
        n_steps:
            Number of timesteps to execute.
        callback:
            Optional callable ``callback(sim)`` invoked after each step,
            e.g. to update a live plot.
        """
        for _ in range(n_steps):
            self.step()
            if callback is not None:
                callback(self)
