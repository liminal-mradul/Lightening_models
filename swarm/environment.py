"""Environment: 3-D bounded arena containing the swarm of nodes."""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .node import Node


class Environment:
    """Bounded 3-D arena that holds and manages the swarm.

    Nodes are distributed so that they fill the arena with approximately
    even density while preserving a stochastic (random) distribution.
    This is achieved by stratified sampling: the volume is divided into a
    grid of equal-volume cells and one node is placed randomly inside each
    cell (Poisson-disk-lite approach).

    Parameters
    ----------
    bounds : tuple of (float, float, float)
        Half-extents ``(Lx, Ly, Lz)`` so the arena spans
        ``[-Lx, Lx] x [-Ly, Ly] x [-Lz, Lz]``.
    num_nodes : int
        Total number of nodes to place.
    detection_radius : float
        Ranging / communication radius for every node.
    noise_std : float
        Standard deviation of distance-measurement noise.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0),
        num_nodes: int = 100,
        detection_radius: float = 3.0,
        noise_std: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.num_nodes = num_nodes
        self.detection_radius = detection_radius
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.nodes: List[Node] = []
        self._place_nodes()

    # ------------------------------------------------------------------
    # Node placement
    # ------------------------------------------------------------------

    def _place_nodes(self) -> None:
        """Stratified random placement for approximately even density."""
        # Determine grid dimensions such that num_cells >= num_nodes.
        # Volume ratio: each cell side proportional to world side.
        ratio = self.bounds / self.bounds.min()
        # Target cells per axis
        n_cells_float = self.num_nodes ** (1.0 / 3.0) * ratio
        nx, ny, nz = np.ceil(n_cells_float).astype(int)
        total_cells = int(nx * ny * nz)

        # Build cell centres
        xs = np.linspace(-self.bounds[0], self.bounds[0], nx + 1)
        ys = np.linspace(-self.bounds[1], self.bounds[1], ny + 1)
        zs = np.linspace(-self.bounds[2], self.bounds[2], nz + 1)

        cell_indices = np.array(
            [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
        )
        # Randomly pick num_nodes distinct cells (if total_cells >= num_nodes)
        chosen = self.rng.choice(total_cells, size=min(self.num_nodes, total_cells), replace=False)
        chosen_cells = cell_indices[chosen]

        positions = []
        for idx, (ci, cj, ck) in enumerate(chosen_cells):
            x = self.rng.uniform(xs[ci], xs[ci + 1])
            y = self.rng.uniform(ys[cj], ys[cj + 1])
            z = self.rng.uniform(zs[ck], zs[ck + 1])
            positions.append([x, y, z])

        # If we need more nodes than cells, fill the rest uniformly
        for extra in range(len(positions), self.num_nodes):
            x = self.rng.uniform(-self.bounds[0], self.bounds[0])
            y = self.rng.uniform(-self.bounds[1], self.bounds[1])
            z = self.rng.uniform(-self.bounds[2], self.bounds[2])
            positions.append([x, y, z])

        self.nodes = [
            Node(i, pos, detection_radius=self.detection_radius)
            for i, pos in enumerate(positions)
        ]

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def update_all_neighbors(self) -> None:
        """Refresh neighbour distance tables for every node."""
        for node in self.nodes:
            node.update_neighbors(self.nodes, noise_std=self.noise_std)

    def move_nodes(self, max_step: float = 0.1) -> None:
        """Apply a small random displacement to every node (simulates crowd movement).

        Nodes are reflected at the arena boundaries so they never leave
        the arena.
        """
        for node in self.nodes:
            delta = self.rng.normal(0.0, max_step, size=3)
            node.position = node.position + delta
            # Reflect at boundaries
            for axis in range(3):
                lo, hi = -self.bounds[axis], self.bounds[axis]
                if node.position[axis] < lo:
                    node.position[axis] = lo + (lo - node.position[axis])
                elif node.position[axis] > hi:
                    node.position[axis] = hi - (node.position[axis] - hi)
                # Clamp after reflection in case of very large step
                node.position[axis] = float(np.clip(node.position[axis], lo, hi))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_positions(self) -> np.ndarray:
        """Return ``(N, 3)`` array of true node positions."""
        return np.array([n.position for n in self.nodes])

    def get_led_colors(self) -> np.ndarray:
        """Return ``(N, 3)`` array of LED RGB colours (zeros when off)."""
        colors = np.zeros((len(self.nodes), 3))
        for i, node in enumerate(self.nodes):
            if node.led_on:
                colors[i] = node.led_color
        return colors
