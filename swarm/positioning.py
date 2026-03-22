"""Relative positioning: MDS-based local coordinate estimation + Kalman filter.

Each node has only pairwise distance measurements to its neighbours.
We use *classical Multidimensional Scaling* (MDS) on the full distance
matrix (using true distances for the simulation, available from the
environment) to produce a globally consistent but *relative* embedding.
A per-node Kalman filter then smooths the estimated position over time.

In a real deployment only local distance measurements would be available;
the MDS step would be replaced by a distributed algorithm.  Here the
simulator uses MDS as a convenient stand-in that produces the same kind of
noise-and-smoothing pipeline.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .node import Node


# ---------------------------------------------------------------------------
# Classical MDS helper
# ---------------------------------------------------------------------------

def _classical_mds(D: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Compute a low-dimensional embedding from a squared-distance matrix.

    Parameters
    ----------
    D:
        ``(N, N)`` matrix of *squared* Euclidean distances.
    n_components:
        Target dimensionality (default 3 for 3-D embedding).

    Returns
    -------
    np.ndarray
        ``(N, n_components)`` coordinates matrix.
    """
    n = D.shape[0]
    # Double-centering
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D @ H
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Keep positive eigenvalues only (numerical noise can make some slightly negative)
    pos_mask = eigenvalues > 0
    k = min(n_components, int(pos_mask.sum()))
    coords = eigenvectors[:, :k] * np.sqrt(np.maximum(eigenvalues[:k], 0.0))
    # Pad with zeros if we got fewer dimensions than requested
    if k < n_components:
        pad = np.zeros((n, n_components - k))
        coords = np.hstack([coords, pad])
    return coords


# ---------------------------------------------------------------------------
# Per-node Kalman filter (constant-velocity model)
# ---------------------------------------------------------------------------

class _NodeKalmanFilter:
    """Simple 3-D constant-velocity Kalman filter for one node.

    State vector: ``[x, y, z, vx, vy, vz]``.
    """

    def __init__(self, initial_position: np.ndarray, process_noise: float = 0.01, measurement_noise: float = 0.1) -> None:
        # State [x y z vx vy vz]
        self.x = np.zeros(6)
        self.x[:3] = initial_position

        self.P = np.eye(6) * 1.0  # Covariance

        # State-transition matrix (dt=1 by default, updated in predict)
        self.F = np.eye(6)

        # Measurement matrix: observe position only
        self.H = np.zeros((3, 6))
        self.H[:, :3] = np.eye(3)

        # Process noise covariance
        self.Q = np.eye(6) * process_noise

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Predict step."""
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update step given a 3-D position measurement."""
        z = np.asarray(measurement)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].copy()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class RelativePositioner:
    """Estimates relative 3-D positions for all nodes from pairwise distances.

    Uses classical MDS to compute an embedding and per-node Kalman filters
    to smooth the estimates over time.

    Parameters
    ----------
    nodes:
        The list of :class:`~swarm.node.Node` objects (order must be stable).
    process_noise:
        Kalman process-noise variance (controls how fast estimates can change).
    measurement_noise:
        Kalman measurement-noise variance (controls how much MDS output is trusted).
    """

    def __init__(
        self,
        nodes: List[Node],
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        self.nodes = nodes
        self._filters: Optional[List[_NodeKalmanFilter]] = None
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise

    # ------------------------------------------------------------------

    def update(self, dt: float = 1.0) -> np.ndarray:
        """Compute updated estimated positions for all nodes.

        Steps
        -----
        1. Build the pairwise distance matrix from each node's
           ``neighbor_distances`` (and symmetric entries).
        2. Run classical MDS to get a ``(N, 3)`` relative embedding.
        3. Push the MDS result through per-node Kalman filters.
        4. Write filtered estimates back into ``node.estimated_position``.

        Returns
        -------
        np.ndarray
            ``(N, 3)`` smoothed estimated positions in relative space.
        """
        n = len(self.nodes)
        id_to_idx = {node.node_id: i for i, node in enumerate(self.nodes)}

        # Build squared-distance matrix (use 0 on diagonal, fill known pairs)
        D_sq = np.zeros((n, n))
        for i, node in enumerate(self.nodes):
            for nb_id, dist in node.neighbor_distances.items():
                if nb_id in id_to_idx:
                    j = id_to_idx[nb_id]
                    D_sq[i, j] = dist ** 2
                    D_sq[j, i] = dist ** 2

        # Fill in missing entries with a heuristic: use shortest-path (Floyd-Warshall)
        D_sq = self._complete_distance_matrix(D_sq)

        raw_positions = _classical_mds(D_sq, n_components=3)

        # Initialise Kalman filters on first call
        if self._filters is None:
            self._filters = [
                _NodeKalmanFilter(
                    raw_positions[i],
                    process_noise=self._process_noise,
                    measurement_noise=self._measurement_noise,
                )
                for i in range(n)
            ]

        smoothed = np.zeros((n, 3))
        for i, kf in enumerate(self._filters):
            kf.predict(dt=dt)
            smoothed[i] = kf.update(raw_positions[i])
            self.nodes[i].estimated_position = smoothed[i].copy()

        return smoothed

    # ------------------------------------------------------------------

    @staticmethod
    def _complete_distance_matrix(D_sq: np.ndarray) -> np.ndarray:
        """Fill missing squared-distance entries using Floyd-Warshall on distances."""
        n = D_sq.shape[0]
        # Work in distance space (not squared) for triangle-inequality propagation
        D = np.sqrt(D_sq)
        INF = 1e9
        # Where D==0 off-diagonal means unknown
        mask = (D == 0.0)
        D_fw = D.copy()
        np.fill_diagonal(D_fw, 0.0)
        D_fw[mask] = INF
        np.fill_diagonal(D_fw, 0.0)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if D_fw[i, k] + D_fw[k, j] < D_fw[i, j]:
                        D_fw[i, j] = D_fw[i, k] + D_fw[k, j]

        # Replace still-INF with max known distance (keeps MDS from blowing up)
        finite_vals = D_fw[D_fw < INF]
        fallback = float(finite_vals.max()) if finite_vals.size > 0 else 1.0
        D_fw[D_fw >= INF] = fallback

        return D_fw ** 2
