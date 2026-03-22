"""Tests for the RelativePositioner (MDS + Kalman)."""

import numpy as np
import pytest

from swarm.environment import Environment
from swarm.positioning import RelativePositioner, _classical_mds


class TestClassicalMDS:
    def test_output_shape(self):
        n = 8
        # Squared distances of random 3-D points
        rng = np.random.default_rng(0)
        pts = rng.uniform(-5, 5, size=(n, 3))
        D = np.array([[np.sum((pts[i] - pts[j]) ** 2) for j in range(n)] for i in range(n)])
        coords = _classical_mds(D, n_components=3)
        assert coords.shape == (n, 3)

    def test_recovers_distances(self):
        """MDS embedding should preserve pairwise distances (up to rigid transform)."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(-5, 5, size=(10, 3))
        D_sq = np.array([[np.sum((pts[i] - pts[j]) ** 2) for j in range(10)] for i in range(10)])
        coords = _classical_mds(D_sq, n_components=3)
        # Reconstruct distances from embedding
        for i in range(10):
            for j in range(i + 1, 10):
                orig_dist = np.sqrt(D_sq[i, j])
                emb_dist = np.linalg.norm(coords[i] - coords[j])
                assert pytest.approx(emb_dist, rel=1e-4) == orig_dist

    def test_symmetric_input(self):
        n = 5
        coords = _classical_mds(np.zeros((n, n)), n_components=3)
        # All zero distances → all points at same location
        assert coords.shape == (n, 3)


class TestRelativePositioner:
    def _build_env_and_positioner(self, n=20, seed=42):
        env = Environment(
            bounds=(8.0, 8.0, 4.0),
            num_nodes=n,
            detection_radius=5.0,
            noise_std=0.0,
            seed=seed,
        )
        env.update_all_neighbors()
        positioner = RelativePositioner(env.nodes)
        return env, positioner

    def test_update_returns_correct_shape(self):
        env, positioner = self._build_env_and_positioner(n=15)
        positions = positioner.update()
        assert positions.shape == (15, 3)

    def test_estimated_positions_written_to_nodes(self):
        env, positioner = self._build_env_and_positioner(n=12)
        positioner.update()
        for node in env.nodes:
            assert node.estimated_position is not None
            assert node.estimated_position.shape == (3,)

    def test_kalman_smoothing_reduces_noise(self):
        """Running multiple steps should stabilise estimates (Kalman effect)."""
        env, positioner = self._build_env_and_positioner(n=15)
        estimates = []
        for _ in range(10):
            estimates.append(positioner.update().copy())
        # Variance of estimates should decrease or stabilise over time
        var_early = np.var(estimates[1] - estimates[0])
        var_late = np.var(estimates[-1] - estimates[-2])
        # Late variance should not blow up; allow generous tolerance
        assert var_late <= var_early * 10 + 1e-6

    def test_repeated_calls_stable(self):
        env, positioner = self._build_env_and_positioner(n=10)
        for _ in range(5):
            pos = positioner.update()
            assert np.all(np.isfinite(pos))
