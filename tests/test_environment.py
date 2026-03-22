"""Tests for the Environment class."""

import numpy as np
import pytest

from swarm.environment import Environment


class TestEnvironmentInit:
    def test_node_count(self):
        env = Environment(num_nodes=50, seed=0)
        assert len(env.nodes) == 50

    def test_nodes_within_bounds(self):
        bounds = (8.0, 6.0, 4.0)
        env = Environment(bounds=bounds, num_nodes=60, seed=1)
        for node in env.nodes:
            for axis, b in enumerate(bounds):
                assert -b - 1e-9 <= node.position[axis] <= b + 1e-9

    def test_unique_ids(self):
        env = Environment(num_nodes=40, seed=2)
        ids = [n.node_id for n in env.nodes]
        assert len(set(ids)) == len(ids)

    def test_reproducible_with_seed(self):
        env1 = Environment(num_nodes=30, seed=7)
        env2 = Environment(num_nodes=30, seed=7)
        p1 = env1.get_positions()
        p2 = env2.get_positions()
        assert np.allclose(p1, p2)

    def test_different_seeds_differ(self):
        env1 = Environment(num_nodes=30, seed=1)
        env2 = Environment(num_nodes=30, seed=2)
        assert not np.allclose(env1.get_positions(), env2.get_positions())


class TestNeighborUpdate:
    def test_neighbors_populated(self):
        env = Environment(num_nodes=20, detection_radius=5.0, seed=3)
        env.update_all_neighbors()
        # With a large radius most nodes should have neighbours
        total_neighbors = sum(len(n.neighbor_distances) for n in env.nodes)
        assert total_neighbors > 0

    def test_no_self_neighbors(self):
        env = Environment(num_nodes=15, detection_radius=10.0, seed=4)
        env.update_all_neighbors()
        for node in env.nodes:
            assert node.node_id not in node.neighbor_distances


class TestMovement:
    def test_nodes_stay_in_bounds_after_movement(self):
        bounds = (5.0, 5.0, 3.0)
        env = Environment(bounds=bounds, num_nodes=30, seed=5)
        for _ in range(20):
            env.move_nodes(max_step=0.5)
        for node in env.nodes:
            for axis, b in enumerate(bounds):
                assert -b - 1e-9 <= node.position[axis] <= b + 1e-9

    def test_positions_change(self):
        env = Environment(num_nodes=20, seed=6)
        before = env.get_positions().copy()
        env.move_nodes(max_step=0.3)
        after = env.get_positions()
        assert not np.allclose(before, after)


class TestGetters:
    def test_get_positions_shape(self):
        n = 25
        env = Environment(num_nodes=n, seed=8)
        pos = env.get_positions()
        assert pos.shape == (n, 3)

    def test_get_led_colors_shape(self):
        n = 10
        env = Environment(num_nodes=n, seed=9)
        colors = env.get_led_colors()
        assert colors.shape == (n, 3)

    def test_led_colors_zero_when_off(self):
        env = Environment(num_nodes=10, seed=10)
        # All LEDs start off
        colors = env.get_led_colors()
        assert np.allclose(colors, 0)

    def test_led_colors_reflect_state(self):
        env = Environment(num_nodes=5, seed=11)
        env.nodes[2].set_led((0.5, 0.8, 0.2))
        colors = env.get_led_colors()
        assert np.allclose(colors[2], [0.5, 0.8, 0.2])
        for i in [0, 1, 3, 4]:
            assert np.allclose(colors[i], 0)
