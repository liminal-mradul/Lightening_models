"""Tests for the Node class."""

import numpy as np
import pytest

from swarm.node import Node


def _make_cluster(n=5, radius=1.5, detection_radius=3.0):
    """Return a list of *n* nodes placed in a small cluster."""
    rng = np.random.default_rng(0)
    nodes = []
    for i in range(n):
        pos = rng.uniform(-radius, radius, size=3)
        nodes.append(Node(i, pos, detection_radius=detection_radius))
    return nodes


class TestNodeInit:
    def test_default_led_off(self):
        node = Node(0, [0, 0, 0])
        assert not node.led_on
        assert np.allclose(node.led_color, 0)

    def test_id_and_position(self):
        node = Node(7, [1.0, 2.0, 3.0])
        assert node.node_id == 7
        assert np.allclose(node.position, [1.0, 2.0, 3.0])

    def test_detection_radius(self):
        node = Node(0, [0, 0, 0], detection_radius=5.0)
        assert node.detection_radius == 5.0


class TestNeighborDetection:
    def test_nearby_nodes_detected(self):
        nodes = [
            Node(0, [0.0, 0.0, 0.0], detection_radius=2.0),
            Node(1, [1.0, 0.0, 0.0], detection_radius=2.0),  # dist=1 → detected
            Node(2, [5.0, 0.0, 0.0], detection_radius=2.0),  # dist=5 → not detected
        ]
        nodes[0].update_neighbors(nodes)
        assert 1 in nodes[0].neighbor_distances
        assert 2 not in nodes[0].neighbor_distances

    def test_self_excluded(self):
        nodes = _make_cluster(n=4)
        nodes[0].update_neighbors(nodes)
        assert 0 not in nodes[0].neighbor_distances

    def test_distance_accuracy(self):
        nodes = [
            Node(0, [0.0, 0.0, 0.0], detection_radius=10.0),
            Node(1, [3.0, 4.0, 0.0], detection_radius=10.0),  # dist=5
        ]
        nodes[0].update_neighbors(nodes, noise_std=0.0)
        assert pytest.approx(nodes[0].neighbor_distances[1], abs=1e-9) == 5.0

    def test_noisy_distance_nonnegative(self):
        nodes = [
            Node(0, [0.0, 0.0, 0.0], detection_radius=10.0),
            Node(1, [0.01, 0.0, 0.0], detection_radius=10.0),
        ]
        for _ in range(50):
            nodes[0].update_neighbors(nodes, noise_std=1.0)
            assert nodes[0].neighbor_distances[1] >= 0.0

    def test_symmetry(self):
        nodes = _make_cluster(n=6, detection_radius=10.0)
        for n in nodes:
            n.update_neighbors(nodes)
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i == j:
                    continue
                if j in ni.neighbor_distances and i in nj.neighbor_distances:
                    assert pytest.approx(ni.neighbor_distances[j], rel=1e-6) == nj.neighbor_distances[i]


class TestLEDControl:
    def test_set_led_on(self):
        node = Node(0, [0, 0, 0])
        node.set_led((1.0, 0.0, 0.5))
        assert node.led_on
        assert np.allclose(node.led_color, [1.0, 0.0, 0.5])

    def test_set_led_clamps(self):
        node = Node(0, [0, 0, 0])
        node.set_led((2.0, -1.0, 0.5))
        assert np.allclose(node.led_color, [1.0, 0.0, 0.5])

    def test_turn_off_preserves_color(self):
        node = Node(0, [0, 0, 0])
        node.set_led((0.8, 0.2, 0.4))
        node.turn_off()
        assert not node.led_on
        assert np.allclose(node.led_color, [0.8, 0.2, 0.4])
