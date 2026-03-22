"""Tests for the PatternEngine."""

import numpy as np
import pytest

from swarm.environment import Environment
from swarm.positioning import RelativePositioner
from swarm.patterns import PatternEngine, PatternType, COLORS


def _ready_env(n=30, seed=0):
    env = Environment(bounds=(8.0, 8.0, 4.0), num_nodes=n, detection_radius=5.0, seed=seed)
    env.update_all_neighbors()
    positioner = RelativePositioner(env.nodes)
    positioner.update()
    return env


@pytest.fixture
def bw_image():
    return np.array([[0.0, 1.0], [1.0, 0.0]])


class TestPatternEngineApply:
    @pytest.mark.parametrize("pattern", list(PatternType))
    def test_all_patterns_run(self, pattern, bw_image):
        env = _ready_env()
        engine = PatternEngine(env.nodes)
        if pattern is PatternType.IMAGE:
            engine.set_image_pattern(bw_image)
        # Should not raise
        engine.apply(pattern, t=0.0)
        engine.apply(pattern, t=2.5)

    @pytest.mark.parametrize("pattern", list(PatternType))
    def test_led_colors_in_range(self, pattern, bw_image):
        env = _ready_env()
        engine = PatternEngine(env.nodes)
        if pattern is PatternType.IMAGE:
            engine.set_image_pattern(bw_image)
        engine.apply(pattern, t=1.0)
        for node in env.nodes:
            assert np.all(node.led_color >= 0.0)
            assert np.all(node.led_color <= 1.0)

    def test_invalid_pattern_raises(self):
        env = _ready_env()
        engine = PatternEngine(env.nodes)
        with pytest.raises((ValueError, KeyError)):
            engine.apply("not_a_pattern", t=0.0)  # type: ignore

    def test_wave_changes_over_time(self):
        env = _ready_env()
        engine = PatternEngine(env.nodes)
        engine.apply(PatternType.WAVE, t=0.0)
        colors_t0 = np.array([n.led_color.copy() for n in env.nodes])
        engine.apply(PatternType.WAVE, t=3.14)
        colors_t1 = np.array([n.led_color.copy() for n in env.nodes])
        # At least some colours should have changed
        assert not np.allclose(colors_t0, colors_t1)

    def test_pattern_uses_estimated_position_when_available(self):
        env = _ready_env()
        engine = PatternEngine(env.nodes)
        # Manually set estimated positions to a fixed value → all nodes same colour for CHASE
        for node in env.nodes:
            node.estimated_position = np.array([0.0, 0.0, 0.0])
        engine.apply(PatternType.CHASE, t=0.0)
        # All nodes are at centre (z=0); should all have same colour
        colors = [node.led_color.copy() for node in env.nodes]
        for c in colors:
            assert np.allclose(c, colors[0]), "All central nodes should have the same color"

    def test_sphere_some_nodes_lit(self):
        env = _ready_env(n=50)
        engine = PatternEngine(env.nodes)
        engine.apply(PatternType.SPHERE, t=0.0)
        lit = [n for n in env.nodes if n.led_on]
        assert len(lit) > 0

    def test_image_pattern_respects_threshold_and_invert(self, bw_image):
        env = _ready_env(n=4)
        engine = PatternEngine(env.nodes)
        engine.set_image_pattern(bw_image, threshold=0.6, invert=True)
        coords = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        for node, coord in zip(env.nodes, coords):
            node.estimated_position = coord
        engine.apply(PatternType.IMAGE, t=0.0)
        states = [n.led_on for n in env.nodes]
        assert states == [False, True, True, False]
