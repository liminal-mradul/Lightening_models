"""Tests for the Simulation orchestrator."""

import numpy as np
import pytest

from swarm.environment import Environment
from swarm.simulation import Simulation
from swarm.patterns import PatternType


def _small_sim(seed=0, enable_movement=False):
    env = Environment(bounds=(6.0, 6.0, 3.0), num_nodes=20, detection_radius=4.0, seed=seed)
    env.update_all_neighbors()
    sim = Simulation(env, dt=0.1, enable_movement=enable_movement)
    return sim


class TestSimulationInit:
    def test_default_pattern_is_wave(self):
        sim = _small_sim()
        assert sim.current_pattern == PatternType.WAVE

    def test_time_starts_at_zero(self):
        sim = _small_sim()
        assert sim.t == 0.0

    def test_custom_pattern_sequence(self):
        env = Environment(num_nodes=10, seed=1)
        env.update_all_neighbors()
        seq = [(PatternType.CHASE, 0.5), (PatternType.SPHERE, 0.5)]
        sim = Simulation(env, pattern_sequence=seq, dt=0.1)
        assert sim.current_pattern == PatternType.CHASE


class TestSimulationStep:
    def test_step_advances_time(self):
        sim = _small_sim()
        sim.step()
        assert pytest.approx(sim.t) == 0.1

    def test_multiple_steps_advance_time(self):
        sim = _small_sim()
        for _ in range(10):
            sim.step()
        assert pytest.approx(sim.t, abs=1e-9) == 1.0

    def test_step_updates_led_states(self):
        sim = _small_sim()
        sim.step()
        # After one step all nodes should have valid colors
        for node in sim.env.nodes:
            assert np.all(node.led_color >= 0.0)
            assert np.all(node.led_color <= 1.0)

    def test_step_with_movement(self):
        sim = _small_sim(enable_movement=True)
        positions_before = sim.env.get_positions().copy()
        sim.step()
        positions_after = sim.env.get_positions()
        # Positions should have changed
        assert not np.allclose(positions_before, positions_after)

    def test_pattern_switching(self):
        env = Environment(num_nodes=10, seed=2)
        env.update_all_neighbors()
        # Two patterns, each 0.2 time-units long, dt=0.1 → switch after 2 steps
        seq = [(PatternType.WAVE, 0.2), (PatternType.RADIAL, 0.2)]
        sim = Simulation(env, pattern_sequence=seq, dt=0.1)
        assert sim.current_pattern == PatternType.WAVE
        sim.step()
        sim.step()
        assert sim.current_pattern == PatternType.RADIAL

    def test_pattern_cycling(self):
        """After finishing all patterns, the sequence should wrap around."""
        env = Environment(num_nodes=10, seed=3)
        env.update_all_neighbors()
        seq = [(PatternType.WAVE, 0.1), (PatternType.CHASE, 0.1)]
        sim = Simulation(env, pattern_sequence=seq, dt=0.1)
        sim.step()  # WAVE done
        sim.step()  # CHASE done
        # Should be back to WAVE
        assert sim.current_pattern == PatternType.WAVE


class TestSimulationRun:
    def test_run_completes(self):
        sim = _small_sim()
        sim.run(n_steps=5)
        assert pytest.approx(sim.t, abs=1e-9) == 0.5

    def test_run_callback_called(self):
        sim = _small_sim()
        call_count = [0]

        def cb(s):
            call_count[0] += 1

        sim.run(n_steps=7, callback=cb)
        assert call_count[0] == 7
