"""Tests for the interactive Plotly 3-D HTML viewer."""

import os
import tempfile

import pytest

from swarm.environment import Environment
from swarm.simulation import Simulation
from swarm.patterns import PatternType
from swarm.interactive_viewer import SwarmInteractiveViewer


def _small_sim(seed: int = 7) -> Simulation:
    env = Environment(bounds=(6.0, 6.0, 3.0), num_nodes=15, detection_radius=4.0, seed=seed)
    env.update_all_neighbors()
    sim = Simulation(env, pattern_sequence=[(PatternType.WAVE, float("inf"))], dt=0.1)
    for _ in range(5):
        sim.step()
    return sim


class TestSwarmInteractiveViewer:
    def test_save_html_creates_file(self, tmp_path):
        sim = _small_sim()
        iv = SwarmInteractiveViewer(sim)
        out = str(tmp_path / "test_view.html")
        iv.save_html(out, n_steps=5)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000  # non-trivial HTML content

    def test_save_html_contains_plotly(self, tmp_path):
        sim = _small_sim()
        iv = SwarmInteractiveViewer(sim)
        out = str(tmp_path / "view.html")
        iv.save_html(out, n_steps=5)
        with open(out) as f:
            content = f.read()
        # Plotly CDN or inline script should be present
        assert "plotly" in content.lower()

    def test_snapshot_html_creates_file(self, tmp_path):
        sim = _small_sim()
        iv = SwarmInteractiveViewer(sim)
        out = str(tmp_path / "snap.html")
        iv.snapshot_html(out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000

    def test_node_colors_hex_format(self):
        sim = _small_sim()
        iv = SwarmInteractiveViewer(sim)
        colors = iv._node_colors_hex()
        assert len(colors) == len(sim.env.nodes)
        for c in colors:
            assert c.startswith("#")
            assert len(c) == 7  # e.g. "#1a2b3c"

    def test_node_colors_length_matches_nodes(self):
        sim = _small_sim()
        iv = SwarmInteractiveViewer(sim)
        assert len(iv._node_colors_hex()) == len(sim.env.nodes)

    def test_save_html_creates_parent_dirs(self, tmp_path):
        sim = _small_sim()
        iv = SwarmInteractiveViewer(sim)
        nested = str(tmp_path / "a" / "b" / "view.html")
        iv.save_html(nested, n_steps=3)
        assert os.path.exists(nested)
