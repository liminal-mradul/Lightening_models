"""Pattern engine: maps visual patterns onto the swarm in relative space.

Patterns are defined in a normalised coordinate system ``[-1, 1]^3``.
The engine normalises each node's estimated position into that space and
then assigns an LED colour according to the active pattern.

Available patterns
------------------
WAVE        – A sinusoidal colour wave that propagates along the X axis.
SPHERE      – Nodes near a spherical shell are lit.
CHECKERBOARD – Alternating colour cubes.
RADIAL      – Rings expanding outward from the centre.
CHASE       – A bright band that sweeps through the swarm.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import List

import numpy as np

from .node import Node


class PatternType(Enum):
    WAVE = auto()
    SPHERE = auto()
    CHECKERBOARD = auto()
    RADIAL = auto()
    CHASE = auto()


# Color palette (RGB, values in [0, 1])
COLORS = {
    "red": np.array([1.0, 0.1, 0.1]),
    "green": np.array([0.1, 1.0, 0.1]),
    "blue": np.array([0.1, 0.4, 1.0]),
    "cyan": np.array([0.0, 0.9, 0.9]),
    "magenta": np.array([1.0, 0.1, 0.9]),
    "yellow": np.array([1.0, 0.9, 0.0]),
    "white": np.array([1.0, 1.0, 1.0]),
    "off": np.array([0.0, 0.0, 0.0]),
}


class PatternEngine:
    """Assigns LED colours to nodes according to a chosen pattern.

    Positions are normalised to ``[-1, 1]^3`` before pattern evaluation so
    that patterns are independent of arena size.

    Parameters
    ----------
    nodes:
        Swarm nodes.  Each node's ``estimated_position`` is used when
        available, otherwise the true ``position`` is used as a fallback.
    """

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes = nodes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, pattern: PatternType, t: float) -> None:
        """Evaluate *pattern* at time *t* and update every node's LED.

        Parameters
        ----------
        pattern:
            Which pattern to display.
        t:
            Simulation time (seconds / ticks).
        """
        norm_positions = self._normalised_positions()

        dispatch = {
            PatternType.WAVE: self._wave,
            PatternType.SPHERE: self._sphere,
            PatternType.CHECKERBOARD: self._checkerboard,
            PatternType.RADIAL: self._radial,
            PatternType.CHASE: self._chase,
        }
        fn = dispatch.get(pattern)
        if fn is None:
            raise ValueError(f"Unknown pattern: {pattern}")
        fn(norm_positions, t)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _normalised_positions(self) -> np.ndarray:
        """Return ``(N, 3)`` positions normalised to ``[-1, 1]^3``."""
        raw = np.array(
            [
                n.estimated_position if n.estimated_position is not None else n.position
                for n in self.nodes
            ],
            dtype=float,
        )
        if raw.size == 0:
            return raw
        # Per-axis min-max normalisation
        mins = raw.min(axis=0)
        maxs = raw.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0  # Avoid division by zero for degenerate axes
        return 2.0 * (raw - mins) / ranges - 1.0  # → [-1, 1]

    # ------------------------------------------------------------------
    # Individual pattern implementations
    # ------------------------------------------------------------------

    def _wave(self, pos: np.ndarray, t: float) -> None:
        """Sinusoidal wave propagating along the X axis with a colour gradient."""
        freq = 2.0 * math.pi
        speed = 1.5
        for i, node in enumerate(self.nodes):
            x = pos[i, 0]
            phase = freq * x - speed * t
            brightness = 0.5 + 0.5 * math.sin(phase)
            # Color transitions from blue (trough) to cyan (peak)
            color = (1.0 - brightness) * COLORS["blue"] + brightness * COLORS["cyan"]
            node.set_led(tuple(color), on=True)

    def _sphere(self, pos: np.ndarray, t: float) -> None:
        """Animate a pulsating spherical shell."""
        pulse_radius = 0.5 + 0.3 * math.sin(t * 1.2)
        shell_width = 0.18
        for i, node in enumerate(self.nodes):
            r = float(np.linalg.norm(pos[i]))
            dist_to_shell = abs(r - pulse_radius)
            if dist_to_shell < shell_width:
                brightness = 1.0 - dist_to_shell / shell_width
                color = (1.0 - brightness) * COLORS["blue"] + brightness * COLORS["magenta"]
                node.set_led(tuple(color), on=True)
            else:
                # Dim background glow
                node.set_led((0.03, 0.03, 0.08), on=True)

    def _checkerboard(self, pos: np.ndarray, t: float) -> None:
        """3-D checkerboard of alternating colours that slowly rotates hue."""
        cell_size = 0.4
        hue_shift = t * 0.3
        for i, node in enumerate(self.nodes):
            ix = int(math.floor((pos[i, 0] + 1.0) / cell_size))
            iy = int(math.floor((pos[i, 1] + 1.0) / cell_size))
            iz = int(math.floor((pos[i, 2] + 1.0) / cell_size))
            parity = (ix + iy + iz) % 2
            if parity == 0:
                r = 0.5 + 0.5 * math.cos(hue_shift)
                g = 0.5 + 0.5 * math.cos(hue_shift + 2.094)
                b = 0.5 + 0.5 * math.cos(hue_shift + 4.189)
                node.set_led((r, g, b), on=True)
            else:
                node.set_led((0.05, 0.05, 0.05), on=True)

    def _radial(self, pos: np.ndarray, t: float) -> None:
        """Concentric rings expanding from the centre."""
        ring_spacing = 0.35
        ring_width = 0.12
        speed = 0.8
        for i, node in enumerate(self.nodes):
            r = float(np.linalg.norm(pos[i, :2]))  # radial distance in XY plane
            phase = r / ring_spacing - speed * t
            brightness = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
            color = brightness * COLORS["yellow"] + (1.0 - brightness) * COLORS["red"]
            node.set_led(tuple(color), on=True)

    def _chase(self, pos: np.ndarray, t: float) -> None:
        """A bright band that sweeps through the swarm along the Z axis."""
        band_pos = math.sin(t * 0.8)  # oscillates between -1 and 1
        band_half_width = 0.15
        for i, node in enumerate(self.nodes):
            z = pos[i, 2]
            dist = abs(z - band_pos)
            if dist < band_half_width:
                brightness = 1.0 - dist / band_half_width
                node.set_led(tuple(brightness * COLORS["white"]), on=True)
            else:
                # Faint residual glow based on proximity
                glow = max(0.0, 0.08 * (1.0 - dist))
                node.set_led((glow * 0.5, glow * 0.5, glow), on=True)
