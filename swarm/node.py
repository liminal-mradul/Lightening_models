"""Node: represents a single wearable device in the swarm."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


class Node:
    """A single wearable device with an LED and local communication.

    Each node knows only its own state and a set of *relative* distance
    measurements to neighbouring nodes.  It has no access to global
    coordinates or GPS.

    Attributes
    ----------
    node_id : int
        Unique identifier within the swarm.
    position : np.ndarray
        True 3-D position used only by the simulator to compute ground-truth
        distances (not available to pattern-assignment logic).
    detection_radius : float
        Maximum communication / ranging distance.
    led_color : np.ndarray
        RGB colour of the LED, values in [0, 1].
    led_on : bool
        Whether the LED is currently illuminated.
    estimated_position : np.ndarray or None
        Local relative position estimate produced by the positioning layer.
    neighbor_distances : Dict[int, float]
        Mapping ``{neighbor_id: distance}`` for every node within
        ``detection_radius``.
    """

    def __init__(
        self,
        node_id: int,
        position: np.ndarray,
        detection_radius: float = 3.0,
    ) -> None:
        self.node_id: int = node_id
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.detection_radius: float = float(detection_radius)

        self.led_color: np.ndarray = np.zeros(3)
        self.led_on: bool = False

        self.estimated_position: Optional[np.ndarray] = None
        self.neighbor_distances: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Neighbour detection
    # ------------------------------------------------------------------

    def update_neighbors(self, all_nodes: List["Node"], noise_std: float = 0.0) -> None:
        """Refresh ``neighbor_distances`` from the current true positions.

        Parameters
        ----------
        all_nodes:
            Every node in the swarm (including self – self is skipped).
        noise_std:
            Standard deviation of Gaussian noise added to each distance
            measurement, simulating measurement uncertainty.
        """
        self.neighbor_distances = {}
        for other in all_nodes:
            if other.node_id == self.node_id:
                continue
            dist = float(np.linalg.norm(self.position - other.position))
            if dist <= self.detection_radius:
                if noise_std > 0.0:
                    dist = max(0.0, dist + float(np.random.normal(0.0, noise_std)))
                self.neighbor_distances[other.node_id] = dist

    # ------------------------------------------------------------------
    # LED control
    # ------------------------------------------------------------------

    def set_led(self, color: Tuple[float, float, float], on: bool = True) -> None:
        """Set the LED colour and state.

        Parameters
        ----------
        color:
            RGB tuple with components in [0, 1].
        on:
            Whether the LED should be illuminated.
        """
        self.led_color = np.clip(np.asarray(color, dtype=float), 0.0, 1.0)
        self.led_on = on

    def turn_off(self) -> None:
        """Turn the LED off (colour is preserved)."""
        self.led_on = False

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Node(id={self.node_id}, pos={self.position.round(2)}, "
            f"led={'ON' if self.led_on else 'OFF'}, "
            f"neighbors={list(self.neighbor_distances.keys())})"
        )
