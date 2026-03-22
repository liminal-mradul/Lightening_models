"""Interactive 3-D viewer: Plotly-based HTML export for the swarm."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOTLY_AVAILABLE = False

from .simulation import Simulation


class SwarmInteractiveViewer:
    """Generate an interactive 3-D HTML visualization of the swarm using Plotly.

    The viewer captures snapshots of the simulation at each step and renders
    them as an animated, fully interactive 3-D scatter plot embedded in a
    single self-contained HTML file.  The resulting file can be opened in any
    modern web browser and supports rotation, zoom, pan, and frame-by-frame
    playback.

    Parameters
    ----------
    sim:
        The :class:`~swarm.simulation.Simulation` instance to visualise.
    node_size:
        Marker size for each swarm node.
    """

    _BG_COLOR = "#0a0a0a"
    _GRID_COLOR = "#333333"
    _AXIS_COLOR = "#888888"

    def __init__(self, sim: Simulation, node_size: float = 6.0) -> None:
        if not _PLOTLY_AVAILABLE:  # pragma: no cover
            raise RuntimeError(
                "plotly is required for the interactive viewer. "
                "Install it with: pip install plotly"
            )
        self.sim = sim
        self.node_size = node_size

    # ------------------------------------------------------------------

    def _current_frame(self, label: str) -> go.Frame:
        """Capture the current simulation state as a Plotly animation frame."""
        positions = self.sim.env.get_positions()
        colors = self._node_colors_hex()

        scatter = go.Scatter3d(
            x=positions[:, 0].tolist(),
            y=positions[:, 1].tolist(),
            z=positions[:, 2].tolist(),
            mode="markers",
            marker=dict(
                size=self.node_size,
                color=colors,
                opacity=0.9,
            ),
            hovertemplate=(
                "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            ),
        )
        return go.Frame(data=[scatter], name=label)

    # ------------------------------------------------------------------

    def _node_colors_hex(self) -> List[str]:
        """Return per-node LED colours as HTML hex strings."""
        hex_colors: List[str] = []
        for node in self.sim.env.nodes:
            if node.led_on:
                r, g, b = node.led_color
            else:
                r, g, b = 0.15, 0.15, 0.15
            ri, gi, bi = (
                int(round(r * 255)),
                int(round(g * 255)),
                int(round(b * 255)),
            )
            hex_colors.append(f"#{ri:02x}{gi:02x}{bi:02x}")
        return hex_colors

    # ------------------------------------------------------------------

    def _axis_style(self, title: str) -> dict:
        bounds = self.sim.env.bounds
        _range_map = {"X": bounds[0], "Y": bounds[1], "Z": bounds[2]}
        half = _range_map.get(title, bounds[0])
        return dict(
            title=title,
            range=[-half, half],
            backgroundcolor=self._BG_COLOR,
            gridcolor=self._GRID_COLOR,
            showbackground=True,
            color=self._AXIS_COLOR,
            zerolinecolor=self._GRID_COLOR,
        )

    # ------------------------------------------------------------------

    def save_html(
        self,
        filepath: str,
        n_steps: int = 60,
        frame_duration_ms: int = 80,
    ) -> None:
        """Run the simulation and save an interactive 3-D HTML file.

        Parameters
        ----------
        filepath:
            Destination ``.html`` file path.
        n_steps:
            Number of simulation steps to capture as animation frames.
        frame_duration_ms:
            Milliseconds each frame is displayed during playback.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        bounds = self.sim.env.bounds
        frames: List[go.Frame] = []

        # Capture initial state
        self.sim.step()
        frames.append(self._current_frame("0"))

        for i in range(1, n_steps):
            self.sim.step()
            frames.append(self._current_frame(str(i)))

        # Build initial data from the last captured frame
        initial_data = frames[0].data

        fig = go.Figure(
            data=list(initial_data),
            frames=frames,
            layout=go.Layout(
                title=dict(
                    text=(
                        f"Swarm LED – Interactive 3-D View"
                        f"  |  N={len(self.sim.env.nodes)} nodes"
                        f"  |  {n_steps} steps"
                    ),
                    font=dict(color="white", size=14),
                ),
                paper_bgcolor=self._BG_COLOR,
                scene=dict(
                    xaxis=self._axis_style("X"),
                    yaxis=self._axis_style("Y"),
                    zaxis=self._axis_style("Z"),
                    bgcolor=self._BG_COLOR,
                    camera=dict(
                        eye=dict(x=1.6, y=1.6, z=1.0),
                    ),
                    aspectmode="cube",
                ),
                font=dict(color=self._AXIS_COLOR),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        y=0.0,
                        x=0.0,
                        xanchor="left",
                        yanchor="bottom",
                        pad=dict(t=10, r=10),
                        buttons=[
                            dict(
                                label="▶  Play",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=frame_duration_ms, redraw=True),
                                        fromcurrent=True,
                                        mode="immediate",
                                        transition=dict(duration=0),
                                    ),
                                ],
                            ),
                            dict(
                                label="⏸  Pause",
                                method="animate",
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0),
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        steps=[
                            dict(
                                method="animate",
                                args=[
                                    [str(k)],
                                    dict(
                                        frame=dict(duration=frame_duration_ms, redraw=True),
                                        mode="immediate",
                                        transition=dict(duration=0),
                                    ),
                                ],
                                label=str(k),
                            )
                            for k in range(n_steps)
                        ],
                        transition=dict(duration=0),
                        x=0.05,
                        y=0.02,
                        len=0.90,
                        bgcolor="#1a1a2e",
                        bordercolor="#444",
                        font=dict(color=self._AXIS_COLOR, size=9),
                        currentvalue=dict(
                            prefix="Step: ",
                            font=dict(color="white", size=11),
                            visible=True,
                            xanchor="center",
                        ),
                    )
                ],
                margin=dict(l=0, r=0, t=40, b=80),
            ),
        )

        fig.write_html(filepath, include_plotlyjs="cdn", full_html=True)
        print(f"[viz] Interactive 3-D HTML saved → {filepath}")

    # ------------------------------------------------------------------

    def snapshot_html(self, filepath: str) -> None:
        """Save a static interactive 3-D snapshot (no animation) to *filepath*.

        This renders the current simulation state only, producing a smaller
        HTML file.  The 3-D view is still fully interactive (rotate/zoom/pan).

        Parameters
        ----------
        filepath:
            Destination ``.html`` file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        positions = self.sim.env.get_positions()
        colors = self._node_colors_hex()
        pattern_name = self.sim.current_pattern.name

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=positions[:, 0].tolist(),
                    y=positions[:, 1].tolist(),
                    z=positions[:, 2].tolist(),
                    mode="markers",
                    marker=dict(
                        size=self.node_size,
                        color=colors,
                        opacity=0.9,
                    ),
                    hovertemplate=(
                        "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                title=dict(
                    text=(
                        f"Swarm LED – Pattern: {pattern_name}"
                        f"  |  N={len(self.sim.env.nodes)} nodes"
                        f"  |  t={self.sim.t:.1f}"
                    ),
                    font=dict(color="white", size=14),
                ),
                paper_bgcolor=self._BG_COLOR,
                scene=dict(
                    xaxis=self._axis_style("X"),
                    yaxis=self._axis_style("Y"),
                    zaxis=self._axis_style("Z"),
                    bgcolor=self._BG_COLOR,
                    camera=dict(
                        eye=dict(x=1.6, y=1.6, z=1.0),
                    ),
                    aspectmode="cube",
                ),
                font=dict(color=self._AXIS_COLOR),
                margin=dict(l=0, r=0, t=40, b=20),
            ),
        )

        fig.write_html(filepath, include_plotlyjs="cdn", full_html=True)
        print(f"[viz] Interactive 3-D snapshot HTML saved → {filepath}")
