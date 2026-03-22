"""Visualization: 3-D matplotlib rendering of the swarm."""

from __future__ import annotations

from typing import Optional, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (works in all environments)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

from .simulation import Simulation


class SwarmVisualizer:
    """3-D scatter-plot renderer for the swarm.

    Parameters
    ----------
    sim:
        The :class:`~swarm.simulation.Simulation` to visualise.
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.
    node_size:
        Marker size for each node in the scatter plot.
    """

    def __init__(
        self,
        sim: Simulation,
        figsize: tuple = (10, 8),
        node_size: float = 60.0,
    ) -> None:
        if not _MPL_AVAILABLE:  # pragma: no cover
            raise RuntimeError("matplotlib is required for visualization.")
        self.sim = sim
        self.node_size = node_size

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._scatter = None
        self._title_text = None

        bounds = sim.env.bounds
        self.ax.set_xlim(-bounds[0], bounds[0])
        self.ax.set_ylim(-bounds[1], bounds[1])
        self.ax.set_zlim(-bounds[2], bounds[2])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_facecolor("#0a0a0a")
        self.fig.patch.set_facecolor("#0a0a0a")
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333333")
        self.ax.tick_params(colors="#888888")
        self.ax.xaxis.label.set_color("#888888")
        self.ax.yaxis.label.set_color("#888888")
        self.ax.zaxis.label.set_color("#888888")

    # ------------------------------------------------------------------

    def render_frame(self) -> None:
        """Redraw the current swarm state."""
        positions = self.sim.env.get_positions()
        colors = self.sim.env.get_led_colors()

        # Nodes with led_on=False use a dim grey
        display_colors = []
        for i, node in enumerate(self.sim.env.nodes):
            if node.led_on:
                display_colors.append(colors[i])
            else:
                display_colors.append(np.array([0.15, 0.15, 0.15]))
        display_colors = np.array(display_colors)

        if self._scatter is not None:
            self._scatter.remove()

        self._scatter = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=display_colors,
            s=self.node_size,
            depthshade=True,
            alpha=0.9,
        )

        pattern_name = self.sim.current_pattern.name
        title = (
            f"Swarm Visualization  |  Pattern: {pattern_name}"
            f"  |  t={self.sim.t:.1f}  |  N={len(self.sim.env.nodes)}"
        )
        if self._title_text is not None:
            self._title_text.set_text(title)
        else:
            self._title_text = self.ax.set_title(title, color="white", fontsize=11)

    # ------------------------------------------------------------------

    def save_snapshot(self, filepath: str) -> None:
        """Save the current rendered frame to *filepath* (PNG/PDF/SVG etc.)."""
        self.render_frame()
        self.fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor=self.fig.get_facecolor())

    # ------------------------------------------------------------------

    def run_animation(
        self,
        n_steps: int,
        interval_ms: int = 50,
        save_path: Optional[str] = None,
        fps: int = 20,
    ) -> None:
        """Run the simulation while updating the plot in real time.

        Parameters
        ----------
        n_steps:
            Number of simulation steps to animate.
        interval_ms:
            Milliseconds between frames (for interactive display).
        save_path:
            If provided, save the animation to this file (requires ffmpeg or
            pillow depending on format).
        fps:
            Frames per second when saving.
        """
        from matplotlib.animation import FuncAnimation

        def _update(_frame):
            self.sim.step()
            self.render_frame()
            return (self._scatter,)

        anim = FuncAnimation(
            self.fig,
            _update,
            frames=n_steps,
            interval=interval_ms,
            blit=False,
        )

        if save_path is not None:
            anim.save(save_path, fps=fps, dpi=100)
        else:  # pragma: no cover
            plt.show()

        plt.close(self.fig)
