"""Visualization: 3-D matplotlib rendering of the swarm."""

from __future__ import annotations

import os
from typing import Optional, List

import numpy as np

try:
    import matplotlib

    _INTERACTIVE_REQUESTED = os.environ.get("SWARM_INTERACTIVE", "").lower() in {"1", "true", "yes"}
    if not _INTERACTIVE_REQUESTED and "MPLBACKEND" not in os.environ:
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

    TOPDOWN_SIZE_SCALE = 0.6
    TOPDOWN_GRID_ALPHA = 0.2
    TOPDOWN_GRID_COLOR = "#666666"

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
        display_colors = self._display_colors()

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
            f"Swarm  |  Pattern forming: {pattern_name}"
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

    def save_topdown(self, filepath: str, pattern_name: Optional[str] = None) -> None:
        """Save a 2-D top-down projection (XY plane) to *filepath*.

        Parameters
        ----------
        filepath:
            Destination file path.
        pattern_name:
            Optional pattern name used in the plot title.  When *None* the
            current simulation pattern is used.
        """
        positions = self.sim.env.get_positions()
        display_colors = self._display_colors()
        label = pattern_name or self.sim.current_pattern.name

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=display_colors,
            s=self.node_size * self.TOPDOWN_SIZE_SCALE,
            alpha=0.9,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Pattern: {label}  –  Top-Down View (XY)")
        ax.set_aspect("equal")
        bounds = self.sim.env.bounds
        ax.set_xlim(-bounds[0], bounds[0])
        ax.set_ylim(-bounds[1], bounds[1])
        ax.grid(alpha=self.TOPDOWN_GRID_ALPHA, color=self.TOPDOWN_GRID_COLOR)
        fig.savefig(filepath, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    # ------------------------------------------------------------------

    def run_animation(
        self,
        n_steps: int,
        interval_ms: int = 50,
        save_path: Optional[str] = None,
        fps: int = 20,
        close: bool = True,
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
            pillow depending on format).  GIF files are generated via Pillow.
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
            ext = os.path.splitext(save_path)[1].lower()
            writer = None
            if ext == ".gif":
                writer = "pillow"
                print(f"[viz] Generating GIF animation ({n_steps} frames @ {fps} fps) → {save_path}")
            elif ext in {".mp4", ".m4v"}:
                writer = "ffmpeg"
                print(f"[viz] Generating MP4 animation ({n_steps} frames @ {fps} fps) → {save_path}")
            else:
                print(f"[viz] Generating animation ({n_steps} frames @ {fps} fps) → {save_path}")
            try:
                if writer is not None:
                    anim.save(save_path, fps=fps, dpi=100, writer=writer)
                else:
                    anim.save(save_path, fps=fps, dpi=100)
            except (ValueError, RuntimeError, OSError) as exc:  # pragma: no cover - depends on system codecs
                raise RuntimeError(f"Failed to write animation to {save_path}: {exc}") from exc
            if ext == ".gif":
                print(f"[viz] GIF saved → {save_path}")
            else:
                print(f"[viz] Animation saved → {save_path}")
        else:  # pragma: no cover
            if not _INTERACTIVE_REQUESTED:
                print("[viz] Headless backend active (set SWARM_INTERACTIVE=1 for a GUI window).")
            plt.show()

        if close:
            plt.close(self.fig)

    # ------------------------------------------------------------------

    def _display_colors(self) -> np.ndarray:
        colors = self.sim.env.get_led_colors()
        display_colors: List[np.ndarray] = []
        for i, node in enumerate(self.sim.env.nodes):
            if node.led_on:
                display_colors.append(colors[i])
            else:
                display_colors.append(np.array([0.15, 0.15, 0.15]))
        return np.array(display_colors)
