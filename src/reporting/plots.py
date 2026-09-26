"""Shared figure style and saving helpers, so every report figure looks the same."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # figures are only written to files; no display is needed

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.config import Config  # noqa: E402


def apply_style(cfg: Config) -> None:
    """Set matplotlib defaults from the ``plots`` section of the config."""
    p = cfg["plots"]
    plt.rcParams.update(
        {
            "figure.dpi": p["dpi"],
            "savefig.dpi": p["dpi"],
            "figure.facecolor": p["surface"],
            "axes.facecolor": p["surface"],
            "savefig.facecolor": p["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": p["font_family"],
            "font.size": p["font_size"],
            "text.color": p["ink_primary"],
            "axes.labelcolor": p["ink_secondary"],
            "axes.titlecolor": p["ink_primary"],
            "axes.titlesize": p["font_size"] + 1,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "xtick.color": p["ink_muted"],
            "ytick.color": p["ink_muted"],
            "ytick.major.size": 0,
            "axes.edgecolor": p["baseline"],
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": p["gridline"],
            "grid.linewidth": 1,
            "grid.linestyle": "-",
            "lines.linewidth": p["line_width"],
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
        }
    )


def new_figure(cfg: Config, ncols: int = 1) -> tuple[Figure, Any]:
    """Create a styled figure with ``ncols`` side-by-side axes.

    Args:
        cfg: Loaded project configuration.
        ncols: Number of panels.

    Returns:
        The figure and its axes (a single ``Axes`` or an array of them).
    """
    apply_style(cfg)
    p = cfg["plots"]
    return plt.subplots(1, ncols, figsize=(p["width"], p["height"]), layout="constrained")


def series_color(cfg: Config, slot: int) -> str:
    """Return the categorical colour for a series slot (fixed order, never cycled)."""
    return cfg["plots"]["categorical"][slot]


def thousands_axis(ax: Axes, axis: str = "y") -> None:
    """Format an axis with comma thousands separators and no decimals."""
    formatter = FuncFormatter(lambda value, _pos: f"{value:,.0f}")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(formatter)


def percent_axis(ax: Axes, axis: str = "y") -> None:
    """Format an axis holding fractions (0-1) as whole percentages."""
    formatter = FuncFormatter(lambda value, _pos: f"{value:.0%}")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(formatter)


def save_figure(fig: Figure, cfg: Config, name: str) -> Path:
    """Write a figure to ``reports/figures/<name>.png`` and close it.

    Args:
        fig: Figure to save.
        cfg: Loaded project configuration.
        name: File name without extension.

    Returns:
        Path of the written PNG.
    """
    path = cfg.path("figures_dir") / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
