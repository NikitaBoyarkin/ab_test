# -*- coding: utf-8 -*-
"""Shared matplotlib styling and figure I/O for the A/B toolkit.

Every figure goes through this module so the whole gallery shares one look: a
headless-safe backend, a colorblind-safe palette, constrained layout, and a
single `save_fig` that applies dpi / tight bbox / close. Plotting code should
use the object-oriented interface (`fig, ax = plt.subplots()`), never the
global pyplot state.

The plotnine figures in `sequential_ab_testing.py` are independent and keep
their own ggplot theme; matplotlib is used for the rest of the gallery.
"""

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files, never open a GUI

from pathlib import Path

import matplotlib.pyplot as plt

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

# Okabe-Ito colorblind-safe qualitative palette.
CB_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
CONTROL_COLOR = CB_PALETTE[0]
TREAT_COLOR = CB_PALETTE[1]
NEUTRAL_COLOR = "#666666"

PRESETS = {
    "light": {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1A1A1A",
        "axes.titlecolor": "#1A1A1A",
        "text.color": "#1A1A1A",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#DDDDDD",
    },
    "dark": {
        "figure.facecolor": "#1E1E1E",
        "axes.facecolor": "#1E1E1E",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#F0F0F0",
        "axes.titlecolor": "#F0F0F0",
        "text.color": "#F0F0F0",
        "xtick.color": "#CCCCCC",
        "ytick.color": "#CCCCCC",
        "grid.color": "#444444",
    },
}


def apply_style(preset: str = "light") -> None:
    """Set the shared rcParams. Call once at the top of a figure script."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, expected one of {sorted(PRESETS)}")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.figsize": (8, 5),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.5,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=CB_PALETTE),
            **PRESETS[preset],
        }
    )


def new_axes(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None, **kwargs):
    """Create a figure with constrained layout, returning `(fig, ax)`.

    With a single subplot `ax` is an `Axes`; with a grid it is the numpy array
    of axes returned by `plt.subplots`.
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", **kwargs)
    return fig, ax


def save_fig(fig, name: str, out_dir: Path | None = None) -> Path:
    """Save `fig` as `PLOTS_DIR/<name>`, then close it to free memory."""
    out_dir = Path(out_dir) if out_dir is not None else PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    return path


def annotate_bars(ax, fmt: str = "{:.3g}", dy: float = 0.0, **kwargs) -> None:
    """Label every bar in a bar chart with its height value."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=2 + dy, **kwargs)
