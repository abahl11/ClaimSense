"""Figures for the modelling stages (validation design, regression and classification)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.reporting.plots import (
    hbar,
    new_figure,
    percent_axis,
    save_figure,
    series_color,
    thousands_axis,
    year_axis,
)


def plot_timeline(periods: pd.DataFrame, cfg: Config) -> Path:
    """One row per fold (and the final split): training span and evaluation span.

    Args:
        periods: Columns ``label``, ``fit_start``, ``fit_end``, ``eval_start``, ``eval_end``
            (decimal years).
        cfg: Loaded project configuration.
    """
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    y = np.arange(len(periods))
    fit_width = periods["fit_end"] - periods["fit_start"]
    eval_width = periods["eval_end"] - periods["eval_start"]
    ax.barh(
        y,
        fit_width,
        left=periods["fit_start"],
        height=0.5,
        color=series_color(cfg, 0),
        label="Training",
    )
    ax.barh(
        y,
        eval_width,
        left=periods["eval_start"],
        height=0.5,
        color=series_color(cfg, 1),
        label="Validation / test",
    )
    ax.set_yticks(y, periods["label"].tolist())
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(True)
    ax.set_xlabel("Accident date")
    ax.set_title("Time-ordered validation: every model is scored on claims after its training data")
    ax.legend(loc="upper right", ncols=2)
    year_axis(ax)
    ax.tick_params(axis="y", colors=p["ink_secondary"])
    return save_figure(fig, cfg, "validation_timeline")


def plot_ranked_scores(
    table: pd.DataFrame, value: str, title: str, name: str, cfg: Config, fmt: str = "{:.3f}"
) -> Path:
    """Horizontal bars of one score per model, best (lowest) at the top."""
    ordered = table.sort_values(value).reset_index(drop=True)
    fig, ax = new_figure(cfg)
    hbar(ax, ordered[value], ordered["name"], fmt, cfg)
    ax.invert_yaxis()
    ax.set_title(title)
    return save_figure(fig, cfg, name)


def plot_bias_by_year(series: dict[str, dict[int, float]], cfg: Config) -> Path:
    """Total predicted / total actual cost per accident year, one line per approach."""
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    for slot, (label, by_year) in enumerate(series.items()):
        years, values = list(by_year), list(by_year.values())
        ax.plot(
            years,
            values,
            color=series_color(cfg, slot),
            label=label,
            marker="o",
            markersize=5,
            markeredgecolor=p["surface"],
            markeredgewidth=1.5,
        )
    ax.axhline(1.0, color=p["baseline"], linewidth=1)
    ax.set_title("Test set: total predicted / total actual cost by accident year")
    ax.set_xlabel("Accident year")
    ax.set_ylabel("Predicted / actual (1.0 = no bias)")
    year_axis(ax)
    ax.legend(loc="upper left")
    return save_figure(fig, cfg, "regression_bias_by_year")


def plot_calibration_deciles(deciles: pd.DataFrame, cfg: Config) -> Path:
    """Mean predicted vs mean actual cost per predicted-cost decile (log axes, y = x)."""
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    ax.plot(
        deciles["mean_predicted"],
        deciles["mean_actual"],
        color=series_color(cfg, 0),
        marker="o",
        markersize=6,
        markeredgecolor=p["surface"],
        markeredgewidth=1.5,
    )
    low = min(deciles["mean_predicted"].min(), deciles["mean_actual"].min()) * 0.8
    high = max(deciles["mean_predicted"].max(), deciles["mean_actual"].max()) * 1.2
    ax.plot([low, high], [low, high], color=p["ink_muted"], linewidth=1)
    ax.text(high, high, "predicted = actual ", ha="right", va="bottom", color=p["ink_secondary"])
    for row in deciles.itertuples():
        ax.text(
            row.mean_predicted,
            row.mean_actual,
            f"  {row.decile}",
            va="center",
            color=p["ink_secondary"],
            fontsize=p["font_size"] - 2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    thousands_axis(ax, "x")
    thousands_axis(ax, "y")
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Mean predicted (expected) cost in the decile")
    ax.set_ylabel("Mean actual cost in the decile")
    ax.set_title("Test set: calibration by predicted-cost decile (numbers are deciles)")
    return save_figure(fig, cfg, "regression_calibration_deciles")


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], base_rate: float, cfg: Config
) -> Path:
    """Precision-recall curves on the test set, one per model."""
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    for slot, (label, (recall, precision)) in enumerate(curves.items()):
        ax.plot(recall, precision, color=series_color(cfg, slot), label=label)
    ax.axhline(base_rate, color=p["ink_muted"], linewidth=1)
    ax.text(
        1.0,
        base_rate,
        f"random review ({base_rate:.0%}) ",
        ha="right",
        va="bottom",
        color=p["ink_secondary"],
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Recall (share of high-cost claims caught)")
    ax.set_ylabel("Precision (share of flagged claims that are high-cost)")
    ax.set_title("Test set: precision-recall for the high-cost flag")
    ax.legend(loc="upper right")
    return save_figure(fig, cfg, "classifier_pr_curves")


def plot_reliability(curves: dict[str, pd.DataFrame], cfg: Config) -> Path:
    """Observed high-cost rate against mean predicted probability, one line per curve."""
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    for slot, (label, table) in enumerate(curves.items()):
        ax.plot(
            table["mean_probability"],
            table["observed_rate"],
            color=series_color(cfg, slot),
            label=label,
            marker="o",
            markersize=5,
            markeredgecolor=p["surface"],
            markeredgewidth=1.5,
        )
    ax.plot([0, 1], [0, 1], color=p["ink_muted"], linewidth=1)
    ax.text(1.0, 1.0, "perfect calibration ", ha="right", va="bottom", color=p["ink_secondary"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Mean predicted probability in the bin")
    ax.set_ylabel("Observed share of high-cost claims")
    ax.set_title("Reliability of the high-cost probabilities (equal-count bins)")
    ax.legend(loc="upper left")
    return save_figure(fig, cfg, "classifier_reliability")
