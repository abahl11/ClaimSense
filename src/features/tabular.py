"""Tabular feature helpers derived from the cleaned claims table.

Everything here is a fixed row-wise transformation (no statistics learned from data), so it
can be computed before the train/test split without leakage. Phase 3 adds the remaining
features and the scikit-learn pipeline that imputes and scales them on training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.clean import report_delay_days


def band_labels(edges: list[float], integer: bool) -> list[str]:
    """Readable labels for bands that include their lower edge; the last band is open.

    Integer bands are labelled with inclusive ranges ("20-24"); continuous bands with
    their edges ("200-300"). A first band starting at 0 is labelled "<upper".
    """
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo == 0:
            labels.append(f"<{hi:,g}")
        elif integer:
            labels.append(f"{lo:,g}-{hi - 1:,g}")
        else:
            labels.append(f"{lo:,g}-{hi:,g}")
    labels.append(f"{edges[-1]:,g}+")
    return labels


def make_bands(values: pd.Series, edges: list[float], integer: bool = True) -> pd.Series:
    """Cut values into ordered bands ``[edge_i, edge_i+1)`` plus an open last band.

    Args:
        values: Numeric values to band. Missing values stay missing.
        edges: Increasing band edges; the first should be at or below the minimum value.
        integer: Whether values are whole numbers (affects the labels only).

    Returns:
        An ordered categorical series.
    """
    bins = [*edges, np.inf]
    return pd.cut(values, bins=bins, right=False, labels=band_labels(edges, integer))


def accident_year(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Calendar year of the accident."""
    return df[cfg.col("accident_date")].dt.year.rename("accident_year")


def delay_days(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Days between the accident and the report."""
    delay = report_delay_days(df[cfg.col("accident_date")], df[cfg.col("report_date")])
    return delay.rename("report_delay_days")


def age_band(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Age grouped into the bands set in config."""
    edges = cfg["features"]["age_band_edges"]
    return make_bands(df[cfg.col("age")], edges).rename("age_band")


def wage_band(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Weekly wage grouped into the bands set in config."""
    edges = cfg["features"]["wage_band_edges"]
    return make_bands(df[cfg.col("weekly_wages")], edges, integer=False).rename("wage_band")


def delay_band(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Reporting delay grouped into the bands set in config."""
    edges = cfg["features"]["delay_band_edges"]
    return make_bands(delay_days(df, cfg), edges).rename("delay_band")


def total_dependents(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Dependent children plus other dependents."""
    total = df[cfg.col("dependent_children")] + df[cfg.col("dependents_other")]
    return total.rename("total_dependents")


def dependents_group(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Total dependents with the top values grouped (for example "0", "1", "2", "3+")."""
    cap = cfg["features"]["dependents_cap"]
    edges = list(range(cap + 1))
    labels = [str(n) for n in range(cap)] + [f"{cap}+"]
    grouped = pd.cut(total_dependents(df, cfg), bins=[*edges, np.inf], right=False, labels=labels)
    return grouped.rename("dependents_group")
