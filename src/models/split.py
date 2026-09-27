"""Time-based validation: the train/test split and time-ordered cross-validation folds.

* **Test split.** The most recent ``split.test_fraction`` of claims by accident date form the
  final test set. Every claim at or after the cutoff timestamp is test, so the test period is
  strictly later than the training period (ties at the cutoff all go to test).
* **Cross-validation.** Inside the training period, claims are cut into ``cv_folds + 1``
  consecutive blocks of equal size. Fold k trains on blocks 0..k-1 and validates on block k,
  so every validation block is strictly later than the data its model was trained on. This
  mirrors deployment: a model always predicts claims from after its training data.
* **Inner holdout.** The most recent ``inner_holdout_fraction`` of any training set is used
  to estimate things that must be measured out of time (the smearing factor and the
  probability calibration), again with the model fit on the earlier part only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config


def split_cutoff(dates: pd.Series, test_fraction: float) -> pd.Timestamp:
    """Earliest accident timestamp of the most recent ``test_fraction`` of claims.

    Args:
        dates: Accident timestamps.
        test_fraction: Share of claims (by count) to hold out as the most recent period.

    Returns:
        The timestamp at which the most recent period starts.
    """
    ordered = dates.sort_values(kind="mergesort").reset_index(drop=True)
    n_test = round(len(ordered) * test_fraction)
    return ordered.iloc[len(ordered) - n_test]


def recent_mask(dates: pd.Series, fraction: float) -> pd.Series:
    """True for the most recent ``fraction`` of rows by date (ties at the cutoff included)."""
    return dates >= split_cutoff(dates, fraction)


def holdout_mask(df: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Timestamp]:
    """Boolean mask of test-period claims, and the cutoff timestamp used.

    Args:
        df: Cleaned claims table.
        cfg: Loaded project configuration.

    Returns:
        ``True`` for test-period rows, and the cutoff.
    """
    dates = df[cfg.col(cfg["split"]["date_column"])]
    cutoff = split_cutoff(dates, cfg["split"]["test_fraction"])
    return dates >= cutoff, cutoff


def train_test_split_by_time(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split claims into an earlier training period and a later test period.

    Returns:
        Training rows, test rows and the cutoff timestamp.
    """
    is_test, cutoff = holdout_mask(df, cfg)
    return df.loc[~is_test], df.loc[is_test], cutoff


def fold_boundaries(dates: pd.Series, n_folds: int) -> list[pd.Timestamp]:
    """Start timestamps of the ``n_folds`` validation blocks (equal-size blocks by count)."""
    ordered = dates.sort_values(kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    return [ordered.iloc[round(n * k / (n_folds + 1))] for k in range(1, n_folds + 1)]


def time_ordered_folds(dates: pd.Series, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: train on everything before a block, validate on the block.

    Args:
        dates: Accident timestamps of the training claims.
        n_folds: Number of folds.

    Returns:
        ``(train_positions, validation_positions)`` pairs of positional indices.

    Raises:
        ValueError: If ties in the dates leave a fold without training or validation rows.
    """
    bounds = fold_boundaries(dates, n_folds)
    values = dates.to_numpy()
    folds = []
    for k, start in enumerate(bounds):
        in_block = values >= start
        if k + 1 < len(bounds):
            in_block &= values < bounds[k + 1]
        train, validation = np.flatnonzero(values < start), np.flatnonzero(in_block)
        if len(train) == 0 or len(validation) == 0:
            raise ValueError(f"Fold {k + 1} is empty; too many tied dates for {n_folds} folds")
        folds.append((train, validation))
    return folds


def period_summary(dates: pd.Series) -> dict[str, str]:
    """First and last accident date of a set of claims, as ISO dates."""
    return {"start": f"{dates.min():%Y-%m-%d}", "end": f"{dates.max():%Y-%m-%d}"}
