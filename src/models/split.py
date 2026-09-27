"""Time-based train/test split.

The most recent ``split.test_fraction`` of claims by accident date form the final test set.
Every claim at or after the cutoff timestamp is test, so the test period is strictly later
than the training period (ties at the cutoff all go to test). Phase 4 adds the time-ordered
cross-validation folds used inside the training period.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config


def split_cutoff(dates: pd.Series, test_fraction: float) -> pd.Timestamp:
    """Earliest accident timestamp of the test period.

    Args:
        dates: Accident timestamps of every claim.
        test_fraction: Share of claims (by count) to hold out as the most recent period.

    Returns:
        The timestamp at the start of the most recent ``test_fraction`` of claims.
    """
    ordered = dates.sort_values(kind="mergesort").reset_index(drop=True)
    n_test = round(len(ordered) * test_fraction)
    return ordered.iloc[len(ordered) - n_test]


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
