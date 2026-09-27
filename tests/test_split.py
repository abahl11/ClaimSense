"""Tests for the time-based train/test split in src/models/split.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.split import (
    recent_mask,
    split_cutoff,
    time_ordered_folds,
    train_test_split_by_time,
)


def claims_on(dates: list[str], cfg) -> pd.DataFrame:
    return pd.DataFrame(
        {
            cfg.col("claim_id"): [f"WC{i:07d}" for i in range(len(dates))],
            cfg.col("accident_date"): pd.to_datetime(dates),
        }
    )


def test_cutoff_holds_out_most_recent_fraction():
    dates = pd.Series(pd.date_range("2000-01-01", periods=10, freq="D"))
    assert split_cutoff(dates, 0.2) == pd.Timestamp("2000-01-09")


def test_cutoff_ignores_input_order():
    dates = pd.Series(pd.date_range("2000-01-01", periods=10, freq="D")).sample(
        frac=1, random_state=0
    )
    assert split_cutoff(dates, 0.2) == pd.Timestamp("2000-01-09")


def test_no_overlap_and_test_strictly_later(cfg):
    dates = [f"2000-01-{day:02d}" for day in range(1, 21)]
    claims = claims_on(dates, cfg)
    train, test, cutoff = train_test_split_by_time(claims, cfg)
    accident = cfg.col("accident_date")
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(claims)
    assert train[accident].max() < test[accident].min()
    assert (test[accident] >= cutoff).all()


def test_ties_at_cutoff_all_go_to_test(cfg):
    dates = ["2000-01-01"] * 6 + ["2000-01-02"] * 4
    train, test, _ = train_test_split_by_time(claims_on(dates, cfg), cfg)
    accident = cfg.col("accident_date")
    assert train[accident].max() < test[accident].min()
    assert (test[accident] == pd.Timestamp("2000-01-02")).all()


# ---------------------------------------------------------------------------
# Time-ordered cross-validation folds
# ---------------------------------------------------------------------------


def daily_dates(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2000-01-01", periods=n, freq="D"))


def test_folds_are_expanding_and_strictly_later():
    dates = daily_dates(60)
    folds = time_ordered_folds(dates, 5)
    assert len(folds) == 5
    previous_train = 0
    for train, validation in folds:
        assert set(train).isdisjoint(validation)
        assert dates.iloc[train].max() < dates.iloc[validation].min()
        assert len(train) > previous_train
        previous_train = len(train)


def test_folds_cover_every_block_after_the_first():
    dates = daily_dates(60)
    folds = time_ordered_folds(dates, 5)
    validated = np.concatenate([validation for _, validation in folds])
    assert len(validated) == len(set(validated)) == 50  # 5 of 6 equal blocks
    assert set(folds[0][0]) == set(range(10))


def test_folds_ignore_input_order():
    dates = daily_dates(60)
    shuffled = dates.sample(frac=1, random_state=0).reset_index(drop=True)
    for train, validation in time_ordered_folds(shuffled, 5):
        assert shuffled.iloc[train].max() < shuffled.iloc[validation].min()


def test_tied_dates_never_straddle_a_fold_boundary():
    dates = pd.Series(pd.to_datetime(["2000-01-01"] * 5 + ["2000-01-02"] * 5 + ["2000-01-03"] * 5))
    for train, validation in time_ordered_folds(dates, 2):
        assert dates.iloc[train].max() < dates.iloc[validation].min()


def test_too_many_ties_raise():
    dates = pd.Series(pd.to_datetime(["2000-01-01"] * 10))
    with pytest.raises(ValueError, match="empty"):
        time_ordered_folds(dates, 3)


def test_recent_mask_selects_latest_share():
    mask = recent_mask(daily_dates(10), 0.2)
    assert mask.tolist() == [False] * 8 + [True] * 2
