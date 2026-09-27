"""Tests for the time-based train/test split in src/models/split.py."""

import pandas as pd

from src.models.split import split_cutoff, train_test_split_by_time


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
