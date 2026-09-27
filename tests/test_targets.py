"""Tests for the target transformations in src/models/targets.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.targets import LOG_TARGETS, make_target, smearing_factor


def frame(cfg, dates, initial):
    return pd.DataFrame(
        {cfg.col("accident_date"): pd.to_datetime(dates), cfg.col("initial_estimate"): initial}
    )


@pytest.fixture
def claims(cfg):
    dates = [f"{year}-06-01" for year in range(1995, 2003)]
    initial = np.linspace(1000, 3000, len(dates))
    costs = 1500 * 1.1 ** np.arange(len(dates))  # 10% a year growth
    return frame(cfg, dates, initial), costs


@pytest.mark.parametrize("name", ["raw", *LOG_TARGETS])
def test_inverse_undoes_forward(cfg, claims, name):
    rows, costs = claims
    target = make_target(name, cfg).fit(rows, costs)
    assert target.inverse(rows, target.forward(rows, costs)) == pytest.approx(costs)


def test_log_ratio_is_relative_to_initial_estimate(cfg, claims):
    rows, costs = claims
    z = make_target("log_ratio", cfg).fit(rows, costs).forward(rows, costs)
    assert z == pytest.approx(np.log(costs / rows[cfg.col("initial_estimate")]))


def test_trend_recovers_known_growth_and_extrapolates(cfg, claims):
    rows, costs = claims
    target = make_target("log_trend", cfg).fit(rows, costs)
    assert np.expm1(target.slope_) == pytest.approx(0.1, abs=0.01)
    # A claim two years after the training data: the same deflated target maps to a
    # higher cost, because the trend continues.
    later = frame(cfg, ["2004-06-01"], [3000.0])
    latest = rows.iloc[[-1]]
    z = target.forward(latest, costs[-1:])
    assert target.inverse(later, z)[0] > costs[-1] * 1.15


def test_trend_uses_only_the_recent_window(cfg):
    window = cfg["regression"]["trend_window_years"]
    years = np.arange(1990, 2003)
    growth = np.where(years >= 2002 - window, 1.2, 1.0)  # flat, then 20% a year
    costs = 1000 * np.cumprod(growth)
    rows = frame(cfg, [f"{year}-06-01" for year in years], np.full(len(years), 1000.0))
    target = make_target("log_ratio_trend", cfg).fit(rows, costs)
    assert np.exp(target.slope_) == pytest.approx(1.2, abs=0.01)


def test_unknown_target_raises(cfg):
    with pytest.raises(ValueError, match="Unknown target"):
        make_target("nope", cfg)


def test_smearing_factor():
    assert smearing_factor(np.zeros(5)) == pytest.approx(1.0)
    residuals = np.array([-0.5, 0.5])
    assert smearing_factor(residuals) == pytest.approx(np.mean(np.exp(residuals)))
    assert smearing_factor(residuals) > 1.0
