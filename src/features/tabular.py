"""Tabular features derived from the cleaned claims table.

Two kinds of transformation live here:

* Row-wise feature engineering (:func:`engineer_features`, :class:`FeatureEngineer`). Fixed
  formulas with no statistics learned from data, so they cannot leak information. Missing
  inputs are left missing so that flags describe what was actually recorded.
* Imputation (:class:`GroupMedianImputer`). This *does* learn from data (group medians), so
  it only ever runs inside the model pipeline, where it is fit on training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import Config
from src.data.clean import report_delay_days
from src.features.text_rules import extract_text_rules


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


def fractional_years(dates: pd.Series) -> np.ndarray:
    """Dates as decimal years, e.g. 2002-07-02 -> about 2002.5."""
    days_in_year = np.where(dates.dt.is_leap_year, 366, 365)
    return (dates.dt.year + (dates.dt.dayofyear - 1) / days_in_year).to_numpy(dtype=float)


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


def as_flag(condition: pd.Series) -> pd.Series:
    """Boolean condition as a 0/1 integer column (missing counts as 0)."""
    return condition.fillna(False).astype(int)


def calendar_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Accident year, month, weekday and hour, weekend and night-shift flags.

    An accident recorded at exactly the suspect hour (00:00) is treated as "time unknown"
    and is not counted as a night-shift accident.
    """
    f = cfg["features"]
    accident = df[cfg.col("accident_date")]
    hour = accident.dt.hour
    time_unknown = hour == cfg["cleaning"]["flags"]["suspect_accident_hour"]
    at_night = (hour > f["night_after_hour"]) | (hour < f["night_before_hour"])
    return pd.DataFrame(
        {
            "report_delay_days": delay_days(df, cfg),
            "accident_year": accident.dt.year,
            "accident_month": accident.dt.month_name().str[:3],
            "accident_weekday": accident.dt.day_name().str[:3],
            "accident_hour": hour,
            "is_weekend": as_flag(accident.dt.weekday.isin(f["weekend_days"])),
            "accident_time_unknown": as_flag(time_unknown),
            "is_night_shift": as_flag(at_night & ~time_unknown),
        },
        index=df.index,
    )


def work_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Wage, hours and part-time features; missing inputs give missing outputs."""
    f, flags = cfg["features"], cfg["cleaning"]["flags"]
    wages, hours, days, part_time = (
        df[column]
        for column in cfg.cols(["weekly_wages", "hours_per_week", "days_per_week", "part_time"])
    )
    return pd.DataFrame(
        {
            "log_weekly_wages": np.log1p(wages),
            "hourly_wage": wages / hours.clip(lower=f["min_hours_for_rate"]),
            "hours_per_day": hours / days,
            "is_part_time": as_flag(part_time == cfg["data"]["codes"]["part_time"]),
            "wage_is_default": as_flag(wages.isin(flags["heaped_wage_values"])),
        },
        index=df.index,
    )


def claimant_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Age band and total dependents (marital status is used as cleaned, with U = unknown)."""
    return pd.DataFrame(
        {
            "age_band": age_band(df, cfg).astype(str),
            "total_dependents": total_dependents(df, cfg),
        },
        index=df.index,
    )


def estimate_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Features of the adjuster's initial estimate, alone and relative to the weekly wage."""
    f, flags = cfg["features"], cfg["cleaning"]["flags"]
    initial = df[cfg.col("initial_estimate")]
    return pd.DataFrame(
        {
            "log_initial_estimate": np.log1p(initial),
            "initial_to_wage_ratio": initial / df[cfg.col("weekly_wages")],
            "initial_is_default": as_flag(initial.isin(f["initial_default_values"])),
            "initial_is_nominal": as_flag(initial <= flags["nominal_initial_estimate_max"]),
        },
        index=df.index,
    )


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add every row-wise engineered feature, including the keyword text rules.

    Args:
        df: Cleaned claims (output of the clean stage, or a new claim for scoring).
        cfg: Loaded project configuration.

    Returns:
        The input columns plus all engineered columns. Nothing is learned from the data.
    """
    parts = [
        calendar_features(df, cfg),
        work_features(df, cfg),
        claimant_features(df, cfg),
        estimate_features(df, cfg),
        extract_text_rules(df[cfg.col("description")], cfg),
    ]
    return pd.concat([df, *parts], axis=1)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Pipeline step wrapping :func:`engineer_features` (stateless: fit learns nothing)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        """Record the input width; there is nothing else to learn."""
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with the engineered columns added."""
        return engineer_features(X, self.cfg)


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill missing numeric values with the median of the row's group, learned in ``fit``.

    Used with the part-time / full-time group, because hours and wages differ strongly
    between the two. A row whose group is missing or unseen falls back to the overall
    median. Missing-value indicators are added for the chosen raw columns, so the model can
    still tell that a value was imputed.

    Args:
        group_column: Column defining the groups.
        columns: Numeric columns to impute.
        indicator_columns: Columns that also get a 0/1 "was missing" indicator.
        indicator_names: Names of those indicator columns, in the same order.
    """

    def __init__(
        self,
        group_column: str,
        columns: list[str],
        indicator_columns: list[str],
        indicator_names: list[str],
    ) -> None:
        self.group_column = group_column
        self.columns = columns
        self.indicator_columns = indicator_columns
        self.indicator_names = indicator_names

    def fit(self, X: pd.DataFrame, y: object = None) -> GroupMedianImputer:
        """Learn per-group and overall medians from the (training) rows passed in."""
        self.group_medians_ = X.groupby(self.group_column)[self.columns].median()
        self.overall_medians_ = X[self.columns].median()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add missing indicators, then fill gaps with the learned medians."""
        out = X.copy()
        for column, name in zip(self.indicator_columns, self.indicator_names, strict=True):
            out[name] = as_flag(out[column].isna())
        for column in self.columns:
            by_group = out[self.group_column].map(self.group_medians_[column])
            fill = by_group.fillna(self.overall_medians_[column])
            out[column] = out[column].fillna(fill)
        return out
