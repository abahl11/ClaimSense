"""Test-set error analysis of the final cost model: where is it accurate, where does it miss?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.metrics import log_errors
from src.features.tabular import accident_year, engineer_features, wage_band
from src.features.text_rules import BODY_PART, INJURY_NATURE

SIZE_GROUP = "size_group"
MISSING_BAND = "missing"


def size_groups(actual: np.ndarray, quantiles: list[float]) -> pd.Series:
    """Label each claim by its actual-cost band, e.g. "Top 1%", "Top 1-10%", "Bottom 90%".

    Args:
        actual: Actual costs of the claims being analysed.
        quantiles: Increasing cut points such as ``[0.90, 0.99]``.
    """
    ordered = sorted(quantiles)
    actual = np.asarray(actual, dtype=float)
    labels = np.full(len(actual), f"Bottom {ordered[0] * 100:g}%", dtype=object)
    for i, (q, cutoff) in enumerate(zip(ordered, np.quantile(actual, ordered), strict=True)):
        top = (1 - q) * 100
        if i + 1 < len(ordered):
            label = f"Top {(1 - ordered[i + 1]) * 100:g}-{top:g}%"
        else:
            label = f"Top {top:g}%"
        labels[actual >= cutoff] = label
    return pd.Series(labels)


def analysis_frame(
    frame: pd.DataFrame, actual: np.ndarray, typical: np.ndarray, expected: np.ndarray, cfg: Config
) -> pd.DataFrame:
    """Predictions alongside the grouping columns used in the error tables."""
    engineered = engineer_features(frame, cfg)
    return pd.DataFrame(
        {
            "actual": actual,
            "typical": typical,
            "expected": expected,
            BODY_PART: engineered[BODY_PART].to_numpy(),
            INJURY_NATURE: engineered[INJURY_NATURE].to_numpy(),
            "wage_band": wage_band(frame, cfg).astype(str).replace("nan", MISSING_BAND).to_numpy(),
            "accident_year": accident_year(frame, cfg).to_numpy(),
            SIZE_GROUP: size_groups(
                actual, cfg["evaluation"]["largest_claim_quantiles"]
            ).to_numpy(),
        }
    )


def rms(values: pd.Series) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(np.square(values))))


def group_errors(frame: pd.DataFrame, by: str, min_size: int = 0) -> pd.DataFrame:
    """Claims, mean actual/expected cost, total bias, MAE and log errors per group."""
    frame = frame.assign(
        _abs_error=(frame["expected"] - frame["actual"]).abs(),
        _log_error=log_errors(frame["actual"], frame["typical"]),
    )
    grouped = frame.groupby(by, observed=True)
    table = pd.DataFrame(
        {
            "claims": grouped.size(),
            "mean_actual": grouped["actual"].mean(),
            "mean_expected": grouped["expected"].mean(),
            "bias_ratio": grouped["expected"].sum() / grouped["actual"].sum(),
            "mae": grouped["_abs_error"].mean(),
            "rmsle": grouped["_log_error"].apply(rms),
            "mean_log_error": grouped["_log_error"].mean(),
        }
    )
    return table[table["claims"] >= min_size].reset_index()


def band_sort_key(label: str) -> tuple[bool, float]:
    """Order band labels such as '<200', '200-300', '1,500+' numerically, 'missing' last."""
    if label == MISSING_BAND:
        return True, 0.0
    if label.startswith("<"):
        return False, -1.0
    return False, float(label.split("-")[0].rstrip("+").replace(",", ""))


def error_tables(frame: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Grouped error tables for the error-analysis section of the results report."""
    min_size = cfg["evaluation"]["min_group_size"]
    tables = {
        field: group_errors(frame, field, min_size).sort_values("claims", ascending=False)
        for field in (BODY_PART, INJURY_NATURE)
    }
    bands = group_errors(frame, "wage_band")
    tables["wage_band"] = bands.iloc[
        sorted(range(len(bands)), key=lambda i: band_sort_key(bands["wage_band"].iloc[i]))
    ]
    tables["accident_year"] = group_errors(frame, "accident_year")
    tables[SIZE_GROUP] = group_errors(frame, SIZE_GROUP).sort_values("mean_actual", ascending=False)
    return {name: table.reset_index(drop=True) for name, table in tables.items()}


def bias_by_year(
    frame: pd.DataFrame, cfg: Config, actual: np.ndarray, expected: np.ndarray
) -> dict[int, float]:
    """Total predicted / total actual cost for each accident year."""
    years = accident_year(frame, cfg).to_numpy()
    totals = (
        pd.DataFrame({"year": years, "actual": actual, "expected": expected}).groupby("year").sum()
    )
    return {int(year): float(row.expected / row.actual) for year, row in totals.iterrows()}
