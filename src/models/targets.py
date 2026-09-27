"""Target transformations: what the regression model is trained to predict.

Four ways of handling the rise in claim costs over time (EDA section 5) are compared:

* **A, log cost**: ``z = log(1 + cost)``. The model must learn the price level from the
  features (initial estimate, accident year), and trees cannot extrapolate past the last
  training year.
* **B, log ratio**: ``z = log(cost / initial estimate)``. The adjuster's estimate is recorded
  at current prices, so the model only has to learn how far the final cost ends up from it.
* **C, trend-adjusted log cost**: a log-linear trend fitted on the most recent training years
  brings every claim to the price level of the latest training date; predictions are moved
  back out by the same trend, so costs keep rising past the training period.
* **D, trend-adjusted log ratio**: B and C combined, for drift in the ratio itself.

Plus ``raw`` for losses that work on the cost directly (Tweedie, Gamma).

Every transform exposes ``fit`` (learns from training rows only), ``forward`` (cost to model
target) and ``inverse`` (model output back to cost). ``inverse`` of a log target gives the
*typical* (median-type) cost; adding ``log(smearing factor)`` first gives the *expected*
(mean) cost, see :func:`smearing_factor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.features.tabular import fractional_years

LOG_TARGETS = ("log", "log_ratio", "log_trend", "log_ratio_trend")
APPROACHES = {
    "log": "A: log cost",
    "log_ratio": "B: log ratio to initial estimate",
    "log_trend": "C: log cost, trend-adjusted",
    "log_ratio_trend": "D: log ratio, trend-adjusted",
}


class RawCost:
    """The cost itself (for Tweedie and Gamma losses, which predict the mean directly)."""

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> RawCost:
        """Nothing to learn."""
        return self

    def forward(self, frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Cost unchanged."""
        return np.asarray(y, dtype=float)

    def inverse(self, frame: pd.DataFrame, z: np.ndarray) -> np.ndarray:
        """Model output is already a cost."""
        return np.asarray(z, dtype=float)


class LogCost:
    """Approach A: ``log(1 + cost)``."""

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> LogCost:
        """Nothing to learn."""
        return self

    def forward(self, frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """``log(1 + cost)``."""
        return np.log1p(np.asarray(y, dtype=float))

    def inverse(self, frame: pd.DataFrame, z: np.ndarray) -> np.ndarray:
        """``exp(z) - 1``."""
        return np.expm1(np.asarray(z, dtype=float))


class LogRatio:
    """Approach B: ``log(cost / initial estimate)``.

    Args:
        initial_column: Column holding the adjuster's initial estimate (always >= 1).
    """

    def __init__(self, initial_column: str) -> None:
        self.initial_column = initial_column

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> LogRatio:
        """Nothing to learn."""
        return self

    def _log_initial(self, frame: pd.DataFrame) -> np.ndarray:
        return np.log(frame[self.initial_column].to_numpy(dtype=float))

    def forward(self, frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """``log(cost) - log(initial estimate)``."""
        return np.log(np.asarray(y, dtype=float)) - self._log_initial(frame)

    def inverse(self, frame: pd.DataFrame, z: np.ndarray) -> np.ndarray:
        """``initial estimate * exp(z)``."""
        return np.exp(np.asarray(z, dtype=float) + self._log_initial(frame))


class Trended:
    """Approaches C and D: remove a log-linear cost trend fitted on recent training claims.

    The trend's slope is estimated by least squares of the base target on the accident date
    (in decimal years) over the last ``window_years`` of the training rows. Targets are then
    expressed at the price level of the latest training date, and predictions are moved back
    to each claim's own date, so the trend carries on into the future.

    Args:
        base: The transform the trend is applied to (``LogCost`` or ``LogRatio``).
        date_column: Accident date column.
        window_years: Length of the recent window used to fit the slope.
    """

    def __init__(self, base: LogCost | LogRatio, date_column: str, window_years: float) -> None:
        self.base = base
        self.date_column = date_column
        self.window_years = window_years

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> Trended:
        """Fit the slope on the most recent ``window_years`` of training rows."""
        years = fractional_years(frame[self.date_column])
        target = self.base.forward(frame, y)
        recent = years >= years.max() - self.window_years
        self.slope_ = float(np.polyfit(years[recent], target[recent], 1)[0])
        self.reference_year_ = float(years.max())
        return self

    def _trend(self, frame: pd.DataFrame) -> np.ndarray:
        return self.slope_ * (fractional_years(frame[self.date_column]) - self.reference_year_)

    def forward(self, frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Base target at the price level of the latest training date."""
        return self.base.forward(frame, y) - self._trend(frame)

    def inverse(self, frame: pd.DataFrame, z: np.ndarray) -> np.ndarray:
        """Move the prediction to the claim's own date, then invert the base transform."""
        return self.base.inverse(frame, np.asarray(z, dtype=float) + self._trend(frame))


def make_target(name: str, cfg: Config) -> RawCost | LogCost | LogRatio | Trended:
    """Build an (unfitted) target transform by name: raw, log, log_ratio, log_trend, ...

    Raises:
        ValueError: For an unknown name.
    """
    initial, accident = cfg.cols(["initial_estimate", "accident_date"])
    window = cfg["regression"]["trend_window_years"]
    builders = {
        "raw": RawCost,
        "log": LogCost,
        "log_ratio": lambda: LogRatio(initial),
        "log_trend": lambda: Trended(LogCost(), accident, window),
        "log_ratio_trend": lambda: Trended(LogRatio(initial), accident, window),
    }
    if name not in builders:
        raise ValueError(f"Unknown target transform: {name}")
    return builders[name]()


def smearing_factor(residuals: np.ndarray) -> float:
    """Duan's smearing estimate: the mean of ``exp(residual)`` on the log scale.

    A model trained on log cost predicts the average *log* cost, and exponentiating that
    gives something close to the median, which is below the mean for skewed costs.
    Multiplying by ``mean(exp(residuals))`` corrects this retransformation bias without
    assuming the residuals are normal. Residuals must come from claims the model did not
    see in training, otherwise they are too small and so is the correction.
    """
    return float(np.mean(np.exp(np.asarray(residuals, dtype=float))))
