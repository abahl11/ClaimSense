"""Evaluation metrics for the cost model and the high-cost classifier.

Regression. Two kinds of prediction are scored:

* the *typical* (median-type) prediction, used for log-scale metrics. On the log scale every
  claim counts equally in relative terms, so a few huge claims cannot dominate;
* the *expected* (mean-type) prediction, used for currency metrics. Reserves must add up, so
  the mean is what the business books. Note that RMSE in currency is dominated by the
  largest claims: one claim that is off by 1,000,000 outweighs thousands of small ones.

Classification. Ranking metrics (PR-AUC, ROC-AUC, capacity-based recall) plus threshold
metrics and probability calibration (Brier score, reliability table).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_gamma_deviance,
    roc_auc_score,
)


def log_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Prediction minus actual on the log1p scale (negative = under-prediction)."""
    return np.log1p(np.asarray(y_pred, dtype=float)) - np.log1p(np.asarray(y_true, dtype=float))


def log_metrics(y_true: np.ndarray, y_typical: np.ndarray) -> dict[str, float]:
    """RMSE, R-squared and mean error on the log1p scale.

    Args:
        y_true: Actual ultimate costs.
        y_typical: Typical (median-type) predicted costs.

    Returns:
        ``rmsle``, ``r2_log`` and ``mean_log_error`` (average log bias).
    """
    errors = log_errors(y_true, y_typical)
    log_true = np.log1p(np.asarray(y_true, dtype=float))
    total = np.sum((log_true - log_true.mean()) ** 2)
    return {
        "rmsle": float(np.sqrt(np.mean(errors**2))),
        "r2_log": float(1 - np.sum(errors**2) / total),
        "mean_log_error": float(np.mean(errors)),
    }


def regression_metrics(
    y_true: np.ndarray, y_typical: np.ndarray, y_expected: np.ndarray
) -> dict[str, float]:
    """Full regression metric set: log-scale metrics plus currency metrics.

    Args:
        y_true: Actual ultimate costs.
        y_typical: Typical (median-type) predictions, for the log-scale metrics.
        y_expected: Expected (mean-type) predictions, for the currency metrics.

    Returns:
        Log metrics plus ``mae``, ``rmse``, ``bias_ratio`` (total predicted / total actual)
        and ``gamma_deviance`` (a relative-error score for mean predictions).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_expected = np.asarray(y_expected, dtype=float)
    errors = y_expected - y_true
    return {
        **log_metrics(y_true, y_typical),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias_ratio": float(y_expected.sum() / y_true.sum()),
        "gamma_deviance": float(mean_gamma_deviance(y_true, np.clip(y_expected, 1e-9, None))),
    }


def decile_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Mean predicted and mean actual cost within each predicted-cost decile.

    Claims are sorted by prediction and cut into ``n_bins`` groups of (almost) equal size.
    A well-calibrated model has mean predicted close to mean actual in every group.
    """
    frame = pd.DataFrame(
        {"actual": np.asarray(y_true, float), "predicted": np.asarray(y_pred, float)}
    )
    ranks = frame["predicted"].rank(method="first")
    frame["decile"] = pd.qcut(ranks, n_bins, labels=range(1, n_bins + 1)).astype(int)
    table = frame.groupby("decile").agg(
        claims=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        mean_actual=("actual", "mean"),
    )
    table["ratio"] = table["mean_predicted"] / table["mean_actual"]
    return table.reset_index()


def top_positions(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Positions of the highest-scoring ``fraction`` of rows (ties broken by row order)."""
    n_review = math.ceil(len(scores) * fraction)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return order[:n_review]


def capacity_table(y_true: np.ndarray, scores: np.ndarray, capacities: list[float]) -> pd.DataFrame:
    """Recall and precision when reviewing only the top-scored share of claims.

    Args:
        y_true: 1 for high-cost claims, 0 otherwise.
        scores: Ranking scores (higher = more likely high-cost).
        capacities: Shares of claims the review team can look at, e.g. ``[0.05, 0.1, 0.2]``.

    Returns:
        One row per capacity with claims reviewed, high-cost claims caught, recall,
        precision and lift over random review.
    """
    y_true = np.asarray(y_true, dtype=int)
    positives, base_rate = y_true.sum(), y_true.mean()
    rows = []
    for capacity in capacities:
        caught = y_true[top_positions(scores, capacity)]
        rows.append(
            {
                "capacity": capacity,
                "reviewed": len(caught),
                "caught": int(caught.sum()),
                "recall": float(caught.sum() / positives) if positives else float("nan"),
                "precision": float(caught.mean()),
                "lift": float(caught.mean() / base_rate) if base_rate else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    """Ranking metrics, plus precision / recall / flag rate at a threshold if given.

    Args:
        y_true: 1 for high-cost claims, 0 otherwise.
        scores: Scores or probabilities (higher = more likely high-cost).
        threshold: Optional decision threshold on ``scores``.

    Returns:
        ``pr_auc``, ``roc_auc``, ``base_rate`` and, with a threshold, ``precision``,
        ``recall`` and ``flag_rate``.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "base_rate": float(y_true.mean()),
    }
    if threshold is not None:
        flagged = scores >= threshold
        caught = int((flagged & (y_true == 1)).sum())
        metrics.update(
            {
                "threshold": float(threshold),
                "flag_rate": float(flagged.mean()),
                "precision": float(caught / flagged.sum()) if flagged.any() else float("nan"),
                "recall": float(caught / y_true.sum()) if y_true.any() else float("nan"),
            }
        )
    return metrics


def reliability_table(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Observed high-cost rate against mean predicted probability, in equal-count bins."""
    frame = pd.DataFrame(
        {"actual": np.asarray(y_true, int), "probability": np.asarray(probabilities, float)}
    )
    ranks = frame["probability"].rank(method="first")
    frame["bin"] = pd.qcut(ranks, n_bins, labels=range(1, n_bins + 1)).astype(int)
    table = frame.groupby("bin").agg(
        claims=("actual", "size"),
        mean_probability=("probability", "mean"),
        observed_rate=("actual", "mean"),
    )
    return table.reset_index()


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of predicted probabilities (lower is better)."""
    return float(brier_score_loss(np.asarray(y_true, int), np.asarray(probabilities, float)))
