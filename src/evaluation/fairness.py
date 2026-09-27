"""Phase 7: fairness checks across gender and age bands, and the model with and without gender.

The checks compare, per group, how accurate the cost model is (bias and log error) and how
often the high-cost flag fires, against how often claims in the group really are high-cost.
Differences are reported neutrally: some differences in flag rates are expected when true
high-cost rates differ; what matters is whether errors are systematically worse for a group.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.metrics import log_errors, top_positions
from src.features.tabular import age_band
from src.models.train_regression import Candidate, Evaluator, costs


def flag_frame(
    frame: pd.DataFrame,
    typical: np.ndarray,
    expected: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cost_threshold: float,
    cfg: Config,
) -> pd.DataFrame:
    """Per-claim actuals, predictions, flags and the grouping columns."""
    y = costs(frame, cfg)
    capacity_flag = np.zeros(len(y), dtype=bool)
    capacity_flag[top_positions(probabilities, cfg["evaluation"]["review_capacity"])] = True
    return pd.DataFrame(
        {
            "gender": frame[cfg.col("gender")].to_numpy(),
            "age_band": age_band(frame, cfg).astype(str).to_numpy(),
            "actual": y,
            "typical": typical,
            "expected": expected,
            "high_cost": y >= cost_threshold,
            "flagged": probabilities >= threshold,
            "flagged_at_capacity": capacity_flag,
        }
    )


def group_fairness(frame: pd.DataFrame, by: str, min_size: int) -> pd.DataFrame:
    """Error and flag metrics per group (groups smaller than ``min_size`` are left out)."""
    frame = frame.assign(_log_error=log_errors(frame["actual"], frame["typical"]))
    rows = []
    for group, part in frame.groupby(by, observed=True):
        if len(part) < min_size:
            continue
        positives = part["high_cost"]
        rows.append(
            {
                by: group,
                "claims": len(part),
                "mean_actual": part["actual"].mean(),
                "mean_expected": part["expected"].mean(),
                "bias_ratio": part["expected"].sum() / part["actual"].sum(),
                "rmsle": float(np.sqrt(np.mean(part["_log_error"] ** 2))),
                "mean_log_error": part["_log_error"].mean(),
                "high_cost_rate": positives.mean(),
                "flag_rate": part["flagged"].mean(),
                "capacity_flag_rate": part["flagged_at_capacity"].mean(),
                "recall": (
                    part.loc[positives, "flagged_at_capacity"].mean() if positives.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def gender_comparison(final: Candidate, evaluator: Evaluator, cfg: Config) -> dict[str, Any]:
    """The final model retrained with and without gender: CV, test and per-gender test bias."""
    split = evaluator.data.final
    genders = split.eval_frame[cfg.col("gender")].to_numpy()
    y = costs(split.eval_frame, cfg)
    results = {}
    for label, include in (("with_gender", True), ("without_gender", False)):
        variant = replace(
            final, name=f"{final.name} ({label.replace('_', ' ')})", include_gender=include
        )
        metrics, prediction = evaluator.test(variant)
        errors = log_errors(y, prediction.typical)
        per_gender = {
            str(g): {
                "bias_ratio": float(
                    prediction.expected[genders == g].sum() / y[genders == g].sum()
                ),
                "mean_log_error": float(errors[genders == g].mean()),
            }
            for g in pd.unique(genders)
            if (genders == g).sum() >= cfg["fairness"]["min_group_size"]
        }
        results[label] = {
            "cv": {k: v for k, v in evaluator.cv(variant).items() if k != "folds"},
            "test": metrics,
            "by_gender": per_gender,
        }
    change = results["without_gender"]["cv"]["rmsle"] - results["with_gender"]["cv"]["rmsle"]
    results["cv_rmsle_change_without_gender"] = float(change)
    results["negligible"] = bool(abs(change) < cfg["fairness"]["negligible_rmsle_change"])
    return results
