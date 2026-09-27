"""Optuna search over model settings, scored by time-ordered cross-validation only.

The objective is the mean validation RMSLE across the folds inside the training period,
so the test set plays no part in tuning. The sampler is seeded and trials run one at a
time, which makes the search reproducible.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

LOG_SCALE = ("learning_rate", "min_child_samples", "reg_lambda", "min_samples_leaf", "alpha")
INTEGER = ("n_estimators", "num_leaves", "min_child_samples", "min_samples_leaf")


@dataclass
class TuningResult:
    """Outcome of a search: best settings, their CV score and every trial."""

    best_params: dict[str, Any]
    best_score: float
    trials: pd.DataFrame


def suggest(trial: optuna.Trial, space: dict[str, list[float]]) -> dict[str, Any]:
    """Draw one set of settings from the configured search space."""
    params: dict[str, Any] = {}
    for name, (low, high) in space.items():
        log = name in LOG_SCALE
        if name in INTEGER:
            params[name] = trial.suggest_int(name, int(low), int(high), log=log)
        else:
            params[name] = trial.suggest_float(name, float(low), float(high), log=log)
    return params


def tune(score: Callable[[dict[str, Any]], float], family: str, cfg: Config) -> TuningResult:
    """Minimise ``score(params)`` (CV RMSLE) with a seeded TPE sampler.

    Args:
        score: Maps a settings dict to the mean cross-validated RMSLE.
        family: Model family whose search space (``tuning.space.<family>``) is used.
        cfg: Loaded project configuration (``tuning`` section).

    Returns:
        The best settings, their score and a table of all trials.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t = cfg["tuning"]
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.random_seed)
    )

    def objective(trial: optuna.Trial) -> float:
        value = score(suggest(trial, t["space"][family]))
        logger.info("Trial %d/%d: CV RMSLE %.4f", trial.number + 1, t["n_trials"], value)
        return value

    study.optimize(objective, n_trials=t["n_trials"])
    trials = pd.DataFrame(
        [{"trial": trial.number, "cv_rmsle": trial.value, **trial.params} for trial in study.trials]
    )
    return TuningResult(study.best_params, float(study.best_value), trials)
