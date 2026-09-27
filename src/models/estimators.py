"""Factories for the regression and classification estimators, configured from the config."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from src.config import Config

REGRESSION_FAMILIES = ("ridge", "random_forest", "lightgbm", "tweedie", "gamma", "quantile")


def lightgbm_params(cfg: Config, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default LightGBM settings, shared settings and the seed, with optional overrides."""
    r = cfg["regression"]
    return {
        **r["lightgbm"],
        **r["lightgbm_common"],
        "random_state": cfg.random_seed,
        "n_jobs": r["n_jobs"],
        **(overrides or {}),
    }


def make_regressor(family: str, cfg: Config, params: dict[str, Any] | None = None) -> Any:
    """Build an unfitted regressor.

    Args:
        family: One of ``REGRESSION_FAMILIES``.
        cfg: Loaded project configuration.
        params: Overrides, e.g. tuned LightGBM settings, ``tweedie_variance_power`` or
            ``alpha`` for a quantile model.

    Raises:
        ValueError: For an unknown family.
    """
    r, params = cfg["regression"], dict(params or {})
    if family == "ridge":
        return Ridge(**{**r["ridge"], **params})
    if family == "random_forest":
        rf = {**r["random_forest"], **params}
        return RandomForestRegressor(**rf, random_state=cfg.random_seed, n_jobs=r["n_jobs"])
    objectives = {
        "lightgbm": "regression",
        "tweedie": "tweedie",
        "gamma": "gamma",
        "quantile": "quantile",
    }
    if family in objectives:
        return LGBMRegressor(objective=objectives[family], **lightgbm_params(cfg, params))
    raise ValueError(f"Unknown regression family: {family}")


def make_classifier(
    family: str, cfg: Config, positive_weight: float, params: dict[str, Any] | None = None
) -> Any:
    """Build an unfitted classifier.

    Args:
        family: ``logistic`` or ``lightgbm``.
        cfg: Loaded project configuration.
        positive_weight: Weight of the positive class (negatives / positives for "balanced").
            Used by LightGBM to handle the class imbalance; logistic regression is left
            unweighted so that its probabilities stay close to calibrated.
        params: LightGBM overrides (for example the tuned regression settings).

    Raises:
        ValueError: For an unknown family.
    """
    if family == "logistic":
        return LogisticRegression(**cfg["classifier"]["logistic"])
    if family == "lightgbm":
        settings = lightgbm_params(cfg, params)
        return LGBMClassifier(**settings, scale_pos_weight=positive_weight)
    raise ValueError(f"Unknown classifier family: {family}")
