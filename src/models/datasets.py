"""Model-ready matrices for every cross-validation fold and for the final train/test split.

The preprocessor is fit once per split, on that split's training rows only, and the
transformed matrices are shared by every model evaluated on the split. This is safe because
preprocessing never looks at the target: medians, scalers, one-hot levels, TF-IDF, SVD and
KMeans are all unsupervised. Only the regressors and classifiers use the target, and each
of those is fit on the split's training rows alone.

Matrices always contain every text layer; ablations drop columns
(:func:`src.features.build.select_columns`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import Config
from src.data.load import load_clean
from src.features.build import build_preprocessor, feature_names
from src.models.split import recent_mask, time_ordered_folds, train_test_split_by_time

logger = logging.getLogger(__name__)

FINGERPRINT_SECTIONS = (
    "data",
    "cleaning",
    "split",
    "features",
    "text_layers",
    "text_rules",
    "tfidf",
    "embeddings",
)


@dataclass
class Split:
    """One fit/evaluate pair of preprocessed matrices, with the matching cleaned rows."""

    name: str
    X_fit: np.ndarray
    X_eval: np.ndarray
    fit_frame: pd.DataFrame
    eval_frame: pd.DataFrame
    feature_names: list[str]
    preprocessor: Pipeline | None = None


@dataclass
class Datasets:
    """Cross-validation folds (training period only) and the final train -> test split."""

    folds: list[Split]
    final: Split
    cutoff: pd.Timestamp


def make_split(
    fit_rows: pd.DataFrame, eval_rows: pd.DataFrame, cfg: Config, name: str, keep: bool = False
) -> Split:
    """Fit the preprocessor on ``fit_rows`` and transform both row sets.

    Args:
        fit_rows: Cleaned claims the preprocessor (and later the model) learns from.
        eval_rows: Cleaned claims to score.
        cfg: Loaded project configuration.
        name: Label for logs and reports.
        keep: Keep the fitted preprocessor (needed only for the final split).
    """
    fit_rows, eval_rows = fit_rows.reset_index(drop=True), eval_rows.reset_index(drop=True)
    preprocessor = build_preprocessor(cfg)
    X_fit = preprocessor.fit_transform(fit_rows).astype(np.float32)
    X_eval = preprocessor.transform(eval_rows).astype(np.float32)
    logger.info(
        "%s: %d fit rows, %d eval rows, %d columns",
        name,
        len(fit_rows),
        len(eval_rows),
        X_fit.shape[1],
    )
    return Split(
        name=name,
        X_fit=X_fit,
        X_eval=X_eval,
        fit_frame=fit_rows,
        eval_frame=eval_rows,
        feature_names=feature_names(preprocessor),
        preprocessor=preprocessor if keep else None,
    )


def inner_split(split: Split, cfg: Config) -> Split:
    """Divide a split's training rows into an earlier part and the most recent part.

    The most recent ``split.inner_holdout_fraction`` of the training rows becomes the
    evaluation part. The already-transformed matrix is reused: its preprocessing saw these
    rows' features but never their targets, and a model fit on the earlier part has never
    seen them at all.
    """
    dates = split.fit_frame[cfg.col(cfg["split"]["date_column"])]
    late = recent_mask(dates, cfg["split"]["inner_holdout_fraction"]).to_numpy()
    return Split(
        name=f"{split.name} (inner holdout)",
        X_fit=split.X_fit[~late],
        X_eval=split.X_fit[late],
        fit_frame=split.fit_frame.loc[~late].reset_index(drop=True),
        eval_frame=split.fit_frame.loc[late].reset_index(drop=True),
        feature_names=split.feature_names,
    )


def build_datasets(cfg: Config) -> Datasets:
    """Preprocess every cross-validation fold and the final train/test split."""
    train, test, cutoff = train_test_split_by_time(load_clean(cfg), cfg)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    dates = train[cfg.col(cfg["split"]["date_column"])]
    folds = [
        make_split(train.iloc[fit], train.iloc[evaluate], cfg, f"fold {k}")
        for k, (fit, evaluate) in enumerate(time_ordered_folds(dates, cfg["split"]["cv_folds"]), 1)
    ]
    final = make_split(train, test, cfg, "final (train -> test)", keep=True)
    return Datasets(folds=folds, final=final, cutoff=cutoff)


def fingerprint(cfg: Config) -> str:
    """Hash of the cleaned data file and every config section that affects the matrices."""
    path = cfg.path("clean_data")
    stat = path.stat()
    sections = {name: cfg[name] for name in FINGERPRINT_SECTIONS}
    payload = json.dumps(
        [str(path), stat.st_size, stat.st_mtime_ns, sections], sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CACHE: dict[str, Datasets] = {}


def get_datasets(cfg: Config) -> Datasets:
    """Datasets for this config, built once per process (stages run together share them)."""
    key = fingerprint(cfg)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = build_datasets(cfg)
    return _CACHE[key]
