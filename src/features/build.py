"""The full preprocessing pipeline: cleaned claims in, model-ready matrix out.

Steps, in order:

1. ``engineer``: row-wise features and the keyword text rules (learns nothing).
2. ``impute``: part-time / full-time group medians plus missing indicators (learned).
3. ``columns``: a ColumnTransformer that scales numerics, log-transforms skewed ones and
   one-hot encodes categoricals, including the keyword-rule labels.

Every learned step sits inside this one scikit-learn Pipeline, so calling ``fit`` on
training rows is the only way anything is learned; cross-validation refits it per fold.
Whether the keyword-rule features are included is controlled by ``text_layers`` in the config.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import Config
from src.features.tabular import FeatureEngineer, GroupMedianImputer

LAYERS = ("rules",)
# Output-column prefixes of each text-feature switch (ColumnTransformer branch names).
LAYER_PREFIXES = {"rules": ("rules_categorical__", "rules_numeric__")}


def log1p_nonnegative(values: np.ndarray) -> np.ndarray:
    """log(1 + x) with negatives clipped to 0 (skewed, non-negative features)."""
    return np.log1p(np.clip(values, 0, None))


def active_layers(cfg: Config, overrides: dict[str, bool] | None = None) -> dict[str, bool]:
    """Text-layer switches from the config, optionally overridden (for the ablation)."""
    layers = {name: bool(cfg["text_layers"][name]) for name in LAYERS}
    layers.update(overrides or {})
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise ValueError(f"Unknown text layers: {sorted(unknown)}")
    return layers


def feature_groups(cfg: Config, layers: dict[str, bool]) -> dict[str, list[str]]:
    """Model input columns per preprocessing branch, resolved to real column names."""
    f = cfg["features"]
    groups = {
        "numeric": cfg.resolve(f["numeric"]),
        "log": cfg.resolve(f["log_numeric"]),
        "binary": cfg.resolve(f["binary"]),
        "categorical": cfg.resolve(f["categorical"]),
    }
    if layers["rules"]:
        groups["rules_categorical"] = list(f["text_rule_categorical"])
        groups["rules_numeric"] = list(f["text_rule_numeric"])
    return groups


def model_input_columns(cfg: Config, layers: dict[str, bool] | None = None) -> list[str]:
    """Every column the ColumnTransformer reads."""
    layers = active_layers(cfg, layers)
    return [c for group in feature_groups(cfg, layers).values() for c in group]


def numeric_pipeline() -> Pipeline:
    """Median-impute (safety net after group imputation), then standardise."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def log_pipeline() -> Pipeline:
    """Median-impute, log1p to tame right skew, then standardise."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )


def categorical_pipeline(cfg: Config) -> Pipeline:
    """Most-frequent impute, then one-hot with rare levels grouped together."""
    onehot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=cfg["features"]["onehot_min_frequency"],
        sparse_output=False,
    )
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])


def branch_transformers(cfg: Config, layers: dict[str, bool]) -> list[tuple[str, Any, Any]]:
    """(name, transformer, columns) triples for the ColumnTransformer."""
    groups = feature_groups(cfg, layers)
    branches: list[tuple[str, Any, Any]] = [
        ("numeric", numeric_pipeline(), groups["numeric"]),
        ("log", log_pipeline(), groups["log"]),
        ("binary", "passthrough", groups["binary"]),
        ("categorical", categorical_pipeline(cfg), groups["categorical"]),
    ]
    if layers["rules"]:
        branches += [
            ("rules_categorical", categorical_pipeline(cfg), groups["rules_categorical"]),
            ("rules_numeric", numeric_pipeline(), groups["rules_numeric"]),
        ]
    return branches


def build_imputer(cfg: Config) -> GroupMedianImputer:
    """Group-median imputer for every numeric model input, with raw-column indicators."""
    f = cfg["features"]
    raw = f["impute_columns"]
    return GroupMedianImputer(
        group_column=cfg.col(f["impute_group"]),
        columns=cfg.resolve(f["numeric"] + f["log_numeric"]),
        indicator_columns=cfg.cols(raw),
        indicator_names=[f"{name}{f['missing_indicator_suffix']}" for name in raw],
    )


def build_preprocessor(cfg: Config, layers: dict[str, bool] | None = None) -> Pipeline:
    """Assemble the full preprocessing pipeline.

    Args:
        cfg: Loaded project configuration.
        layers: Optional overrides of the ``text_layers`` switches, e.g. ``{"rules": False}``.

    Returns:
        An unfitted Pipeline mapping cleaned claims to a dense feature matrix.
    """
    columns = ColumnTransformer(
        branch_transformers(cfg, active_layers(cfg, layers)),
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("engineer", FeatureEngineer(cfg)),
            ("impute", build_imputer(cfg)),
            ("columns", columns),
        ]
    )


def feature_names(preprocessor: Pipeline) -> list[str]:
    """Output column names of a fitted preprocessor, e.g. ``numeric__Age``."""
    return list(preprocessor.named_steps["columns"].get_feature_names_out())


def select_columns(
    names: list[str], layers: dict[str, bool] | None = None, drop_prefixes: tuple[str, ...] = ()
) -> np.ndarray:
    """Positions of the output columns to keep for a text-feature subset.

    Every ColumnTransformer branch is fit independently, so dropping the keyword-rule columns
    from the full matrix gives exactly the matrix a pipeline without them would produce.

    Args:
        names: Output column names of a fitted preprocessor.
        layers: Text-feature switches (missing keys count as kept). ``None`` keeps everything.
        drop_prefixes: Further column-name prefixes to drop, e.g. one feature's one-hot columns.

    Returns:
        Sorted column positions.
    """
    dropped = tuple(drop_prefixes)
    for layer, keep in (layers or {}).items():
        if not keep:
            dropped += LAYER_PREFIXES[layer]
    return np.array([i for i, name in enumerate(names) if not name.startswith(dropped)], dtype=int)
