"""Leakage checks: the target is never an input, and learned steps see training rows only."""

import numpy as np
import pandas as pd
import pytest

from src.features.build import LAYERS, build_preprocessor, feature_names, model_input_columns
from src.features.tabular import engineer_features

ALL_LAYERS_OFF = {layer: False for layer in LAYERS}


def split_small(claims):
    """First 45 rows as 'training', last 15 as 'test' (the fixture is in date order)."""
    return claims.iloc[:45].reset_index(drop=True), claims.iloc[45:].reset_index(drop=True)


@pytest.mark.parametrize("layers", [ALL_LAYERS_OFF, {}])
def test_target_is_never_a_model_input(cfg, layers):
    inputs = model_input_columns(cfg, layers)
    assert cfg.col("target") not in inputs


def test_target_values_do_not_change_the_features(cfg, small_claims):
    train, test = split_small(small_claims)
    pipe = build_preprocessor(cfg).fit(train)
    shuffled = test.copy()
    shuffled[cfg.col("target")] = shuffled[cfg.col("target")].sample(frac=1, random_state=0).values
    np.testing.assert_array_equal(pipe.transform(test), pipe.transform(shuffled))
    assert not any(cfg.col("target") in name for name in feature_names(pipe))


def test_group_medians_are_learned_from_training_rows_only(cfg, small_claims):
    train, test = split_small(small_claims)
    test.loc[:, cfg.col("weekly_wages")] = 1e6  # extreme test values must not move the medians
    pipe = build_preprocessor(cfg).fit(train)
    group = cfg.col(cfg["features"]["impute_group"])
    expected = engineer_features(train, cfg).groupby(group)["log_weekly_wages"].median()
    learned = pipe.named_steps["impute"].group_medians_["log_weekly_wages"]
    assert learned.to_dict() == pytest.approx(expected.to_dict())
    refit = build_preprocessor(cfg).fit(pipe_input(train, test))
    assert refit.named_steps["impute"].group_medians_["log_weekly_wages"].max() > learned.max()


def pipe_input(train, test):
    """Training and test rows together (only used to show that medians would move)."""
    return pd.concat([train, test], ignore_index=True)


def test_scaler_statistics_come_from_training_rows(cfg, small_claims):
    train, _ = split_small(small_claims)
    pipe = build_preprocessor(cfg).fit(train)
    scaler = pipe.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    age_position = cfg.resolve(cfg["features"]["numeric"]).index(cfg.col("age"))
    assert scaler.mean_[age_position] == pytest.approx(train[cfg.col("age")].mean())


def test_keyword_label_seen_only_in_test_gets_no_column(cfg, small_claims):
    train, test = split_small(small_claims)
    test.loc[:, cfg.col("description")] = "AMPUTATED LEFT FINGER"
    pipe = build_preprocessor(cfg).fit(train)
    assert not any("amputation" in name for name in feature_names(pipe))
    matrix = pipe.transform(test)  # unseen labels are handled, not an error
    assert matrix.shape[1] == len(feature_names(pipe))


def test_high_cost_threshold_uses_training_rows_only(cfg, small_claims):
    from src.models.train_classifier import cost_threshold, labels

    target = cfg.col("target")
    claims = small_claims.copy()
    claims.loc[:, target] = np.arange(1, len(claims) + 1) * 100.0
    train, test = split_small(claims)
    test.loc[:, target] = test[target] * 100  # a much pricier future must not move the cutoff
    threshold = cost_threshold(train, cfg)
    quantile = cfg["labels"]["high_cost_quantile"]
    assert threshold == pytest.approx(np.quantile(train[target], quantile))
    assert labels(test, threshold, cfg).mean() > labels(train, threshold, cfg).mean()


def test_relative_label_is_computed_within_each_year(cfg, small_claims):
    from src.models.train_classifier import relative_labels

    claims = small_claims.copy()
    claims.loc[:, cfg.col("target")] = np.arange(1, len(claims) + 1) * 100.0
    flagged = relative_labels(claims, cfg)
    assert 0 < flagged.mean() <= 1 - cfg["labels"]["high_cost_quantile"] + 0.05
